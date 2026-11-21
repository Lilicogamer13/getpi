use crate::gpu::GpuContext;
use wgpu::*;

// ── Pure-Rust helpers exposed for unit tests (no GPU required) ───────────────

/// The 16 uppercase hex characters used for BBP output.
pub const HEX_CHARS: &[u8; 16] = b"0123456789ABCDEF";

/// Convert a raw BBP digit value (0–15) to its hex character.
/// Panics in debug if `d >= 16`.
#[allow(dead_code)]
pub fn hex_digit_char(d: u32) -> char {
    HEX_CHARS[(d & 0xF) as usize] as char
}

/// Given the index of the first digit in a batch (`start_index`) produce the
/// "3." prefix string that must be prepended to the very first batch.
#[allow(dead_code)]
pub fn bbp_prefix(start_index: usize) -> &'static str {
    if start_index == 0 { "3." } else { "" }
}

/// Trim a BBP batch string so total hex digits (excluding the "3." prefix)
/// never exceeds `max_digits`.
///
/// `digits_before` – hex digits already written before this batch.
#[allow(dead_code)]
pub fn trim_bbp_to_limit<'a>(raw: &'a str, digits_before: usize, max_digits: usize) -> &'a str {
    if max_digits == 0 {
        return "";
    }
    let available = max_digits.saturating_sub(digits_before);
    if available == 0 || raw.is_empty() {
        return "";
    }
    if raw.len() <= available {
        raw
    } else {
        // All chars are ASCII, every boundary is valid.
        &raw[..available]
    }
}

// ─────────────────────────────────────────────────────────────────────────────

const BBP_SHADER: &str = r#"
struct Config {
    start_index: u32,
    count: u32,
};

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read_write> out_digits: array<u32>;

fn mul_mod(a: u32, b: u32, m: u32) -> u32 {
    if (m <= 65535u) {
        return (a * b) % m;
    }
    var res: u32 = 0u;
    var x: u32 = a % m;
    var y: u32 = b;
    while (y > 0u) {
        if ((y & 1u) == 1u) {
            let sum = res + x;
            if (sum >= m || sum < res) {
                res = sum - m;
            } else {
                res = sum;
            }
        }
        let double_x = x + x;
        if (double_x >= m || double_x < x) {
            x = double_x - m;
        } else {
            x = double_x;
        }
        y = y >> 1u;
    }
    return res;
}

fn mod_exp16(exp: u32, m: u32) -> u32 {
    var b = 16u % m;
    var e = exp;
    var r = 1u % m;
    while (e > 0u) {
        if ((e & 1u) == 1u) {
            r = mul_mod(r, b, m);
        }
        b = mul_mod(b, b, m);
        e = e >> 1u;
    }
    return r;
}

// Neumaier compensated summation for double-precision equivalent accuracy in f32
fn series_sum(j: u32, p: u32) -> f32 {
    var s: f32 = 0.0;
    var c: f32 = 0.0;

    // Sum for k <= p
    for (var k: u32 = 0u; k <= p; k = k + 1u) {
        let denom = 8u * k + j;
        let num = mod_exp16(p - k, denom);
        let term = f32(num) / f32(denom);
        let t = s + term;
        if (abs(s) >= abs(term)) {
            c = c + ((s - t) + term);
        } else {
            c = c + ((term - t) + s);
        }
        s = t;
        s = fract(s);
    }

    // Sum for k > p
    var k: u32 = p + 1u;
    var p16: f32 = 1.0 / 16.0;
    for (var step: u32 = 0u; step < 16u; step = step + 1u) {
        let denom = 8u * k + j;
        let term = p16 / f32(denom);
        if (term < 1e-12) {
            break;
        }
        let t = s + term;
        if (abs(s) >= abs(term)) {
            c = c + ((s - t) + term);
        } else {
            c = c + ((term - t) + s);
        }
        s = t;
        s = fract(s);
        p16 = p16 * (1.0 / 16.0);
        k = k + 1u;
    }
    return fract(s + c);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = id.x;
    if (tid >= config.count) {
        return;
    }
    let p = config.start_index + tid;
    let s1 = series_sum(1u, p);
    let s4 = series_sum(4u, p);
    let s5 = series_sum(5u, p);
    let s6 = series_sum(6u, p);

    var s = fract(4.0 * s1 - 2.0 * s4 - s5 - s6);
    if (s < 0.0) {
        s = s + 1.0;
    }
    let digit = u32(s * 16.0) % 16u;
    out_digits[tid] = digit;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BbpConfigUniform {
    start_index: u32,
    count: u32,
}

pub struct BbpCalculator {
    target_digits: usize,
    batch_size: u32,
    completed_digits: usize,
    infinite: bool,

    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,

    config_buf: Buffer,
    out_buf: Buffer,
    staging_buf: Buffer,
}

impl BbpCalculator {
    pub fn new(gpu: &GpuContext, target_digits: usize, batch_size: usize, infinite: bool) -> Self {
        let batch_size = batch_size.max(256) as u32;

        let shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("BbpPiShader"),
            source: ShaderSource::Wgsl(BBP_SHADER.into()),
        });

        let config_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("bbp_config_buf"),
            size: std::mem::size_of::<BbpConfigUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let out_buf_size = (batch_size as usize * 4) as u64;
        let out_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("bbp_out_buf"),
            size: out_buf_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("bbp_staging_buf"),
            size: out_buf_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bbp_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("bbp_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let compute_pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("bbp_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            target_digits,
            batch_size,
            completed_digits: 0,
            infinite,

            compute_pipeline,
            bind_group_layout,

            config_buf,
            out_buf,
            staging_buf,
        }
    }

    pub fn compute_next_batch(&mut self, gpu: &GpuContext) -> Option<String> {
        // Fixed mode: bail out as soon as we have reached the target.
        if !self.infinite && self.completed_digits >= self.target_digits {
            return None;
        }

        // Clamp batch to remaining digits in fixed mode.
        let count = if self.infinite {
            self.batch_size
        } else {
            let remaining = (self.target_digits - self.completed_digits) as u32;
            remaining.min(self.batch_size)
        };

        if count == 0 {
            return None;
        }

        let config = BbpConfigUniform {
            start_index: self.completed_digits as u32,
            count,
        };
        gpu.queue.write_buffer(&self.config_buf, 0, bytemuck::bytes_of(&config));

        let bind_group = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bbp_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.config_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.out_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = (count + 255) / 256;
        let mut encoder = gpu.device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        let copy_size = (count as usize * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging_buf, 0, copy_size);
        gpu.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buf.slice(..copy_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        gpu.device.poll(PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        // Build the output string.
        // Capacity: prefix ("3.") + count hex chars.
        let mut batch_digits = String::with_capacity(count as usize + 2);
        {
            let data = buffer_slice.get_mapped_range().unwrap();
            let digits: &[u32] = bytemuck::cast_slice(&data);

            // Prepend "3." only on the very first batch.
            if self.completed_digits == 0 {
                batch_digits.push_str("3.");
            }

            for &d in digits.iter().take(count as usize) {
                batch_digits.push(hex_digit_char(d));
            }
        }
        self.staging_buf.unmap();

        // Advance counter by exactly `count` (already clamped to remaining).
        self.completed_digits += count as usize;

        Some(batch_digits)
    }

    pub fn total_digits(&self) -> usize {
        self.completed_digits
    }

    pub fn is_finished(&self) -> bool {
        !self.infinite && self.completed_digits >= self.target_digits
    }
}
