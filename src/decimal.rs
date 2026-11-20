use crate::gpu::GpuContext;
use wgpu::*;

const DECIMAL_SHADER: &str = r#"
struct Config {
    total_blocks: u32,
    blocks_to_run: u32,
    start_block: u32,
    base: u32,
};

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read_write> f_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_digits: array<u32>;
@group(0) @binding(3) var<storage, read_write> state: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = config.base;
    var c = state[0];
    var e = state[1];

    for (var step: u32 = 0u; step < config.blocks_to_run; step = step + 1u) {
        if (c == 0u) {
            break;
        }
        var d: u32 = 0u;
        var g: u32 = c * 2u;
        var b: u32 = c;

        loop {
            if (b == 0u) {
                break;
            }
            d = d + f_buf[b] * base;
            g = g - 1u;
            f_buf[b] = d % g;
            d = d / g;
            g = g - 1u;
            b = b - 1u;
            if (b > 0u) {
                d = d * b;
            }
        }

        c = c - 14u;
        let digit_val = e + d / base;
        e = d % base;
        out_digits[step] = digit_val;
    }

    state[0] = c;
    state[1] = e;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConfigUniform {
    total_blocks: u32,
    blocks_to_run: u32,
    start_block: u32,
    base: u32,
}

pub struct DecimalCalculator {
    total_blocks: u32,
    blocks_per_batch: u32,
    completed_blocks: u32,
    total_digits_calculated: usize,
    base: u32,
    infinite: bool,

    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,

    f_buf: Buffer,
    state_buf: Buffer,
    config_buf: Buffer,
    out_buf: Buffer,
    staging_buf: Buffer,
}

impl DecimalCalculator {
    pub fn new(gpu: &GpuContext, target_digits: usize, batch_digits: usize, infinite: bool) -> Self {
        let base = 10000u32;
        // If infinite, default to 20,000 digits capacity for immediate responsive output
        let effective_digits = if infinite {
            target_digits.max(20_000)
        } else {
            target_digits
        };

        let target_blocks = ((effective_digits.max(4) - 3 + 3) / 4 + 1) as u32;
        let blocks_per_batch = ((batch_digits.max(4) + 3) / 4) as u32;

        let total_c = target_blocks * 14;

        let shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("DecimalPiShader"),
            source: ShaderSource::Wgsl(DECIMAL_SHADER.into()),
        });

        // Initialize state on GPU VRAM
        let f_init: Vec<u32> = vec![base / 5; (total_c + 1) as usize];
        let f_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("f_buf"),
            size: (f_init.len() * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&f_buf, 0, bytemuck::cast_slice(&f_init));

        let state_init: [u32; 2] = [total_c, 0];
        let state_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("state_buf"),
            size: 8,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&state_buf, 0, bytemuck::cast_slice(&state_init));

        let config_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("config_buf"),
            size: std::mem::size_of::<ConfigUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let out_buf_size = (blocks_per_batch as usize * 4) as u64;
        let out_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("out_buf"),
            size: out_buf_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("staging_buf"),
            size: out_buf_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("decimal_bgl"),
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
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("decimal_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let compute_pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("decimal_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            total_blocks: target_blocks,
            blocks_per_batch,
            completed_blocks: 0,
            total_digits_calculated: 0,
            base,
            infinite,

            compute_pipeline,
            bind_group_layout,

            f_buf,
            state_buf,
            config_buf,
            out_buf,
            staging_buf,
        }
    }

    pub fn compute_next_batch(&mut self, gpu: &GpuContext) -> Option<String> {
        if self.completed_blocks >= self.total_blocks {
            return None;
        }

        let blocks_to_run = (self.total_blocks - self.completed_blocks).min(self.blocks_per_batch);
        if blocks_to_run == 0 {
            return None;
        }

        let config = ConfigUniform {
            total_blocks: self.total_blocks,
            blocks_to_run,
            start_block: self.completed_blocks,
            base: self.base,
        };
        gpu.queue.write_buffer(&self.config_buf, 0, bytemuck::bytes_of(&config));

        let bind_group = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("decimal_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.config_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.f_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.out_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.state_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        let copy_size = (blocks_to_run as usize * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging_buf, 0, copy_size);
        gpu.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buf.slice(..copy_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        gpu.device.poll(PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let mut batch_digits = String::with_capacity((blocks_to_run * 4) as usize);
        {
            let data = buffer_slice.get_mapped_range().unwrap();
            let blocks: &[u32] = bytemuck::cast_slice(&data);

            for (i, &block_val) in blocks.iter().enumerate() {
                let global_block = self.completed_blocks + i as u32;
                if global_block == 0 {
                    let d0 = block_val / 1000;
                    let rest = block_val % 1000;
                    batch_digits.push_str(&format!("{}.{:03}", d0, rest));
                    self.total_digits_calculated += 3;
                } else {
                    batch_digits.push_str(&format!("{:04}", block_val));
                    self.total_digits_calculated += 4;
                }
            }
        }
        self.staging_buf.unmap();
        self.completed_blocks += blocks_to_run;

        Some(batch_digits)
    }

    pub fn total_digits(&self) -> usize {
        self.total_digits_calculated
    }

    pub fn is_finished(&self) -> bool {
        if self.infinite {
            false
        } else {
            self.completed_blocks >= self.total_blocks
        }
    }
}
