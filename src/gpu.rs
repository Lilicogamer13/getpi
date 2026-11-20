use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};
use wgpu::*;

pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let waker = Waker::noop();
    let mut cx = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub info: AdapterInfo,
}

pub fn get_adapters() -> (Instance, Vec<Adapter>) {
    let instance = Instance::default();
    let adapters = block_on(instance.enumerate_adapters(Backends::all()));
    (instance, adapters)
}

pub fn list_adapters() {
    let (_, adapters) = get_adapters();
    println!("Available Compute Adapters (GPUs & Accelerators):");
    if adapters.is_empty() {
        println!("  (No GPU/Vulkan/Metal/DX12 adapters found)");
        return;
    }
    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        let type_str = match info.device_type {
            DeviceType::DiscreteGpu => "Discrete GPU (High Performance)",
            DeviceType::IntegratedGpu => "Integrated GPU",
            DeviceType::Cpu => "CPU Software Rasterizer",
            DeviceType::VirtualGpu => "Virtual GPU",
            DeviceType::Other => "Other Accelerator",
        };
        println!(
            "  [{}] {} | Type: {} | Backend: {:?} | Driver: {}",
            i, info.name, type_str, info.backend, info.driver_info
        );
    }
    println!("\nUse --device <INDEX> or -d <INDEX> to select a specific GPU.");
}

pub fn init_gpu(device_index: Option<usize>) -> Result<GpuContext, String> {
    let (_, adapters) = get_adapters();
    if adapters.is_empty() {
        return Err("No compatible GPU / compute adapter found on this system.".into());
    }

    let selected_adapter = if let Some(idx) = device_index {
        if idx >= adapters.len() {
            return Err(format!(
                "Invalid device index {}. Found {} adapter(s) (indices 0 to {}). Use --list-devices to view them.",
                idx,
                adapters.len(),
                adapters.len() - 1
            ));
        }
        &adapters[idx]
    } else {
        // Auto-select: Discrete GPU first, then Integrated, then first available
        adapters
            .iter()
            .find(|a| a.get_info().device_type == DeviceType::DiscreteGpu)
            .or_else(|| {
                adapters
                    .iter()
                    .find(|a| a.get_info().device_type == DeviceType::IntegratedGpu)
            })
            .unwrap_or(&adapters[0])
    };

    let info = selected_adapter.get_info();

    let (device, queue) = block_on(selected_adapter.request_device(&DeviceDescriptor {
        label: Some("GetPiComputeDevice"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        memory_hints: MemoryHints::Performance,
        ..Default::default()
    }))
    .map_err(|e| format!("Failed to initialize GPU device '{}': {}", info.name, e))?;

    Ok(GpuContext {
        device,
        queue,
        info,
    })
}
