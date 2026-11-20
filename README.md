# getpi: Super-Optimized GPU-Native Pi Calculator

A high-performance Pi calculator written in Rust that runs **entirely on the GPU** via **WebGPU (wgpu)** compute shaders.

- **Vendor-Agnostic**: Runs on **any GPU** supporting Vulkan, Metal, DirectX 12, or WebGPU (NVIDIA, AMD Radeon, Intel Arc / Iris / HD Graphics, Apple Silicon). Not hardcoded to CUDA.
- **GPU Selection**: List all available compute devices (`--list-devices`) and select any GPU (`--device <INDEX>`).
- **Infinite Mode**: Pass `--infinite` (or `-i`) to compute digits continuously until stopped with `Ctrl+C`.
- **In-Place Live TTY Output**: Displays real-time progress (digits calculated, digits/sec, elapsed time) on a single updating line without repeating or scrolling lines.
- **Real-Time Streaming to Disk**: Streams calculated digits to `pi.txt` (or custom `-o` path).
- **Graceful Shutdown**: Intercepts `Ctrl+C` cleanly to flush buffers, preserve progress, and exit.
- **Dual Computation Modes**:
  - `decimal` (default): Exact decimal digits (`3.1415926535...`) via an integer spigot pipeline running directly in GPU VRAM with zero floating-point error.
  - `hex`: Massively parallel Bailey–Borwein–Plouffe (BBP) formula (`3.243F6A88...`) with Neumaier compensated summation, saturating thousands of GPU cores simultaneously.

---

## Quick Start

### 1. Build Optimized Release Binary
```bash
cargo build --release
```

### 2. List Detected GPUs / Accelerators
```bash
./target/release/getpi --list-devices
```

### 3. Run Infinite Calculation
Continuous calculation (until you press `Ctrl+C`):
```bash
./target/release/getpi --infinite
```

Run in Massively Parallel Hexadecimal BBP Mode infinitely:
```bash
./target/release/getpi --mode hex --infinite
```

### 4. Run Fixed-Digit Calculation
```bash
./target/release/getpi --digits 50000
```

Select a specific GPU:
```bash
./target/release/getpi --device 0 --digits 20000
```

---

## CLI Options

```text
Usage: getpi [OPTIONS]

Options:
  -d, --device <DEVICE>          Select which GPU to run on by index
  -l, --list-devices             List all available GPUs / devices and exit
  -m, --mode <MODE>              Mode: 'decimal' (default) or 'hex' [default: decimal]
  -n, --digits <DIGITS>          Number of digits to calculate [default: 100000]
  -i, --infinite                 Calculate continuously/infinitely until Ctrl+C
  -b, --batch-size <BATCH_SIZE>  Digits calculated per GPU dispatch
  -o, --output <OUTPUT>          Path to output file [default: pi.txt]
  -h, --help                     Print help
  -V, --version                  Print version
```
