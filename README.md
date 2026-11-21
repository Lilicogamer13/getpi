# getpi: Super-Optimized GPU-Native Pi Calculator

A high-performance Pi calculator written in Rust that runs **entirely on the GPU** via **WebGPU (wgpu)** compute shaders.

- **Vendor-Agnostic**: Runs on **any GPU** supporting Vulkan, Metal, DirectX 12, or WebGPU (NVIDIA, AMD Radeon, Intel Arc / Iris / HD Graphics, Apple Silicon). Not hardcoded to CUDA.
- **GPU Selection**: List all available compute devices (`--list-devices`) and select any GPU (`--device <INDEX>`).
- **Infinite Mode**: Pass `--infinite` (or `-i`) to compute digits continuously until stopped with `Ctrl+C`.
- **In-Place Live TTY Output**: Displays real-time progress (digits calculated, digits/sec, elapsed time) on a single updating line without repeating or scrolling lines.
- **Real-Time Streaming to Disk**: Streams calculated digits to `pi.txt` (or custom `-o` path).
- **Graceful Shutdown**: Intercepts `Ctrl+C` cleanly to flush buffers, preserve progress, and exit.
- **Exact Digit Count**: Output never exceeds the requested digit limit, even when the GPU dispatches more than needed.
- **Dual Computation Modes**:
  - `decimal` (default): Exact decimal digits (`3.1415926535...`) via an integer spigot pipeline running directly in GPU VRAM with zero floating-point error.
  - `hex`: Massively parallel Bailey-Borwein-Plouffe (BBP) formula (`3.243F6A88...`) with Neumaier compensated summation, saturating thousands of GPU cores simultaneously.

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

---

## Running the Tests

The test suite is split into two tiers so it works everywhere, with or without a GPU.

### CPU-only tests (always run, no GPU required)

These cover all pure-logic code: number formatting, digit trimming helpers, block
formatting, hex-character tables, and known Pi digit reference checks.

```bash
cargo test
```

Expected output:

```
running 32 tests
test tests::bbp_helpers::bbp_trim_at_limit ... ok
test tests::bbp_helpers::hex_chars_values ... ok
...
test result: ok. 32 passed; 0 failed; 0 ignored
```

### GPU integration tests (opt-in)

These tests actually dispatch compute shaders and verify correctness: exact digit
counts, known prefix checks for both decimal and hex modes, `is_finished()` state,
and GPU adapter selection.

Enable them with the `gpu_tests` feature flag:

```bash
cargo test --features gpu_tests
```

#### Selecting which GPU to test on

By default the tests use the same auto-selection logic as the main binary
(Discrete GPU first, then Integrated, then first available).

To target a specific GPU, set the `GETPI_TEST_GPU` environment variable to the
index shown by `--list-devices`:

```bash
# See available GPUs and their indices
./target/release/getpi --list-devices
#   [0] NVIDIA RTX 4090   | Discrete GPU     | Vulkan
#   [1] Intel Iris Xe     | Integrated GPU   | Vulkan
#   [2] llvmpipe          | CPU Rasterizer   | Vulkan

# Run GPU tests on adapter index 1
GETPI_TEST_GPU=1 cargo test --features gpu_tests

# See which adapter was selected during each test
GETPI_TEST_GPU=0 cargo test --features gpu_tests -- --nocapture
# [gpu_test] Using adapter [0]: NVIDIA RTX 4090 (DiscreteGpu, Vulkan)
```

| `GETPI_TEST_GPU` value | Behaviour |
|------------------------|-----------|
| Unset or empty         | Auto-selects best available GPU |
| `0`, `1`, `2`, ...     | Uses that exact adapter index |
| Non-numeric string     | Test panics immediately with a descriptive error |
| Out-of-range index     | Test fails with a descriptive error |
| No GPU present at all  | All GPU tests are skipped gracefully |

No test hard-codes any vendor name, backend, or adapter index - GPU selection is
always driven by the runtime environment or the `GETPI_TEST_GPU` variable.
