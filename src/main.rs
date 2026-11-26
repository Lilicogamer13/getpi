mod bbp;
mod decimal;
mod gpu;
#[cfg(test)]
mod tests;

use clap::{Parser, ValueEnum};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static RUNNING: AtomicBool = AtomicBool::new(true);

extern "C" fn sigint_handler(_: libc::c_int) {
    RUNNING.store(false, Ordering::SeqCst);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// Exact decimal digits of Pi (3.14159265...)
    Decimal,
    /// Massively parallel hexadecimal BBP digits (3.243F6A88...)
    Hex,
}

#[derive(Parser, Debug)]
#[command(
    name = "getpi",
    author = "Lio",
    version = "0.1.0",
    about = "Super-optimized GPU-native Pi Calculator for any GPU vendor (NVIDIA, AMD, Intel, Apple Silicon)"
)]
struct Args {
    /// Select which GPU to run on by index (use --list-devices to view available GPUs)
    #[arg(short = 'd', long = "device")]
    device: Option<usize>,

    /// List all available GPUs / compute devices on this machine and exit
    #[arg(short = 'l', long = "list-devices")]
    list_devices: bool,

    /// Calculation mode: decimal (default) or hex (BBP)
    #[arg(short = 'm', long = "mode", value_enum, default_value_t = Mode::Decimal)]
    mode: Mode,

    /// Number of digits to calculate (ignored if --infinite is passed)
    #[arg(short = 'n', long = "digits", default_value_t = 100_000)]
    digits: usize,

    /// Calculate digits continuously/infinitely until stopped with Ctrl+C
    #[arg(short = 'i', long = "infinite")]
    infinite: bool,

    /// Number of digits to compute per GPU dispatch
    #[arg(short = 'b', long = "batch-size")]
    batch_size: Option<usize>,

    /// File path to save calculated digits
    #[arg(short = 'o', long = "output", default_value = "pi.txt")]
    output: PathBuf,
}

pub(crate) fn format_with_commas(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    let rem = s.len() % 3;
    let (head, tail) = s.split_at(if rem == 0 { 3 } else { rem });
    result.push_str(head);
    for chunk in tail.as_bytes().chunks(3) {
        result.push(',');
        result.push_str(std::str::from_utf8(chunk).unwrap());
    }
    result
}

fn main() {
    let args = Args::parse();

    if args.list_devices {
        gpu::list_adapters();
        return;
    }

    // Register clean Ctrl+C handler
    unsafe {
        libc::signal(libc::SIGINT, sigint_handler as *const () as usize);
        libc::signal(libc::SIGTERM, sigint_handler as *const () as usize);
    }

    println!("============================================================");
    println!("     getpi: Super-Optimized GPU-Native Pi Calculator");
    println!("============================================================");

    // Initialize GPU device
    let gpu_ctx = match gpu::init_gpu(args.device) {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("Error: {}", err);
            std::process::exit(1);
        }
    };

    println!(
        "Selected GPU  : {} ({:?}, backend: {:?})",
        gpu_ctx.info.name, gpu_ctx.info.device_type, gpu_ctx.info.backend
    );
    if args.infinite {
        println!("Target digits : ∞ (Infinite continuous mode, press Ctrl+C to stop)");
    } else {
        println!("Target digits : {}", format_with_commas(args.digits));
    }
    println!("Mode          : {:?}", args.mode);
    println!("Output file   : {}", args.output.display());
    println!("Press Ctrl+C at any time to stop and save progress cleanly.\n");

    let file = match File::create(&args.output) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error creating output file '{}': {}", args.output.display(), e);
            std::process::exit(1);
        }
    };
    let mut writer = BufWriter::with_capacity(64 * 1024, file);

    let start_time = Instant::now();
    let mut total_digits: usize = 0;

    match args.mode {
        Mode::Decimal => {
            let batch_size = args.batch_size.unwrap_or(200);
            let mut calc = decimal::DecimalCalculator::new(&gpu_ctx, args.digits, batch_size, args.infinite);

            while RUNNING.load(Ordering::SeqCst) && !calc.is_finished() {
                if let Some(digits_chunk) = calc.compute_next_batch(&gpu_ctx) {
                    writer.write_all(digits_chunk.as_bytes()).unwrap();
                    writer.flush().unwrap();
                    total_digits = calc.total_digits();

                    let elapsed = start_time.elapsed().as_secs_f64();
                    let speed = if elapsed > 0.0 {
                        total_digits as f64 / elapsed
                    } else {
                        0.0
                    };

                    // Overwrite the same line in CLI/TTY using carriage return \r and ANSI clear
                    print!(
                        "\r\x1B[2K[getpi GPU] Digits: {} | Speed: {:>8.0} digits/s | Time: {:>6.2}s | Saved: {}",
                        format_with_commas(total_digits),
                        speed,
                        elapsed,
                        args.output.display()
                    );
                    std::io::stdout().flush().unwrap();
                } else {
                    break;
                }
            }
        }
        Mode::Hex => {
            let batch_size = args.batch_size.unwrap_or(4096);
            let mut calc = bbp::BbpCalculator::new(&gpu_ctx, args.digits, batch_size, args.infinite);

            while RUNNING.load(Ordering::SeqCst) && !calc.is_finished() {
                if let Some(digits_chunk) = calc.compute_next_batch(&gpu_ctx) {
                    writer.write_all(digits_chunk.as_bytes()).unwrap();
                    writer.flush().unwrap();
                    total_digits = calc.total_digits();

                    let elapsed = start_time.elapsed().as_secs_f64();
                    let speed = if elapsed > 0.0 {
                        total_digits as f64 / elapsed
                    } else {
                        0.0
                    };

                    // Overwrite the same line in CLI/TTY using carriage return \r and ANSI clear
                    print!(
                        "\r\x1B[2K[getpi GPU-BBP] Digits: {} | Speed: {:>8.0} digits/s | Time: {:>6.2}s | Saved: {}",
                        format_with_commas(total_digits),
                        speed,
                        elapsed,
                        args.output.display()
                    );
                    std::io::stdout().flush().unwrap();
                } else {
                    break;
                }
            }
        }
    }

    let elapsed = start_time.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let speed = if elapsed_secs > 0.0 {
        total_digits as f64 / elapsed_secs
    } else {
        0.0
    };

    writer.flush().unwrap();

    let was_interrupted = !RUNNING.load(Ordering::SeqCst);

    // Print final newline to conclude in-place CLI output
    println!("\n");
    println!("------------------------------------------------------------");
    if was_interrupted {
        println!("Calculation Stopped by User (Ctrl+C)");
    } else {
        println!("Calculation Complete!");
    }
    println!("Total Digits  : {}", format_with_commas(total_digits));
    println!("Elapsed Time  : {:.2?}", elapsed);
    println!("Average Speed : {:>8.0} digits/s", speed);
    println!("File Saved    : {}", args.output.display());
    println!("------------------------------------------------------------");
}
