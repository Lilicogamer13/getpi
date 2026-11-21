/// Unit tests for getpi.
///
/// All tests that do NOT touch the GPU are marked with no special attribute
/// and run on every platform (Linux, Windows, macOS) with `cargo test`.
///
/// Tests that DO require a GPU adapter are gated on the `gpu_tests` feature
/// flag so CI without a GPU can skip them:
///   cargo test                        # CPU-only, runs everywhere
///   cargo test --features gpu_tests   # also runs GPU integration tests
///
/// No test ever hard-codes a vendor, adapter name, or backend – the GPU is
/// always selected by the same auto-selection logic as the main binary.

// ──────────────────────────────────────────────────────────────────────────────
// Pure-logic (no GPU) tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod format_with_commas_tests {
    use crate::format_with_commas;

    #[test]
    fn zero() {
        assert_eq!(format_with_commas(0), "0");
    }

    #[test]
    fn single_digit() {
        assert_eq!(format_with_commas(7), "7");
    }

    #[test]
    fn three_digits() {
        assert_eq!(format_with_commas(999), "999");
    }

    #[test]
    fn four_digits() {
        assert_eq!(format_with_commas(1_000), "1,000");
    }

    #[test]
    fn six_digits() {
        assert_eq!(format_with_commas(123_456), "123,456");
    }

    #[test]
    fn seven_digits() {
        assert_eq!(format_with_commas(1_234_567), "1,234,567");
    }

    #[test]
    fn large() {
        assert_eq!(format_with_commas(1_000_000_000), "1,000,000,000");
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// decimal module helpers
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod decimal_helpers {
    use crate::decimal::{format_decimal_block, trim_to_limit};

    // ── format_decimal_block ─────────────────────────────────────────────────

    /// Block 0, value 3141 → "3.141"
    #[test]
    fn first_block_formatting() {
        assert_eq!(format_decimal_block(3141, true), "3.141");
    }

    /// Block 0 pads the decimal part to 3 digits.
    #[test]
    fn first_block_zero_pad() {
        assert_eq!(format_decimal_block(3001, true), "3.001");
    }

    /// Non-first blocks are zero-padded to 4 digits.
    #[test]
    fn later_block_zero_pad() {
        assert_eq!(format_decimal_block(5, false), "0005");
        assert_eq!(format_decimal_block(42, false), "0042");
        assert_eq!(format_decimal_block(1234, false), "1234");
    }

    /// Maximum value for a base-10000 block.
    #[test]
    fn later_block_max_value() {
        assert_eq!(format_decimal_block(9999, false), "9999");
    }

    // ── trim_to_limit ────────────────────────────────────────────────────────

    #[test]
    fn trim_no_overflow() {
        // batch is 4 chars, limit is 10, already have 4 → still 4 more fits.
        assert_eq!(trim_to_limit("5926", 4, 10), "5926");
    }

    #[test]
    fn trim_exact_fit() {
        assert_eq!(trim_to_limit("5926", 6, 10), "5926");
    }

    #[test]
    fn trim_partial() {
        // already 8 of 10 written; only 2 more allowed.
        assert_eq!(trim_to_limit("5358", 8, 10), "53");
    }

    #[test]
    fn trim_already_at_limit() {
        assert_eq!(trim_to_limit("5358", 10, 10), "");
    }

    #[test]
    fn trim_zero_limit() {
        assert_eq!(trim_to_limit("anything", 0, 0), "");
    }

    #[test]
    fn trim_empty_raw() {
        assert_eq!(trim_to_limit("", 0, 100), "");
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// bbp module helpers
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod bbp_helpers {
    use crate::bbp::{bbp_prefix, hex_digit_char, trim_bbp_to_limit, HEX_CHARS};

    #[test]
    fn hex_chars_length() {
        assert_eq!(HEX_CHARS.len(), 16);
    }

    #[test]
    fn hex_chars_values() {
        for (i, &b) in HEX_CHARS.iter().enumerate() {
            let expected = format!("{:X}", i).as_bytes()[0];
            assert_eq!(b, expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn hex_digit_char_all() {
        let expected = "0123456789ABCDEF";
        for (i, ch) in expected.chars().enumerate() {
            assert_eq!(hex_digit_char(i as u32), ch, "index {}", i);
        }
    }

    #[test]
    fn hex_digit_char_masks_high_bits() {
        // Only lower 4 bits are used.
        assert_eq!(hex_digit_char(0x1A), hex_digit_char(0xA)); // 0x1A & 0xF = 0xA
    }

    #[test]
    fn prefix_first_batch() {
        assert_eq!(bbp_prefix(0), "3.");
    }

    #[test]
    fn prefix_subsequent_batches() {
        assert_eq!(bbp_prefix(1), "");
        assert_eq!(bbp_prefix(4096), "");
    }

    // ── trim_bbp_to_limit ────────────────────────────────────────────────────

    #[test]
    fn bbp_trim_no_overflow() {
        assert_eq!(trim_bbp_to_limit("243F", 0, 10), "243F");
    }

    #[test]
    fn bbp_trim_partial() {
        // limit=6, already wrote 4 → 2 more allowed.
        assert_eq!(trim_bbp_to_limit("6A88", 4, 6), "6A");
    }

    #[test]
    fn bbp_trim_at_limit() {
        assert_eq!(trim_bbp_to_limit("6A88", 6, 6), "");
    }

    #[test]
    fn bbp_trim_zero_limit() {
        assert_eq!(trim_bbp_to_limit("6A88", 0, 0), "");
    }

    #[test]
    fn bbp_trim_empty_raw() {
        assert_eq!(trim_bbp_to_limit("", 0, 100), "");
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Known Pi digit sanity checks (CPU / reference values, no GPU)
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod pi_reference {
    /// The first N decimal digits of Pi after the decimal point.
    const PI_DECIMAL_REF: &str =
        "14159265358979323846264338327950288419716939937510\
         58209749445923078164062862089986280348253421170679";

    /// The first N hex digits of Pi after the decimal point (BBP formula).
    const PI_HEX_REF: &str = "243F6A8885A308D313198A2E03707344A4093822299F31D008";

    #[test]
    fn decimal_reference_starts_with_14159() {
        assert!(PI_DECIMAL_REF.starts_with("14159265358979"));
    }

    #[test]
    fn hex_reference_starts_with_243f() {
        assert!(PI_HEX_REF.starts_with("243F6A88"));
    }

    #[test]
    fn decimal_reference_length_at_least_50() {
        assert!(PI_DECIMAL_REF.len() >= 50);
    }

    #[test]
    fn hex_reference_length_at_least_48() {
        assert!(PI_HEX_REF.len() >= 48);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GPU integration tests  (opt-in with --features gpu_tests)
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "gpu_tests"))]
mod gpu_integration {
    use crate::gpu;
    use crate::decimal::DecimalCalculator;
    use crate::bbp::BbpCalculator;

    /// Read the GPU index to use from the environment variable `GETPI_TEST_GPU`.
    ///
    /// Usage:
    ///   GETPI_TEST_GPU=1 cargo test --features gpu_tests
    ///
    /// If the variable is unset or empty, `None` is returned and the normal
    /// auto-selection logic (Discrete > Integrated > first) is used.
    /// If the variable is set to an invalid number the test panics immediately
    /// with a clear message so you don't silently run on the wrong device.
    fn test_gpu_index() -> Option<usize> {
        match std::env::var("GETPI_TEST_GPU") {
            Ok(val) if !val.trim().is_empty() => {
                Some(val.trim().parse::<usize>().unwrap_or_else(|_| {
                    panic!(
                        "GETPI_TEST_GPU={:?} is not a valid non-negative integer. \
                         Run `cargo run -- --list-devices` to see available indices.",
                        val
                    )
                }))
            }
            _ => None,
        }
    }

    /// Initialise the GPU selected by `GETPI_TEST_GPU` (or auto-select).
    /// Prints which adapter was chosen so it shows up in `cargo test -- --nocapture`.
    /// Returns `None` (skipping the test) if no adapter is available.
    fn try_gpu() -> Option<gpu::GpuContext> {
        let idx = test_gpu_index();
        match gpu::init_gpu(idx) {
            Ok(ctx) => {
                println!(
                    "[gpu_test] Using adapter [{}]: {} ({:?}, {:?})",
                    idx.map(|i| i.to_string()).unwrap_or_else(|| "auto".into()),
                    ctx.info.name,
                    ctx.info.device_type,
                    ctx.info.backend,
                );
                Some(ctx)
            }
            Err(e) => {
                println!("[gpu_test] No GPU available, skipping: {}", e);
                None
            }
        }
    }

    // ── Decimal mode ──────────────────────────────────────────────────────────

    /// First 10 decimal digits of Pi: 3.141592653
    const PI_DEC_10: &str = "3.141592653";
    /// First 20 decimal digits: 3.14159265358979323846
    const PI_DEC_20: &str = "3.14159265358979323846";

    #[test]
    fn decimal_first_10_digits_correct() {
        let Some(gpu) = try_gpu() else { return };
        let mut calc = DecimalCalculator::new(&gpu, 10, 50, false);
        let mut out = String::new();
        while !calc.is_finished() {
            if let Some(chunk) = calc.compute_next_batch(&gpu) {
                out.push_str(&chunk);
            }
        }
        // The output should start with the known prefix.
        assert!(
            out.starts_with(PI_DEC_10),
            "got: {:?}, expected prefix: {:?}",
            out,
            PI_DEC_10
        );
    }

    #[test]
    fn decimal_digit_count_matches_request() {
        let Some(gpu) = try_gpu() else { return };
        let target = 100;
        let mut calc = DecimalCalculator::new(&gpu, target, 50, false);
        let mut out = String::new();
        while !calc.is_finished() {
            if let Some(chunk) = calc.compute_next_batch(&gpu) {
                out.push_str(&chunk);
            }
        }
        // "3." does not count toward the digit limit; only the digits after ".".
        // The output format is "3.DDDDDDD…" so len = 2 + target.
        let decimal_digits = out.len() - 2; // subtract "3."
        assert_eq!(
            decimal_digits, target,
            "output had {} decimal digits, expected {}; output = {:?}",
            decimal_digits, target, &out[..out.len().min(30)]
        );
    }

    #[test]
    fn decimal_never_exceeds_limit_with_large_batch() {
        // batch_size > target → the calculator must still stop at target.
        let Some(gpu) = try_gpu() else { return };
        let target = 7;
        let mut calc = DecimalCalculator::new(&gpu, target, 200, false);
        let mut out = String::new();
        while !calc.is_finished() {
            if let Some(chunk) = calc.compute_next_batch(&gpu) {
                out.push_str(&chunk);
            }
        }
        let decimal_digits = out.len() - 2;
        assert_eq!(decimal_digits, target,
            "over-wrote: got {} digits, wanted {}; out={:?}", decimal_digits, target, out);
    }

    #[test]
    fn decimal_total_digits_counter_never_exceeds_target() {
        let Some(gpu) = try_gpu() else { return };
        let target = 13;
        let mut calc = DecimalCalculator::new(&gpu, target, 50, false);
        while !calc.is_finished() {
            calc.compute_next_batch(&gpu);
            assert!(
                calc.total_digits() <= target,
                "counter {} > target {}",
                calc.total_digits(),
                target
            );
        }
    }

    #[test]
    fn decimal_is_finished_after_target() {
        let Some(gpu) = try_gpu() else { return };
        let mut calc = DecimalCalculator::new(&gpu, 50, 50, false);
        while !calc.is_finished() {
            calc.compute_next_batch(&gpu);
        }
        assert!(calc.is_finished());
    }

    #[test]
    fn decimal_known_prefix_20_digits() {
        let Some(gpu) = try_gpu() else { return };
        let mut calc = DecimalCalculator::new(&gpu, 20, 50, false);
        let mut out = String::new();
        while !calc.is_finished() {
            if let Some(chunk) = calc.compute_next_batch(&gpu) {
                out.push_str(&chunk);
            }
        }
        assert!(
            out.starts_with(PI_DEC_20),
            "expected {:?}, got {:?}", PI_DEC_20, out
        );
    }

    // ── BBP hex mode ─────────────────────────────────────────────────────────

    /// First 8 hex digits of Pi after the decimal point.
    const PI_HEX_8: &str = "3.243F6A88";

    #[test]
    fn bbp_first_8_hex_digits_correct() {
        let Some(gpu) = try_gpu() else { return };
        let mut calc = BbpCalculator::new(&gpu, 8, 256, false);
        let mut out = String::new();
        while !calc.is_finished() {
            if let Some(chunk) = calc.compute_next_batch(&gpu) {
                out.push_str(&chunk);
            }
        }
        assert!(
            out.starts_with(PI_HEX_8),
            "expected {:?}, got {:?}", PI_HEX_8, out
        );
    }

    #[test]
    fn bbp_digit_count_matches_request() {
        let Some(gpu) = try_gpu() else { return };
        let target = 100;
        let mut calc = BbpCalculator::new(&gpu, target, 256, false);
        let mut out = String::new();
        while !calc.is_finished() {
            if let Some(chunk) = calc.compute_next_batch(&gpu) {
                out.push_str(&chunk);
            }
        }
        // Output is "3.HHHH…" so digit count = len - 2.
        let hex_digits = out.len() - 2;
        assert_eq!(hex_digits, target,
            "got {} hex digits, wanted {}; out starts: {:?}", hex_digits, target, &out[..20.min(out.len())]);
    }

    #[test]
    fn bbp_never_exceeds_limit_with_large_batch() {
        let Some(gpu) = try_gpu() else { return };
        let target = 5;
        // batch_size (256) >> target; must still stop at 5.
        let mut calc = BbpCalculator::new(&gpu, target, 256, false);
        let mut out = String::new();
        while !calc.is_finished() {
            if let Some(chunk) = calc.compute_next_batch(&gpu) {
                out.push_str(&chunk);
            }
        }
        let hex_digits = out.len() - 2;
        assert_eq!(hex_digits, target,
            "over-wrote: got {} hex digits, wanted {}", hex_digits, target);
    }

    #[test]
    fn bbp_is_finished_after_target() {
        let Some(gpu) = try_gpu() else { return };
        let mut calc = BbpCalculator::new(&gpu, 50, 256, false);
        while !calc.is_finished() {
            calc.compute_next_batch(&gpu);
        }
        assert!(calc.is_finished());
    }

    #[test]
    fn bbp_total_digits_counter_never_exceeds_target() {
        let Some(gpu) = try_gpu() else { return };
        let target = 13;
        let mut calc = BbpCalculator::new(&gpu, target, 256, false);
        while !calc.is_finished() {
            calc.compute_next_batch(&gpu);
            assert!(
                calc.total_digits() <= target,
                "counter {} > target {}", calc.total_digits(), target
            );
        }
    }

    // ── GPU selection ─────────────────────────────────────────────────────────

    #[test]
    fn list_adapters_does_not_panic() {
        // Should always succeed; at worst prints "(No GPU…)" and returns.
        gpu::list_adapters();
    }

    #[test]
    fn init_gpu_auto_select_succeeds() {
        // If there is any GPU the auto-selector must return Ok.
        let (_, adapters) = gpu::get_adapters();
        if adapters.is_empty() {
            return; // no GPU in this environment – skip
        }
        assert!(gpu::init_gpu(None).is_ok());
    }

    #[test]
    fn init_gpu_selected_index_succeeds() {
        // Uses the same GETPI_TEST_GPU index that all other GPU tests use.
        let (_, adapters) = gpu::get_adapters();
        if adapters.is_empty() {
            return;
        }
        let idx = test_gpu_index();
        assert!(
            gpu::init_gpu(idx).is_ok(),
            "init_gpu({:?}) failed; run --list-devices to check available indices",
            idx
        );
    }

    #[test]
    fn init_gpu_explicit_index_zero() {
        let (_, adapters) = gpu::get_adapters();
        if adapters.is_empty() {
            return;
        }
        assert!(gpu::init_gpu(Some(0)).is_ok());
    }

    #[test]
    fn init_gpu_out_of_range_returns_error() {
        let (_, adapters) = gpu::get_adapters();
        let bad_idx = adapters.len() + 999;
        assert!(
            gpu::init_gpu(Some(bad_idx)).is_err(),
            "expected Err for index {}", bad_idx
        );
    }
}
