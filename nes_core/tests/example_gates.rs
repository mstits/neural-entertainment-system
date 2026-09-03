//! Source-level gates on `examples/`.
//!
//! An example that gates itself with a file-scope `#![cfg(...)]` deletes
//! every item in the file when that condition is false, including any
//! fallback `main` written to cover the disabled case. The result is
//! `error[E0601]: main function not found`, which breaks bare
//! `cargo test`, `cargo test --no-run`, `cargo check --examples` and
//! `cargo build --examples` while leaving `cargo test --lib --tests`
//! green, because `--tests` never compiles examples.
//!
//! These two tests catch that shape by reading the sources, so the
//! default-configuration build stays covered by the same
//! `cargo test --lib --tests` run everything else uses.

use std::fs;
use std::path::PathBuf;

const ASM_GATE: &str = "#[cfg(all(target_arch = \"aarch64\", feature = \"asm_cpu\"))]";
const ASM_FALLBACK_GATE: &str =
    "#[cfg(not(all(target_arch = \"aarch64\", feature = \"asm_cpu\")))]";

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

/// Every `.rs` file directly under `examples/`, as (file name, body).
fn example_sources() -> Vec<(String, String)> {
    let dir = examples_dir();
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("example file name")
            .to_string();
        let body = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        out.push((name, body));
    }
    out.sort();
    out
}

/// A file-scope `#![cfg(...)]` in an example removes that example's own
/// `main` in the configuration it excludes, so the default build fails.
/// Gate per item instead, the way `examples/asm_cpu_demo.rs` does.
#[test]
fn no_example_gates_itself_at_file_scope() {
    let sources = example_sources();
    assert!(
        sources.len() >= 20,
        "expected the examples directory to be populated, found {} files: the scan must not \
         pass by finding nothing",
        sources.len()
    );

    let offenders: Vec<String> = sources
        .iter()
        .filter(|(_, body)| {
            body.lines()
                .any(|line| line.trim_start().starts_with("#![cfg"))
        })
        .map(|(name, _)| name.clone())
        .collect();

    assert!(
        offenders.is_empty(),
        "these examples carry a file-scope #![cfg(...)], which deletes their own main in the \
         excluded configuration and breaks bare `cargo test` / `cargo check --examples`: {}. \
         Put the same condition on each item instead.",
        offenders.join(", ")
    );
}

/// `examples/asm_diff_fuzz.rs` builds under two disjoint conditions: the
/// real fuzzer when `asm_cpu` is on and the target is aarch64, and a
/// fallback `main` that prints how to enable it otherwise. Both must be
/// present and item-gated, or one configuration has no `main`.
#[test]
fn asm_diff_fuzz_keeps_a_main_in_both_configurations() {
    let path = examples_dir().join("asm_diff_fuzz.rs");
    let body = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let lines: Vec<&str> = body.lines().collect();

    let mains: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.starts_with("fn main()"))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        mains.len(),
        2,
        "expected exactly two top-level `fn main()` in {} (feature-on fuzzer and feature-off \
         fallback), found {} at lines {:?}",
        path.display(),
        mains.len(),
        mains.iter().map(|i| i + 1).collect::<Vec<_>>()
    );

    let gates: Vec<&str> = mains
        .iter()
        .map(|i| {
            assert!(*i > 0, "a `fn main()` at line 1 cannot carry a cfg attribute");
            lines[i - 1].trim()
        })
        .collect();
    assert!(
        gates.contains(&ASM_GATE),
        "no `fn main()` in {} is gated on {ASM_GATE}; attributes found: {gates:?}",
        path.display()
    );
    assert!(
        gates.contains(&ASM_FALLBACK_GATE),
        "no `fn main()` in {} is gated on {ASM_FALLBACK_GATE}, so the default build has no \
         main; attributes found: {gates:?}",
        path.display()
    );
}
