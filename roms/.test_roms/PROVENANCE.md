# `roms/.test_roms/`: provenance and redistribution

These two files are the only third-party ROM artifacts tracked in this
repository. Everything else under `roms/` is user-supplied and gitignored.

| File | Bytes | SHA-256 |
|---|---|---|
| `nestest.nes` | 24,592 | `f67d55fd6b3cf0bad1cc85f1df0d739c65b53e79cecb7fea8f77ec0eadab0004` |
| `nestest.log` | 859,167 | `442c4dd5539c7e88b3fd73c7b732a7eadbd22b47c2cd9e58397ef147f64f6f8f` |

`nestest.nes` MD5 is `4068f00f3db2fe783e437681fa6b419a`, the value the NESdev
community quotes for the canonical build. `nestest.log` is 8,991 lines, one per
instruction boundary, and `nes_core/tests/nestest_validation.rs` asserts that
count so a truncated copy cannot quietly shrink the gate.

## What they are

`nestest.nes` is kevtris's (Kevin Horton's) NES CPU diagnostic ROM, a homebrew
program written to exercise every official and undocumented 6502 opcode,
addressing mode and flag-update path. It contains no commercial game code.
`nestest.log` is a per-instruction execution trace of that ROM captured from
Nintendulator, a third-party emulator, and used here as the golden reference.

## Redistribution terms, stated plainly

**Neither file carries an explicit licence.** No licence text, copyright
notice or dedication ships with either artifact upstream, and this repository
has no grant on file from either author. Do not read the word "public domain"
into that absence; earlier revisions of the README used it, and it was an
assumption, not a citation.

What is true and checkable: both have been published for free download and
redistributed in the test suites of open-source NES emulators for roughly two
decades, they were authored and released for exactly this purpose, and neither
is a commercial work. That is the basis on which they are vendored here, and it
is the basis on which a downstream consumer inherits them.

If either author objects, delete both files and the nestest gate will fail
loudly rather than skip; see the `Err` arm of
`nestest_full_log_matches_canonical_byte_exact`. Replace them with a fetch
script at that point, and correct the README's testing table in the same
change, because it quotes the 8,991-instruction number.

## Why they are tracked at all

The gate they feed is the CPU spec gate, the claim the README leads with. If
the artifacts did not ship, every clone would run that gate against nothing.
That was the state of the repository until 2026-09-02: `.gitignore` excluded
`roms/`, git therefore never descended into it, the `!roms/.test_roms/`
negations below it could never fire, and the test reported a green in 0.00s
having validated zero instructions. The ignore rule is now `roms/*`, which
leaves the directory walkable so the negation can re-include this one child.
