//! Versioned savestate envelope.
//!
//! State blobs have historically been raw untagged bincode of
//! `nes::State` — their first 8 bytes are the u64 length prefix of the
//! RAM vector (always 2048 = `00 08 00 00 00 00 00 00`), which is what
//! makes a magic-tagged envelope safe to introduce: no legacy blob can
//! ever begin with the envelope magic (see
//! `tests::legacy_bytes_never_alias_envelope_magic`).
//!
//! Envelope wire format (all little-endian):
//!
//! ```text
//! [u32 magic 0x5341_5645 "SAVE"] [u32 version] [payload]
//! ```
//!
//! * version 1 — payload is `bincode(nes::State)`; the OAM-DMA engine
//!   is not captured and defaults to inactive on load (identical to a
//!   legacy untagged blob).
//! * version 2 — payload is `bincode((nes::State, oam_dma::State))`;
//!   an in-flight sprite-upload DMA survives the round trip, so a
//!   restore at a mid-DMA cycle-locked frame boundary resumes
//!   byte-identically instead of dropping the remaining stall.
//!
//! Writes stay legacy by default: version-2 blobs are produced only
//! when `NES_STATE_V2` is set in the environment (or a caller opts in
//! via `encode_state_with`), so the default write path is byte-
//! identical to the pre-envelope core. The reader accepts every
//! format unconditionally.

use std::sync::OnceLock;

use super::nes;
use super::nes::Nes;
use crate::oam_dma;

/// "SAVE" read as big-endian text; serialized little-endian, so the
/// on-disk bytes are `45 56 41 53`.
pub const ENVELOPE_MAGIC: u32 = 0x5341_5645;
pub const ENVELOPE_VERSION_1: u32 = 1;
pub const ENVELOPE_VERSION_2: u32 = 2;
/// version 3 — payload is `bincode((nes::State, oam_dma::State,
/// ppu::OdoState))`. Written only when the PPU odometer is enabled, so
/// default saves stay byte-identical to the previous writer and every
/// consumer that has not opted in sees no format change at all. The
/// odometer accumulator MUST travel inside the savestate: Go-Explore
/// restores states thousands of times per run, and an external
/// accumulator desyncs on the first restore.
pub const ENVELOPE_VERSION_3: u32 = 3;
/// version 4 — same triple as v3 but `ppu::OdoState` now carries the
/// scene ordinal. bincode is positional, so the field addition needs
/// its own version; v3 blobs decode through `OdoStateV3` and get
/// scene = 0.
pub const ENVELOPE_VERSION_4: u32 = 4;

/// The v3-era odometer payload layout, kept only to decode old blobs.
#[derive(serde_derive::Deserialize)]
struct OdoStateV3 {
    enabled: bool,
    odometer_x: i64,
    odometer_y: i64,
    prev_modal_x: i32,
    prev_modal_y: i32,
    have_prev: bool,
}
pub const ENVELOPE_HEADER_LEN: usize = 8;

/// Opt-in knob for version-2 writes. Read once per process: flipping
/// the variable mid-run cannot produce a mixed-format save stream.
pub fn v2_writes_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("NES_STATE_V2")
            .map(|v| !v.is_empty() && v != *"0")
            .unwrap_or(false)
    })
}

/// Encode with the process-default format (legacy unless
/// `NES_STATE_V2` is set).
pub fn encode_state(nes: &Nes) -> bincode::Result<Vec<u8>> {
    encode_state_with(nes, v2_writes_enabled())
}

/// Encode with an explicit format choice. `v2 == false` is the legacy
/// untagged writer, byte-identical to `bincode::serialize(&get_state())`.
pub fn encode_state_with(nes: &Nes, v2: bool) -> bincode::Result<Vec<u8>> {
    // Odometer on -> version 3 unconditionally: coherence across
    // restores is the entire point of carrying it in the blob.
    if nes.ppu.odometer_enabled {
        let payload = bincode::serialize(&(
            nes.get_state(),
            nes.oam_dma.get_state(),
            nes.ppu.get_odo_state(),
        ))?;
        let mut blob = Vec::with_capacity(ENVELOPE_HEADER_LEN + payload.len());
        blob.extend_from_slice(&ENVELOPE_MAGIC.to_le_bytes());
        blob.extend_from_slice(&ENVELOPE_VERSION_4.to_le_bytes());
        blob.extend_from_slice(&payload);
        return Ok(blob);
    }
    if !v2 {
        return bincode::serialize(&nes.get_state());
    }
    let payload = bincode::serialize(&(nes.get_state(), nes.oam_dma.get_state()))?;
    let mut blob = Vec::with_capacity(ENVELOPE_HEADER_LEN + payload.len());
    blob.extend_from_slice(&ENVELOPE_MAGIC.to_le_bytes());
    blob.extend_from_slice(&ENVELOPE_VERSION_2.to_le_bytes());
    blob.extend_from_slice(&payload);
    Ok(blob)
}

/// True when `body` carries the versioned envelope header.
pub fn has_envelope(body: &[u8]) -> bool {
    body.len() >= ENVELOPE_HEADER_LEN && body[..4] == ENVELOPE_MAGIC.to_le_bytes()
}

/// Decode a state blob of any supported format. Legacy and version-1
/// blobs yield a default (inactive) OAM-DMA snapshot.
pub fn decode_state(
    body: &[u8],
) -> bincode::Result<(nes::State, oam_dma::State, crate::ppu::OdoState)> {
    if !has_envelope(body) {
        let state: nes::State = bincode::deserialize(body)?;
        return Ok((state, oam_dma::State::default(), Default::default()));
    }
    let version = u32::from_le_bytes(body[4..8].try_into().expect("length checked"));
    let payload = &body[ENVELOPE_HEADER_LEN..];
    match version {
        ENVELOPE_VERSION_1 => {
            let state: nes::State = bincode::deserialize(payload)?;
            Ok((state, oam_dma::State::default(), Default::default()))
        }
        ENVELOPE_VERSION_2 => {
            let (s, o) = bincode::deserialize::<(nes::State, oam_dma::State)>(payload)?;
            Ok((s, o, Default::default()))
        }
        ENVELOPE_VERSION_3 => {
            let (s, o, v3) = bincode::deserialize::<(
                nes::State,
                oam_dma::State,
                OdoStateV3,
            )>(payload)?;
            Ok((s, o, crate::ppu::OdoState {
                scene: 0,
                enabled: v3.enabled,
                odometer_x: v3.odometer_x,
                odometer_y: v3.odometer_y,
                prev_modal_x: v3.prev_modal_x,
                prev_modal_y: v3.prev_modal_y,
                have_prev: v3.have_prev,
            }))
        }
        ENVELOPE_VERSION_4 => bincode::deserialize::<(
            nes::State,
            oam_dma::State,
            crate::ppu::OdoState,
        )>(payload),
        v => Err(Box::new(bincode::ErrorKind::Custom(format!(
            "unsupported savestate envelope version {v} \
             (this build reads versions 1..={ENVELOPE_VERSION_4})"
        )))),
    }
}

/// Apply a decoded pair. The OAM-DMA snapshot is applied
/// unconditionally: a legacy/version-1 decode carries the inactive
/// default, so a restore never inherits a stale in-flight transfer
/// from the destination machine.
pub fn apply_decoded(
    nes: &mut Nes,
    state: &nes::State,
    oam_dma: &oam_dma::State,
    odo: &crate::ppu::OdoState,
) {
    nes.apply_state(state);
    nes.oam_dma.apply_state(oam_dma);
    // A v1/v2 blob carries the zeroed default. Restoring it would erase
    // a live odometer on every legacy-rooted restore, so a default-and-
    // disabled snapshot leaves the current odometer configuration
    // alone; a v3 snapshot is applied verbatim (that IS coherence).
    if odo.enabled {
        nes.ppu.apply_odo_state(odo);
    } else {
        // The accumulator survives, but the fold anchor and in-flight
        // line votes still point at the pre-restore timeline; the
        // first fold after the restore would otherwise integrate a
        // phantom delta (or bump the scene) across two unrelated
        // timelines.
        nes.ppu.odo_reanchor();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartridge::Cartridge;
    use crate::sink::{AudioSink, VideoSink};
    use std::path::PathBuf;

    const CPU_CYCLES_PER_FRAME: usize = 29781;

    struct NullVideo;
    impl VideoSink for NullVideo {
        fn write_frame(&mut self, _: &[u8]) {}
        fn frame_written(&self) -> bool {
            false
        }
        fn pixel_size(&self) -> usize {
            4
        }
    }
    struct NullAudio;
    impl AudioSink for NullAudio {
        fn write_sample(&mut self, _: f32) {}
        fn samples_written(&self) -> usize {
            0
        }
    }

    /// Synthetic 32 KB NROM whose PRG repeats INC $0F / STA $4014:
    /// RAM $0F ticks every loop pass and each STA arms an OAM DMA, so
    /// cycle-locked frame boundaries land mid-transfer almost always
    /// (same trick as `pool.rs`'s
    /// `advance_one_frame_stays_cycle_locked_across_oam_dma`).
    /// Reset vector -> $8000.
    fn fixture_rom() -> Vec<u8> {
        let mut rom = Vec::with_capacity(16 + 32 * 1024);
        rom.extend_from_slice(b"NES\x1a");
        rom.push(2); // PRG = 2 × 16 KB = 32 KB
        rom.push(0); // CHR = 0 → CHR-RAM
        rom.push(0); // flags6: mapper 0 low nibble, H-mirror
        rom.push(0); // flags7: mapper 0 high nibble
        rom.extend_from_slice(&[0u8; 8]);
        let mut prg = vec![0u8; 32 * 1024];
        const FILL: &[u8] = &[0xE6, 0x0F, 0x8D, 0x14, 0x40];
        let mut i = 0;
        while i + FILL.len() <= prg.len() {
            prg[i..i + FILL.len()].copy_from_slice(FILL);
            i += FILL.len();
        }
        let n = prg.len();
        prg[n - 4] = 0x00; // low byte of $8000
        prg[n - 3] = 0x80; // high byte of $8000
        rom.extend(prg);
        rom
    }

    fn fixture_nes() -> Nes {
        let cart = Cartridge::load(&mut std::io::Cursor::new(fixture_rom()))
            .expect("synthetic NROM should parse");
        Nes::new(cart)
    }

    fn tick_to_cycle(nes: &mut Nes, target: usize) {
        let mut v = NullVideo;
        let mut a = NullAudio;
        while nes.cycles < target {
            nes.tick(&mut v, &mut a);
        }
    }

    fn advance_frames(nes: &mut Nes, frames: usize) {
        tick_to_cycle(nes, nes.cycles + frames * CPU_CYCLES_PER_FRAME);
    }

    /// The exact machine the golden legacy fixture was captured from:
    /// the fixture ROM ticked to the 3-frame cycle mark. Both the
    /// (ignored) regenerator and the characterization tests go through
    /// here so the recipe can never drift from the golden bytes.
    fn fixture_machine() -> Nes {
        let mut nes = fixture_nes();
        tick_to_cycle(&mut nes, 3 * CPU_CYCLES_PER_FRAME);
        nes
    }

    /// A machine frozen strictly mid-OAM-DMA: transfer armed and part
    /// way through the 256-byte copy.
    fn mid_dma_machine() -> Nes {
        let mut nes = fixture_machine();
        let mut v = NullVideo;
        let mut a = NullAudio;
        let cap = nes.cycles + 4 * CPU_CYCLES_PER_FRAME;
        while nes.cycles < cap {
            nes.tick(&mut v, &mut a);
            if nes.oam_dma.active && nes.oam_dma.count >= 8 && nes.oam_dma.count < 200 {
                break;
            }
        }
        assert!(
            nes.oam_dma.active && nes.oam_dma.count >= 8 && nes.oam_dma.count < 200,
            "fixture ROM should reach a mid-transfer OAM DMA window",
        );
        nes
    }

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("legacy_state_v1.bin")
    }

    fn state_digest(nes: &Nes) -> Vec<u8> {
        bincode::serialize(&nes.get_state()).expect("serialize nes state")
    }

    /// One-shot generator for the golden legacy fixture. Ignored so a
    /// routine test run can never silently rewrite the golden bytes;
    /// run explicitly (`cargo test --lib regenerate_legacy_fixture --
    /// --ignored`) only when the fixture is intentionally re-captured.
    #[test]
    #[ignore]
    fn regenerate_legacy_fixture() {
        let path = fixture_path();
        std::fs::create_dir_all(path.parent().unwrap()).expect("create fixtures dir");
        std::fs::write(&path, state_digest(&fixture_machine())).expect("write fixture");
    }

    /// The legacy writer must stay byte-identical to the golden blob
    /// captured before the envelope landed — the default (knob-off)
    /// write path is characterized, not merely trusted.
    #[test]
    fn golden_legacy_fixture_matches_current_legacy_writer() {
        let golden = std::fs::read(fixture_path()).expect("golden fixture present");
        let now = encode_state_with(&fixture_machine(), false).expect("legacy encode");
        assert_eq!(
            golden, now,
            "legacy write path drifted from the pre-envelope golden bytes",
        );
        assert_eq!(golden[..8], 2048u64.to_le_bytes());
    }

    /// Golden legacy blob through the envelope-aware reader resumes a
    /// 60-frame trajectory byte-identically to the pre-envelope reader.
    #[test]
    fn golden_legacy_fixture_resumes_identically_through_new_reader() {
        let golden = std::fs::read(fixture_path()).expect("golden fixture present");

        let mut old_nes = fixture_nes();
        let state: nes::State =
            bincode::deserialize(&golden).expect("old reader parses golden blob");
        old_nes.apply_state(&state);

        let mut new_nes = fixture_nes();
        let (state, oam_dma, odo) = decode_state(&golden).expect("new reader parses golden blob");
        apply_decoded(&mut new_nes, &state, &oam_dma, &odo);

        assert_eq!(state_digest(&old_nes), state_digest(&new_nes));
        for frame in 0..60 {
            advance_frames(&mut old_nes, 1);
            advance_frames(&mut new_nes, 1);
            assert_eq!(
                state_digest(&old_nes),
                state_digest(&new_nes),
                "trajectory diverged at frame {frame}",
            );
            assert_eq!(old_nes.oam_dma.get_state(), new_nes.oam_dma.get_state());
        }
    }

    /// Non-aliasing proof: every legacy blob starts with the RAM
    /// vector's u64 length prefix (2048), so its first four bytes are
    /// `00 08 00 00` — never the envelope magic in either byte order.
    #[test]
    fn legacy_bytes_never_alias_envelope_magic() {
        let golden = std::fs::read(fixture_path()).expect("golden fixture present");
        let fresh = encode_state_with(&mid_dma_machine(), false).expect("legacy encode");
        for blob in [&golden[..], &fresh[..]] {
            assert_eq!(blob[..8], 2048u64.to_le_bytes());
            assert_ne!(blob[..4], ENVELOPE_MAGIC.to_le_bytes()); // 45 56 41 53
            assert_ne!(blob[..4], ENVELOPE_MAGIC.to_be_bytes()); // 53 41 56 45
            assert!(!has_envelope(blob));
        }
        // The 2048 prefix is structural, not incidental: it IS the
        // serialized RAM length, pinned by the RAM array size.
        let (state, _, _) = decode_state(&golden).expect("decode golden");
        assert_eq!(state.ram.len(), 0x0800);

        let v2 = encode_state_with(&mid_dma_machine(), true).expect("v2 encode");
        assert!(has_envelope(&v2));
        assert_eq!(v2[..4], ENVELOPE_MAGIC.to_le_bytes());
        assert_eq!(
            u32::from_le_bytes(v2[4..8].try_into().unwrap()),
            ENVELOPE_VERSION_2,
        );
    }

    /// V2 round trip from a strictly mid-transfer machine: the DMA
    /// snapshot survives, and the restored machine tracks the source
    /// byte-for-byte for 60 frames.
    #[test]
    fn v2_round_trip_preserves_in_flight_dma() {
        let mut src = mid_dma_machine();
        let blob = encode_state_with(&src, true).expect("v2 encode");

        let (state, oam_dma, odo) = decode_state(&blob).expect("v2 decode");
        assert_eq!(oam_dma, src.oam_dma.get_state());
        assert!(oam_dma.active, "mid-transfer DMA must survive the round trip");

        let mut restored = fixture_nes();
        apply_decoded(&mut restored, &state, &oam_dma, &odo);
        assert_eq!(state_digest(&src), state_digest(&restored));
        assert_eq!(restored.oam_dma.get_state(), src.oam_dma.get_state());

        for frame in 0..60 {
            advance_frames(&mut src, 1);
            advance_frames(&mut restored, 1);
            assert_eq!(
                state_digest(&src),
                state_digest(&restored),
                "trajectory diverged at frame {frame}",
            );
            assert_eq!(src.oam_dma.get_state(), restored.oam_dma.get_state());
        }
    }

    /// Legacy and version-1 blobs decode to an inactive DMA default,
    /// and applying them forces the destination's DMA inactive even
    /// when the destination machine is itself mid-transfer.
    #[test]
    fn legacy_and_v1_envelope_default_dma_inactive() {
        let src = mid_dma_machine();

        let legacy = encode_state_with(&src, false).expect("legacy encode");
        let (state, oam_dma, odo) = decode_state(&legacy).expect("legacy decode");
        assert!(!oam_dma.active);
        assert_eq!(oam_dma, crate::oam_dma::State::default());

        let mut v1 = Vec::new();
        v1.extend_from_slice(&ENVELOPE_MAGIC.to_le_bytes());
        v1.extend_from_slice(&ENVELOPE_VERSION_1.to_le_bytes());
        v1.extend_from_slice(&bincode::serialize(&src.get_state()).expect("serialize"));
        let (v1_state, v1_dma, _v1_odo) = decode_state(&v1).expect("v1 envelope decode");
        assert!(!v1_dma.active);
        assert_eq!(
            bincode::serialize(&v1_state).expect("serialize"),
            bincode::serialize(&state).expect("serialize"),
        );

        let mut target = mid_dma_machine();
        assert!(target.oam_dma.active);
        apply_decoded(&mut target, &state, &oam_dma, &odo);
        assert!(
            !target.oam_dma.active,
            "legacy restore must not inherit the destination's stale DMA",
        );
    }

    /// An enveloped blob with a version this build does not know must
    /// fail loudly instead of feeding the payload to the wrong layout.
    #[test]
    fn unsupported_envelope_version_is_a_clear_error() {
        let src = fixture_machine();
        let mut blob = Vec::new();
        blob.extend_from_slice(&ENVELOPE_MAGIC.to_le_bytes());
        blob.extend_from_slice(&99u32.to_le_bytes());
        blob.extend_from_slice(&bincode::serialize(&src.get_state()).expect("serialize"));
        let err = match decode_state(&blob) {
            Ok(_) => panic!("version 99 must not decode"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("version"),
            "error should name the version problem, got: {err}",
        );
    }

    /// Knob-off default: with `NES_STATE_V2` unset, `encode_state`
    /// produces exactly the golden legacy bytes.
    #[test]
    fn v2_writes_default_off() {
        if std::env::var_os("NES_STATE_V2").is_some() {
            return;
        }
        assert!(!v2_writes_enabled());
        let blob = encode_state(&fixture_machine()).expect("encode");
        assert_eq!(
            blob,
            std::fs::read(fixture_path()).expect("golden fixture present"),
        );
    }

    #[test]
    fn odometer_off_keeps_writer_byte_identical() {
        // The entire compatibility story: with the odometer disabled,
        // both writer modes produce exactly the bytes they did before
        // the odometer existed.
        let nes = fixture_machine();
        assert!(!nes.ppu.odometer_enabled);
        let legacy = encode_state(&nes).expect("legacy encode");
        assert!(!has_envelope(&legacy));
        let v2 = encode_state_with(&nes, true).expect("v2 encode");
        let version = u32::from_le_bytes(v2[4..8].try_into().unwrap());
        assert_eq!(version, ENVELOPE_VERSION_2);
    }

    #[test]
    fn v3_blob_decodes_with_scene_zero() {
        // A v3-era blob (pre-scene OdoState layout) must decode with
        // scene = 0 rather than misparsing positionally.
        let mut nes = fixture_machine();
        nes.ppu.apply_odo_state(&crate::ppu::OdoState {
            scene: 0,
            enabled: true,
            odometer_x: 777,
            odometer_y: -3,
            prev_modal_x: 12,
            prev_modal_y: 34,
            have_prev: true,
        });
        // Hand-build the v3 payload with the old layout.
        #[derive(serde_derive::Serialize)]
        struct OldOdo {
            enabled: bool,
            odometer_x: i64,
            odometer_y: i64,
            prev_modal_x: i32,
            prev_modal_y: i32,
            have_prev: bool,
        }
        let payload = bincode::serialize(&(
            nes.get_state(),
            nes.oam_dma.get_state(),
            OldOdo { enabled: true, odometer_x: 777, odometer_y: -3,
                     prev_modal_x: 12, prev_modal_y: 34, have_prev: true },
        )).expect("v3 payload");
        let mut blob = Vec::new();
        blob.extend_from_slice(&ENVELOPE_MAGIC.to_le_bytes());
        blob.extend_from_slice(&ENVELOPE_VERSION_3.to_le_bytes());
        blob.extend_from_slice(&payload);
        let (_, _, odo) = decode_state(&blob).expect("v3 decode");
        assert!(odo.enabled);
        assert_eq!(odo.odometer_x, 777);
        assert_eq!(odo.scene, 0);
    }

    #[test]
    fn odometer_on_writes_v3_and_round_trips() {
        let mut nes = fixture_machine();
        nes.ppu.apply_odo_state(&crate::ppu::OdoState {
            scene: 0,
            enabled: true,
            odometer_x: 12345,
            odometer_y: -67,
            prev_modal_x: 300,
            prev_modal_y: 100,
            have_prev: true,
        });
        nes.ppu.odometer_scene = 5;
        let blob = encode_state(&nes).expect("v4 encode");
        assert!(has_envelope(&blob));
        let version = u32::from_le_bytes(blob[4..8].try_into().unwrap());
        assert_eq!(version, ENVELOPE_VERSION_4);

        let (state, oam_dma, odo) = decode_state(&blob).expect("v4 decode");
        assert!(odo.enabled);
        assert_eq!(odo.scene, 5);
        assert_eq!(odo.odometer_x, 12345);
        assert_eq!(odo.odometer_y, -67);
        assert_eq!(odo.prev_modal_x, 300);

        let mut target = fixture_nes();
        apply_decoded(&mut target, &state, &oam_dma, &odo);
        assert!(target.ppu.odometer_enabled);
        assert_eq!(target.ppu.odometer_x, 12345);
        assert_eq!(target.ppu.odometer_y, -67);
    }

    #[test]
    fn restore_reverts_odometer_to_saved_value() {
        // Savestate coherence — the reason the odometer lives in the
        // blob at all. Save at x=1000, drift to x=2000, restore: the
        // odometer must read 1000 again, not 2000 and not 3000.
        let mut nes = fixture_machine();
        nes.ppu.apply_odo_state(&crate::ppu::OdoState {
            scene: 0,
            enabled: true,
            odometer_x: 1000,
            odometer_y: 0,
            prev_modal_x: 0,
            prev_modal_y: 0,
            have_prev: true,
        });
        let blob = encode_state(&nes).expect("encode at 1000");
        nes.ppu.odometer_x = 2000;
        let (state, oam_dma, odo) = decode_state(&blob).expect("decode");
        apply_decoded(&mut nes, &state, &oam_dma, &odo);
        assert_eq!(nes.ppu.odometer_x, 1000);
    }

    #[test]
    fn legacy_blob_restore_leaves_live_odometer_alone() {
        // Go-Explore restores legacy-rooted states constantly. A v1/v2
        // blob carries no odometer; restoring one must not zero a live
        // accumulator (the disabled-default guard in apply_decoded).
        let mut nes = fixture_machine();
        let legacy = encode_state(&nes).expect("legacy encode");
        nes.ppu.apply_odo_state(&crate::ppu::OdoState {
            scene: 0,
            enabled: true,
            odometer_x: 777,
            odometer_y: 42,
            prev_modal_x: 5,
            prev_modal_y: 6,
            have_prev: true,
        });
        let (state, oam_dma, odo) = decode_state(&legacy).expect("legacy decode");
        assert!(!odo.enabled);
        apply_decoded(&mut nes, &state, &oam_dma, &odo);
        assert!(nes.ppu.odometer_enabled, "restore must not disable");
        assert_eq!(nes.ppu.odometer_x, 777);
        assert_eq!(nes.ppu.odometer_y, 42);
    }

    // Audit 2026-08 regressions: a blob without an odometer payload
    // (legacy/v1/v2 — every start-state and Go-Explore cell minted
    // before the odometer existed) keeps the accumulator but must
    // re-anchor the fold, or the first post-restore fold splices the
    // pre-restore timeline into the restored one.

    #[test]
    fn legacy_restore_does_not_splice_phantom_delta() {
        let mut nes = fixture_machine();
        let legacy = encode_state(&nes).expect("legacy encode");
        nes.ppu.odometer_enabled = true;
        nes.ppu.odo_test_fold_uniform(300, 0);
        nes.ppu.odo_test_fold_uniform(316, 0);
        assert_eq!(nes.ppu.odometer_x, 16);

        let (state, oam_dma, odo) = decode_state(&legacy).expect("legacy decode");
        assert!(!odo.enabled);
        apply_decoded(&mut nes, &state, &oam_dma, &odo);

        // First fold after the restore comes from the restored
        // timeline's origin. A stale anchor at 316 would read 380 as
        // motion and integrate a phantom +64; a re-anchored fold
        // integrates nothing.
        nes.ppu.odo_test_fold_uniform(380, 0);
        assert_eq!(nes.ppu.odometer_x, 16, "phantom delta spliced across restore");
        assert_eq!(nes.ppu.odometer_scene, 0);
        // The anchor is live again: the next fold integrates normally.
        nes.ppu.odo_test_fold_uniform(388, 0);
        assert_eq!(nes.ppu.odometer_x, 24);
    }

    #[test]
    fn legacy_restore_does_not_bump_scene() {
        let mut nes = fixture_machine();
        let legacy = encode_state(&nes).expect("legacy encode");
        nes.ppu.odometer_enabled = true;
        nes.ppu.odo_test_fold_uniform(300, 0);
        nes.ppu.odo_test_fold_uniform(316, 0);

        let (state, oam_dma, odo) = decode_state(&legacy).expect("legacy decode");
        apply_decoded(&mut nes, &state, &oam_dma, &odo);

        // Restored origin far from the stale anchor (|100 - 316| > 64):
        // a stale anchor would read it as a rendered cut.
        nes.ppu.odo_test_fold_uniform(100, 0);
        assert_eq!(nes.ppu.odometer_scene, 0, "phantom scene bump across restore");
        assert_eq!(nes.ppu.odometer_x, 16);
    }

    #[test]
    fn legacy_restore_clears_in_flight_line_votes() {
        let mut nes = fixture_machine();
        let legacy = encode_state(&nes).expect("legacy encode");
        nes.ppu.odometer_enabled = true;
        // Worker abandoned mid-frame by a cycle-locked save.
        nes.ppu.odo_test_seed_lines(130, 999, 0);

        let (state, oam_dma, odo) = decode_state(&legacy).expect("legacy decode");
        apply_decoded(&mut nes, &state, &oam_dma, &odo);
        assert_eq!(
            nes.ppu.odo_test_line_n(),
            0,
            "dead-timeline line votes survived the restore",
        );
    }

    #[test]
    fn cycle_locked_saves_land_mid_visible_frame() {
        // Premise behind the re-anchor guards above: the cycle-locked
        // advance (29781 CPU cycles) drifts against the true PPU frame
        // (~29780.5), so save boundaries sweep the visible region
        // rather than resting at vblank. Measured 284/600 mid-visible
        // on this fixture when the guard was written.
        let mut nes = fixture_machine();
        let mut mid_visible = 0usize;
        for _ in 0..600 {
            advance_frames(&mut nes, 1);
            if (1..240).contains(&nes.ppu.scanline) {
                mid_visible += 1;
            }
        }
        assert!(
            mid_visible > 100,
            "cycle-locked boundaries no longer land mid-frame ({mid_visible}/600); \
             if the advance became frame-aligned, the restore scratch guards \
             and this premise test can both be retired together",
        );
    }
}
