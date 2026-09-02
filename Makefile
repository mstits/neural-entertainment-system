.PHONY: help test parity engine show launcher control-panel bench bench-hot bench-scaling bench-phases bench-all \
        build build-pgo build-pgo-apply selftest clean clean-rust train eval scoreboard \
        test-fast selftest-learning demo gui setup-check setup-game \
        ppu_layout_check ppu-batch-profile rust-check unsafe-inventory-check \
        clear-lint mechanism-check purity-check

help:
	@echo "NES-Evolve Makefile targets:"
	@echo ""
	@echo "  Build:"
	@echo "    make build             - build nes_core (release, no PGO)"
	@echo "    make build-pgo         - build + instrument + rebuild with PGO (~3 min; +81% throughput)"
	@echo "    make build-pgo-apply   - reapply cached PGO profile (~15 s; use after small nes_core edits)"
	@echo ""
	@echo "  Train (Phase 0 onward — see docs/proposals/unified_learning_thesis.md):"
	@echo "    make setup-check       - verify venv + nes_core + torch MPS, and list which"
	@echo "                             per-game ROMs are present with their exact filenames"
	@echo "    make setup-game GAME=x - validate/hash a game's ROM and capture its start-state"
	@echo "    make train GAME=mario  - headless training for the named game (mario, contra,"
	@echo "                             megaman, castlevania, zelda, metroid)"
	@echo "                             Per-game checkpoints land in checkpoints/<game_slug>/"
	@echo "    make eval GAME=mario   - load latest checkpoint, run N eval episodes, report"
	@echo "                             clear rate + furthest stage reached"
	@echo "    make demo GAME=mario   - play the best checkpoint + record demos/<game>.gif"
	@echo "    make gui               - launch the desktop GUI (pick ROM + profile, watch live)"
	@echo "    make show              - Beat the Game (Live): the search system plays SMB"
	@echo "                             power-on through 8-4 in a window, audio on every clear"
	@echo "                             (make show GAME=contra or PROFILE=configs/x.yaml for others)"
	@echo "    make launcher          - the show control panel: browse every game + banked wins,"
	@echo "                             edit every show/profile knob, save/load profiles, then"
	@echo "                             launch the show in Live Solve or Replay mode"
	@echo "    make control-panel     - alias for make launcher"
	@echo "    make scoreboard        - mission-control: progress across all six games"
	@echo ""
	@echo "  Test:"
	@echo "    make test              - pytest suite (incl. slow real-emulator guards)"
	@echo "    make test-fast         - pytest suite minus slow tests (fast inner loop)"
	@echo "    make selftest-learning - real-loop guard: vanilla_ppo learns SMB (~25s)"
	@echo "    make selftest          - GUI widget construction (headless)"
	@echo "    make parity            - nes_core vs nes-py diff harness (under 2 min)"
	@echo "    make mechanism-check RUN=<dir> [REQUIRE=redo,sil]"
	@echo "                           - VOID a run whose armed mechanisms never fired"
	@echo ""
	@echo "  Bench (single-run; thermal-sensitive, run on a cool machine):"
	@echo "    make bench             - preprocess + policy micro-benches"
	@echo "    make bench-hot         - per-layer trainer hot-path breakdown"
	@echo "    make bench-scaling     - worker-count sweep (1,4,8,...,32)"
	@echo "    make bench-phases      - CPU+APU vs PPU-pixel split inside pool_step"
	@echo "    make bench-all         - run every bench; print a summary"
	@echo ""
	@echo "  Maint:"
	@echo "    make clean             - remove cached artifacts"
	@echo "    make clean-rust        - cargo clean nes_core/target (mixed-build hazard)"

# Default game arg for `make train` / `make eval`. Override:
#   make train GAME=zelda
GAME ?= mario

setup-check:
	. .venv/bin/activate && python scripts/train_game.py --setup-check

setup-game:
	. .venv/bin/activate && python scripts/train_game.py --setup-game --game $(GAME)

train:
	. .venv/bin/activate && python scripts/train_game.py --game $(GAME)

eval:
	. .venv/bin/activate && python scripts/eval_game.py --game $(GAME)

demo:
	. .venv/bin/activate && python scripts/demo_game.py --game $(GAME)

gui:
	. .venv/bin/activate && python src/gui/main.py

# Live-solve campaign window (streaming entry). Default = SMB from
# power-on; any profile with a verified `solve:` section works:
#   make show PROFILE=configs/castlevania.yaml
#   make show GAME=contra          # convenience: resolves configs/contra.yaml
PROFILE ?= configs/smb_4_4_micro.yaml
# GAME defaults to `mario` project-wide (train/eval/demo), so only treat it as
# a show override when it's given on the command line -- otherwise `make show`
# stays byte-identical, launching --profile $(PROFILE).
GAME_ARG := $(if $(filter command line,$(origin GAME)),--game $(GAME),--profile $(PROFILE))
show:
	. .venv/bin/activate && caffeinate -dis python scripts/live_solve_show.py $(GAME_ARG)

# The show control panel: a persistent settings GUI over the catalog
# (browse every game, its start-state thumbnail + banked wins, edit every
# show/profile knob, save/load self-contained profiles, then Launch the
# show in Live Solve or Replay mode as a non-blocking subprocess).
launcher control-panel:
	. .venv/bin/activate && python scripts/show_launcher.py

scoreboard:
	. .venv/bin/activate && python scripts/scoreboard.py

rust-check:
	cd nes_core && cargo check --lib

# Mechanical gate on nes_core/SECURITY.md's per-file `unsafe` inventory:
# fails if a file under nes_core/src has `unsafe` but no entry in the
# doc, if a doc entry names a file no longer in the tree, or if a
# per-file count has drifted beyond a small tolerance. This is what
# should have caught the doc claiming "three call sites in the entire
# crate" for ~4 months while the real count grew to 155 across 11
# files -- see nes_core/SECURITY.md and scripts/check_unsafe_inventory.py.
unsafe-inventory-check:
	.venv/bin/python scripts/check_unsafe_inventory.py

# Refuse a solve profile that declares clear machinery it cannot fire, and
# name every profile whose `solutions` count is a compile-time constant
# rather than a search outcome. See scripts/clear_reachability.py for why
# `() > ()` made 40 of 45 profiles structurally unable to bank a win.
clear-lint:
	@.venv/bin/python scripts/clear_reachability.py --all --quiet
	@echo "  (drop --quiet for the per-profile listing, including every"
	@echo "   profile whose solution count is a constant)"

# Refuse a RUN whose armed mechanisms never fired. Returns VOID (not FAIL)
# for any mechanism that announced itself and whose own counter never moved
# for the whole run, and UNAUDITABLE for one that emits no counter at all --
# because a counter at zero is a null result while zero observations of a
# counter is ignorance. ReDo logged `recycled 0 cum 0` on ~2,000 per-iteration
# checks across all 8 v27+v28 runs while being registered as one of two
# variables; see docs/research/LEARNING_TRACK_AUDIT_2026-08-27.md section 6.
#
#   make mechanism-check RUN=runs/v27_fresh_recovery REQUIRE=redo
mechanism-check:
	@.venv/bin/python scripts/check_mechanism_receipt.py $(RUN) \
	    $(if $(REQUIRE),--require $(REQUIRE),)

# Refuse a RAM address this repo formally quarantined that is still live
# in the layer that EXECUTES. A `quarantined_external_knowledge:` block in
# configs/ retracts the documentation claim; it does not touch the Rust
# constant or the Python literal that actually runs. This target closes
# that gap tree-wide — configs, Rust, Python, tests — and fails on any live
# use that is not disclosed in
# docs/purity/engine_quarantine_disclosures.yaml. It lives under tests/
# rather than scripts/ on purpose: `test_no_source_file_reads_the_
# quarantine_key` forbids any PRODUCTION path from reading a quarantine
# block, and a checker that enforces the quarantine must not become the
# one thing that consumes it. Run it bare for the per-site listing,
# ENFORCED and reported alike:
#
#   .venv/bin/python tests/purity_engine_scan.py
# The Rust half of the purity gate. `rust-check` above runs `cargo check`,
# which compiles but runs NOTHING — so the five `win_witness_guard` tests
# (the only ones that DRIVE each reward arm through the byte its ledger row
# names, and prove the disarmed arms cannot report success) were not in any
# default gate. They run in well under two seconds.
purity-check:
	@.venv/bin/python tests/purity_engine_scan.py --check
	@cd nes_core && cargo test --lib win_witness_guard --quiet

mistakes-check:
	@.venv/bin/python scripts/mistakes_tally.py --check > /dev/null || \
	  (.venv/bin/python scripts/mistakes_tally.py --check; exit 1)

test: rust-check unsafe-inventory-check clear-lint purity-check mistakes-check
	. .venv/bin/activate && pytest tests/ -q --timeout=120

test-fast:
	. .venv/bin/activate && pytest tests/ -q -m "not slow" --timeout=120

selftest-learning:
	. .venv/bin/activate && pytest tests/test_learning_regression.py -q -m slow --timeout=180 -s

parity:
	. .venv/bin/activate && pytest tests/parity/ -q -m parity --timeout=60

# Pool/spectator/frame-anchor lib tests live behind `--features python`,
# whose extension-module linking drops libpython from test binaries —
# they are invisible to plain `cargo test --lib`. This target links with
# dynamic_lookup and preloads the venv's libpython so they actually run.
# Single-threaded: the spectator pacing tests assert wall-clock bands.
pool-test:
	CARGO_TARGET_DIR=/tmp/nes_pool_test_target \
	RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
	DYLD_INSERT_LIBRARIES=$$(.venv/bin/python -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.dylib \
	cargo test --manifest-path nes_core/Cargo.toml --lib --features python -- --test-threads=1

# ROM-library scan: probe every ROM in a directory for load/reset/step
# panics and timeouts, then re-probe every `ok` ROM for a frozen (static)
# screen. See scripts/rom_library_scan.py module docstring.
.PHONY: rom-scan rom-scan-static-selftest
rom-scan:
	. .venv/bin/activate && python scripts/rom_library_scan.py \
	  --roms-dir roms --out scan_results --workers 8

rom-scan-static-selftest:
	. .venv/bin/activate && pytest tests/test_rom_library_scan_static.py -v

# `asm_cpu` (AArch64 6502 fast path) is ON by default here, and that is a
# deliberate, receipted decision — not drift. Rationale, the list of which
# ROMs/mappers are lockstep-verified clean vs unverified, and the rule for
# when `disable_asm_cpu` is a real safety net vs superstition:
#   docs/research/ASM_CPU_STATUS_2026-08-25.md
# Do not flip this default without re-reading that doc; an older memory
# claiming "default builds skip asm_cpu" is stale (it never held in this
# repo's history) and the fidelity premise behind it is refuted by the
# nes_core/tests/asm_vs_slow_*.rs family.
build:
	(cd nes_core && ../.venv/bin/maturin develop --release --features "python,asm_cpu")

build-pgo:
	bash scripts/pgo_build.sh full

build-pgo-apply:
	bash scripts/pgo_build.sh apply

# Event-driven-PPU campaign (docs/proposals/ppu_event_driven_catchup.md).
# Rung-0 layout gate: fail if a rung changed the machine code of the hot
# Ppu::tick body. Regenerate the golden only on an intentional,
# gate-passing tick change: bash scripts/ppu_layout_check.sh regen
ppu_layout_check:
	bash scripts/ppu_layout_check.sh check

# Rung-0 batchable-fraction measurement for one ROM+state. Example:
#   make ppu-batch-profile ROM="roms/zelda.nes" STATE="roms/zelda_start_ctrl.state.bin"
ROM ?= roms/Super Mario Bros. (World).nes
STATE ?=
ppu-batch-profile:
	(cd nes_core && cargo build --release --features ppu_batch_stats --example ppu_batch_profile)
	PROF_ROM="$(ROM)" PROF_STATE="$(STATE)" \
	  nes_core/target/release/examples/ppu_batch_profile

bench:
	. .venv/bin/activate && python scripts/bench.py

bench-hot:
	. .venv/bin/activate && python scripts/bench_hot_path.py --workers 16 --steps 200 --frame-skip 16

bench-scaling:
	. .venv/bin/activate && python scripts/bench_worker_scaling.py

bench-phases:
	. .venv/bin/activate && python scripts/bench_emulator_phases.py

bench-all: bench-hot bench-scaling bench-phases
	@echo "=== done; see docs/proposals/pgo_results.md for the perf shape ==="

selftest:
	. .venv/bin/activate && QT_QPA_PLATFORM=offscreen python tests/gui_selftest.py

clean:
	rm -rf .pytest_cache __pycache__ **/__pycache__ **/**/__pycache__
	find . -name "*.pyc" -delete

clean-rust:
	cd nes_core && cargo clean

# Provenance gate for Learned-ledger training inputs (see CLAIMS.md):
# allowlist integrity, quarantine intact, no profile references
# quarantined artifacts. Run before any Learned-ledger training run
# and before publishing any number.
engine:
	@. .venv/bin/activate && python scripts/engine_status.py

engine-check:
	@. .venv/bin/activate && python scripts/engine_status.py --check

provenance-check:
	.venv/bin/python scripts/provenance_check.py

# Cold-boot replay of the banked flagship tape (docs/receipts/full_run/
# full_tape.npy) against every level boundary in receipts.json, writing a
# dated JSON receipt with full build provenance. Takes 2-4 minutes; not a
# `test` prerequisite for that reason. Run after any nes_core change that
# could touch emulation fidelity.
verify-full-run:
	.venv/bin/python scripts/verify_full_run.py
