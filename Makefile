.PHONY: help test smoke parity bench bench-hot bench-scaling bench-phases bench-all \
        build build-pgo build-pgo-apply selftest clean train eval scoreboard \
        test-fast selftest-learning demo

help:
	@echo "NES-Evolve Makefile targets:"
	@echo ""
	@echo "  Build:"
	@echo "    make build             - build nes_core (release, no PGO)"
	@echo "    make build-pgo         - build + instrument + rebuild with PGO (~3 min; +81% throughput)"
	@echo "    make build-pgo-apply   - reapply cached PGO profile (~15 s; use after small nes_core edits)"
	@echo ""
	@echo "  Train (Phase 0 onward — see docs/proposals/unified_learning_thesis.md):"
	@echo "    make train GAME=mario  - headless training for the named game (mario, contra,"
	@echo "                             megaman, castlevania, zelda, metroid)"
	@echo "                             Per-game checkpoints land in checkpoints/<game_slug>/"
	@echo "    make eval GAME=mario   - load latest checkpoint, run N eval episodes, report"
	@echo "                             clear rate + furthest stage reached"
	@echo "    make demo GAME=mario   - play the best checkpoint + record demos/<game>.gif"
	@echo "    make scoreboard        - mission-control: progress across all six games"
	@echo ""
	@echo "  Test:"
	@echo "    make test              - pytest suite (incl. slow real-emulator guards)"
	@echo "    make test-fast         - pytest suite minus slow tests (fast inner loop)"
	@echo "    make selftest-learning - real-loop guard: vanilla_ppo learns SMB (~25s)"
	@echo "    make smoke             - 60-second full-stack CLI checks"
	@echo "    make selftest          - GUI widget construction (headless)"
	@echo "    make parity            - nes_core vs nes-py diff harness (under 2 min)"
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

# Default game arg for `make train` / `make eval`. Override:
#   make train GAME=zelda
GAME ?= mario

train:
	. .venv/bin/activate && python scripts/train_game.py --game $(GAME)

eval:
	. .venv/bin/activate && python scripts/eval_game.py --game $(GAME)

demo:
	. .venv/bin/activate && python scripts/demo_game.py --game $(GAME)

scoreboard:
	. .venv/bin/activate && python scripts/scoreboard.py

test:
	. .venv/bin/activate && pytest tests/ -q --timeout=120

test-fast:
	. .venv/bin/activate && pytest tests/ -q -m "not slow" --timeout=120

selftest-learning:
	. .venv/bin/activate && pytest tests/test_learning_regression.py -q -m slow --timeout=180 -s

smoke:
	. .venv/bin/activate && bash scripts/smoke.sh

parity:
	. .venv/bin/activate && pytest tests/parity/ -q -m parity --timeout=60

build:
	(cd nes_core && ../.venv/bin/maturin develop --release --features "python,asm_cpu")

build-pgo:
	bash scripts/pgo_build.sh full

build-pgo-apply:
	bash scripts/pgo_build.sh apply

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
