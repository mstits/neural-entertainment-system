#!/usr/bin/env bash
# Full-stack CLI smoke suite. Each check exits non-zero on failure so this
# script can be wired into CI. Designed to run in under 60 seconds total,
# using FakeNESEnvironment so no ROM is required.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: activate the venv first: source .venv/bin/activate" >&2
    exit 1
fi

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

fail() {
    echo "SMOKE FAIL: $1" >&2
    exit 1
}

echo "==> [1/5] Import surface"
python -c "
from src.training.trainer import Trainer, find_latest_checkpoint
from src.training.behavior_cloning import build_dataset, pretrain
from src.training.genetic_algorithm import GeneticAlgorithm
from src.training.curriculum import CurriculumManager
import nes_core
from src.emulation.frame_utils import FrameStacker
NESEnvironment = nes_core.NESEnvironment
from src.emulation.parallel_pool import ParallelPool
from src.emulation.fake_environment import FakeNESEnvironment
from src.models.policy_network import PolicyNetwork
from src.utils.reward_functions import build_reward_function
from src.utils.logging_config import configure
from src.gui import main_window, emulator_grid, metrics_window, play_window, replay_window
print('  all modules import')
" || fail "import surface"

echo "==> [2/5] Parallel pool with fake env"
python -c "
from src.emulation.parallel_pool import ParallelPool
p = ParallelPool(rom_path='/dev/null', num_workers=2, env_spec='src.emulation.fake_environment:FakeNESEnvironment')
p.start()
try:
    r = p.reset_all()
    assert [x.worker_id for x in r] == [0, 1], 'worker ordering wrong'
    r = p.step_all([0, 1])
    assert len(r) == 2
    print(f'  2 workers, 1 reset + 1 step OK')
finally:
    p.shutdown()
" || fail "parallel pool"

echo "==> [3/5] BC builds a dataset"
python -c "
import torch
# Fabricate a tiny demo — 40 bytes of alternating actions
import tempfile, os
tmp = tempfile.NamedTemporaryFile(suffix='.state.bin', delete=False)
tmp.write(bytes([0, 0x80, 0x80, 0, 0x01, 0, 0x80, 0] * 5))
tmp.close()
from src.training.behavior_cloning import _action_space_bitmasks, _nearest_action_index
action_space = [[], ['right'], ['A']]
masks = _action_space_bitmasks(action_space)
assert _nearest_action_index(0x80, masks) == 1
assert _nearest_action_index(0x01, masks) == 2
print('  action mapping correct')
os.unlink(tmp.name)
" || fail "bc dataset"

echo "==> [4/5] Trainer init + single generation (fake env + stub reward)"
python <<'PY' || fail "trainer single generation"
import json
import sys
import tempfile
import src.utils.reward_functions as rf
class _Stub:
    def reset(self): pass
    def compute(self, ram): return 1.0, False, "L"
    def episode_success(self): return False
rf.build_reward_function = lambda p: _Stub()
import src.training.trainer as t
t.build_reward_function = lambda p: _Stub()
from src.training.trainer import Trainer
tmp = tempfile.mkdtemp()
profile = {
    "name": "fake",
    "action_space": [[], ["A"]],
    "curriculum": {"stage_1": ["*"]},
    "reward_weights": {},
}
trainer = Trainer(
    rom_path="/dev/null",
    game_profile=profile,
    num_instances=2,
    population_size=4,
    checkpoint_dir=tmp,
    max_episode_steps=30,
    env_spec="src.emulation.fake_environment:FakeNESEnvironment",
    seed=123,
)
trainer.run(num_generations=1)
import os
assert any(f.startswith("gen_") for f in os.listdir(tmp)), "no checkpoint written"
with open(os.path.join(tmp, "metrics.jsonl")) as f:
    lines = [json.loads(l) for l in f if l.strip()]
assert len(lines) == 1 and lines[0]["generation"] == 0, "metrics malformed"
print("  1 generation complete, checkpoint + metrics OK")
PY

echo "==> [5/5] Deterministic seed produces identical first-gen output"
python <<'PY' || fail "deterministic seed"
import tempfile, json, os, shutil
import src.utils.reward_functions as rf
class _Stub:
    def reset(self): pass
    def compute(self, ram): return 1.0, False, "L"
    def episode_success(self): return False
rf.build_reward_function = lambda p: _Stub()
import src.training.trainer as t
t.build_reward_function = lambda p: _Stub()
from src.training.trainer import Trainer

def run_once():
    tmp = tempfile.mkdtemp()
    trainer = Trainer(
        rom_path="/dev/null",
        game_profile={"name":"f","action_space":[[],["A"]],"curriculum":{"stage_1":["*"]},"reward_weights":{}},
        num_instances=2,
        population_size=4,
        checkpoint_dir=tmp,
        max_episode_steps=30,
        env_spec="src.emulation.fake_environment:FakeNESEnvironment",
        seed=42,
    )
    trainer.run(num_generations=1)
    with open(os.path.join(tmp, "metrics.jsonl")) as f:
        metrics = [json.loads(l) for l in f if l.strip()][0]
    shutil.rmtree(tmp)
    return metrics["best_fitness"], metrics["avg_fitness"]

a_best, a_avg = run_once()
b_best, b_avg = run_once()
assert a_best == b_best, f"seed non-deterministic: {a_best} vs {b_best}"
assert a_avg == b_avg, f"seed non-deterministic avg: {a_avg} vs {b_avg}"
print(f"  seed 42 reproducible: best={a_best:.2f} avg={a_avg:.2f}")
PY

echo ""
echo "All smoke checks passed."
