"""Narrator caption behavior — locks in the live-stream narration win.

The narrator/caption pipeline (nes_core.Narrator via src.training.narrator)
feeds the GUI's caption + gold-banner overlay during training. It was
originally Zelda-centric; these tests guard the non-Zelda (SMB/Contra)
captions that make the vanilla_ppo training stream watchable:

  - a `completion` breakdown delta (flagpole / level clear) captions a
    live Success ("CLEARS the stage!") — the most watchable moment;
  - an episode `done` without success captions a Death;
  - signals the narrator doesn't map (e.g. `forward`) stay silent.

These are deterministic (no emulation) so they run in milliseconds and
can't flake. They are the regression guard for the streaming wiring in
Trainer._run_vanilla_ppo + the SIGNAL_TO_EVENT mapping in narrator.rs.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.narrator import Narrator  # noqa: E402


def _fresh() -> Narrator:
    # Zero gap so rapid synthetic events aren't rate-limited away.
    return Narrator(min_event_gap_s=0.0)


def test_completion_delta_captions_a_level_clear():
    """The flagpole moment: completion 0 -> bonus fires a Success event
    with a clear-themed caption. This is the wiring that makes a level
    clear visible on the live stream."""
    n = _fresh()
    n.observe(
        worker_id=2, genome_name="Mario",
        prev_breakdown={"forward": 100.0, "completion": 0.0},
        new_breakdown={"forward": 130.0, "completion": 2000.0},
        done=False, success=False,
    )
    events = n.drain()
    kinds = [e.kind for e in events]
    assert "success" in kinds, f"completion delta should fire a Success event, got {kinds}"
    clear_ev = next(e for e in events if e.kind == "success")
    assert "Mario" in clear_ev.caption


def test_done_without_success_is_a_death():
    n = _fresh()
    n.observe(
        worker_id=7, genome_name="Luigi",
        prev_breakdown={"forward": 50.0},
        new_breakdown={"forward": 50.0, "death": -15.0},
        done=True, success=False,
    )
    kinds = [e.kind for e in n.drain()]
    assert kinds == ["death"], f"a done-without-success should caption a single Death, got {kinds}"


def test_done_with_success_is_a_clear():
    n = _fresh()
    n.observe(
        worker_id=1, genome_name="Mario",
        prev_breakdown={"forward": 3000.0},
        new_breakdown={"forward": 3100.0},
        done=True, success=True,
    )
    kinds = [e.kind for e in n.drain()]
    assert "success" in kinds, f"a successful episode end should caption Success, got {kinds}"


def test_unmapped_signal_stays_silent():
    """Forward progress alone is not a caption-worthy event — only the
    mapped signals (completion, Zelda items) and episode-end fire. This
    guards against caption spam on every step."""
    n = _fresh()
    n.observe(
        worker_id=0, genome_name="agent",
        prev_breakdown={"forward": 100.0},
        new_breakdown={"forward": 250.0},  # only forward moved
        done=False, success=False,
    )
    assert n.drain() == [], "plain forward progress must not emit a caption"


def test_drain_empties_the_queue():
    n = _fresh()
    n.observe(
        worker_id=0, genome_name="agent",
        prev_breakdown={"completion": 0.0},
        new_breakdown={"completion": 2000.0},
        done=False, success=False,
    )
    first = n.drain()
    assert len(first) >= 1
    assert n.drain() == [], "drain must empty the event queue"
