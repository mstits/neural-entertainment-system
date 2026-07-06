//! Narrator event detector — pure logic shared with Python via PyO3.
//!
//! Input: per-step reward-breakdown diffs for each worker. Output:
//! zero or more event dicts per call. Caption phrasing + the GUI
//! queue push live on the Python side; this module owns the hot
//! detection path (delta check, rate limiting, first-ever dedup,
//! combo-kill tracker).

use std::collections::{HashMap, HashSet};

/// Discrete event kinds emitted by the narrator. Mirror
/// `_CAPTION_TEMPLATES` keys on the Python side — adding a new kind
/// requires adding a template entry too.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventKind {
    DungeonEnter,
    NewItem,
    HeartContainer,
    Triforce,
    Key,
    MapCompass,
    Death,
    Success,
    ComboKill,
}

impl EventKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DungeonEnter => "dungeon_enter",
            Self::NewItem => "new_item",
            Self::HeartContainer => "heart_container",
            Self::Triforce => "triforce",
            Self::Key => "key",
            Self::MapCompass => "map_compass",
            Self::Death => "death",
            Self::Success => "success",
            Self::ComboKill => "combo_kill",
        }
    }
}

/// Breakdown signal key → event kind mapping. One signal fires one
/// event kind per step.
const SIGNAL_TO_EVENT: &[(&str, EventKind)] = &[
    ("dungeon_enter", EventKind::DungeonEnter),
    ("new_item", EventKind::NewItem),
    ("heart_container", EventKind::HeartContainer),
    // ZeldaReward emits "triforce", "map", and "compass" (see
    // rewards.rs) — the old "triforce_piece"/"map_compass" keys never
    // matched, so those events could never fire.
    ("triforce", EventKind::Triforce),
    ("key_collected", EventKind::Key),
    ("map", EventKind::MapCompass),
    ("compass", EventKind::MapCompass),
    // Platformer level clear: the `completion` bonus fires once when the
    // agent touches the flagpole / beats the stage. Caption it live as a
    // Success ("CLEARS the stage!") — the most watchable stream moment.
    // The done-path Success only fires on episode-end-with-success, which
    // for SMB (episodes continue across levels, end on death) rarely hits.
    ("completion", EventKind::Success),
];

/// One detected event. Caller decorates with a caption on the Python
/// side (template lookup + genome name substitution).
#[derive(Clone, Debug)]
pub struct Event {
    pub worker_id: u32,
    pub kind: EventKind,
    pub first_ever: bool,
    /// Monotonic seconds-since-start timestamp, filled by the caller.
    pub timestamp_monotonic: f64,
}

pub struct Narrator {
    min_event_gap_s: f64,
    /// (worker_id, kind) → last emission time. Rate-limits noisy kinds.
    last_emit: HashMap<(u32, EventKind), f64>,
    /// First-ever banner is a per-run flag per kind.
    seen_kinds: HashSet<EventKind>,
    /// Per-worker rolling kill counter + last-kill timestamp.
    kill_streak: HashMap<u32, u32>,
    last_kill_t: HashMap<u32, f64>,
    combo_window_s: f64,
    combo_threshold: u32,
}

impl Narrator {
    pub fn new(min_event_gap_s: f64) -> Self {
        Self {
            min_event_gap_s,
            last_emit: HashMap::new(),
            seen_kinds: HashSet::new(),
            kill_streak: HashMap::new(),
            last_kill_t: HashMap::new(),
            combo_window_s: 2.0,
            combo_threshold: 3,
        }
    }

    /// Diff two breakdowns + return the events that fired. The caller
    /// supplies the current monotonic time so parity with Python's
    /// `time.monotonic()` is caller-controlled (important for tests).
    ///
    /// `prev` and `new` are slices of `(key, value)` pairs — matches
    /// the Python dict contract without forcing a HashMap build on
    /// every step.
    pub fn observe(
        &mut self,
        worker_id: u32,
        now: f64,
        prev: &[(&str, f64)],
        new: &[(&str, f64)],
        done: bool,
        success: bool,
    ) -> Vec<Event> {
        let mut out = Vec::new();

        let get = |slice: &[(&str, f64)], k: &str| -> f64 {
            slice
                .iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| *v)
                .unwrap_or(0.0)
        };

        for (sig, kind) in SIGNAL_TO_EVENT {
            let p = get(prev, sig);
            let n = get(new, sig);
            if n > p + 1e-9 {
                if let Some(ev) = self.emit(worker_id, *kind, now, false) {
                    out.push(ev);
                }
            }
        }

        // Enemy-kill combo window.
        let prev_k = get(prev, "enemy_kill");
        let new_k = get(new, "enemy_kill");
        if new_k > prev_k + 1e-9 {
            let in_window = self
                .last_kill_t
                .get(&worker_id)
                .is_some_and(|last_t| now - *last_t <= self.combo_window_s);
            let streak = if in_window {
                self.kill_streak.get(&worker_id).copied().unwrap_or(0) + 1
            } else {
                1
            };
            self.kill_streak.insert(worker_id, streak);
            self.last_kill_t.insert(worker_id, now);
            if streak >= self.combo_threshold {
                if let Some(ev) = self.emit(worker_id, EventKind::ComboKill, now, false) {
                    out.push(ev);
                }
                self.kill_streak.insert(worker_id, 0);
            }
        }

        if done {
            let kind = if success {
                EventKind::Success
            } else {
                EventKind::Death
            };
            if let Some(ev) = self.emit(worker_id, kind, now, true) {
                out.push(ev);
            }
            self.kill_streak.remove(&worker_id);
            self.last_kill_t.remove(&worker_id);
        }

        out
    }

    fn emit(
        &mut self,
        worker_id: u32,
        kind: EventKind,
        now: f64,
        force: bool,
    ) -> Option<Event> {
        let key = (worker_id, kind);
        if !force {
            if let Some(last) = self.last_emit.get(&key) {
                if now - *last < self.min_event_gap_s {
                    return None;
                }
            }
        }
        self.last_emit.insert(key, now);
        let first_ever = !self.seen_kinds.contains(&kind);
        self.seen_kinds.insert(kind);
        Some(Event {
            worker_id,
            kind,
            first_ever,
            timestamp_monotonic: now,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zelda_reward_keys_map_to_events() {
        // The breakdown keys ZeldaReward actually emits must resolve to
        // their narrator events. Before the fix the table listened for
        // "triforce_piece"/"map_compass", so none of these fired.
        let mut n = Narrator::new(0.0);

        let tri = n.observe(0, 0.0, &[("triforce", 0.0)], &[("triforce", 1.0)], false, false);
        assert_eq!(tri.len(), 1);
        assert_eq!(tri[0].kind, EventKind::Triforce);

        let map = n.observe(1, 0.0, &[("map", 0.0)], &[("map", 1.0)], false, false);
        assert_eq!(map.len(), 1);
        assert_eq!(map[0].kind, EventKind::MapCompass);

        let compass = n.observe(2, 0.0, &[("compass", 0.0)], &[("compass", 1.0)], false, false);
        assert_eq!(compass.len(), 1);
        assert_eq!(compass[0].kind, EventKind::MapCompass);
    }

    #[test]
    fn delta_increase_emits() {
        let mut n = Narrator::new(1.5);
        let events = n.observe(
            0,
            0.0,
            &[("dungeon_enter", 0.0)],
            &[("dungeon_enter", 1.0)],
            false,
            false,
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, EventKind::DungeonEnter);
        assert!(events[0].first_ever);
    }

    #[test]
    fn rate_limit_suppresses_repeats() {
        let mut n = Narrator::new(1.5);
        let _ = n.observe(
            0,
            0.0,
            &[("dungeon_enter", 0.0)],
            &[("dungeon_enter", 1.0)],
            false,
            false,
        );
        let ev2 = n.observe(
            0,
            0.5,
            &[("dungeon_enter", 1.0)],
            &[("dungeon_enter", 2.0)],
            false,
            false,
        );
        assert!(ev2.is_empty());
        let ev3 = n.observe(
            0,
            2.0,
            &[("dungeon_enter", 2.0)],
            &[("dungeon_enter", 3.0)],
            false,
            false,
        );
        assert_eq!(ev3.len(), 1);
        assert!(!ev3[0].first_ever);
    }

    #[test]
    fn combo_kill_after_threshold() {
        let mut n = Narrator::new(1.5);
        for i in 0..3 {
            let evs = n.observe(
                0,
                i as f64 * 0.5,
                &[("enemy_kill", i as f64)],
                &[("enemy_kill", (i + 1) as f64)],
                false,
                false,
            );
            // First two kills: no combo. Third kill: combo.
            if i < 2 {
                assert!(evs.is_empty());
            } else {
                assert_eq!(evs.len(), 1);
                assert_eq!(evs[0].kind, EventKind::ComboKill);
            }
        }
    }

    #[test]
    fn terminal_emits_death_or_success() {
        let mut n = Narrator::new(1.5);
        let evs = n.observe(0, 0.0, &[], &[], true, false);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].kind, EventKind::Death);
        let evs2 = n.observe(1, 0.0, &[], &[], true, true);
        assert_eq!(evs2[0].kind, EventKind::Success);
    }
}
