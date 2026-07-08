//! Per-step reward functions for every supported game.
//!
//! Ports `src/utils/reward_functions/` from Python to Rust. The hot
//! path is called once per worker per step (16 workers × thousands of
//! steps per gen), so moving it off Python eliminates a noticeable
//! GIL-serialized chunk from `_evaluate_batch`.
//!
//! Each game is a single struct with per-frame state (previous RAM
//! values, sets of visited screens, "already died" flags, etc.) and
//! one `compute(ram, action) -> RewardOutput` method. Semantics match
//! the Python originals byte-for-byte — verified via
//! `scripts/test_rewards_parity.py` which feeds synthetic RAM to both
//! implementations and diffs the outputs.
//!
//! The Python trainer constructs these through
//! `nes_core.build_reward_function(game_profile_dict)` — a factory
//! that inspects `profile["name"]` and `profile["reward_weights"]`
//! and dispatches to the right enum variant. Same surface as the
//! Python `build_reward_function` it replaces.

use std::collections::{HashMap, HashSet};

/// Output of `compute()` — reward for this step, whether the episode
/// should terminate, the level-ID string the trainer uses for the
/// curriculum stats, and the per-signal breakdown delta this step
/// contributed. The trainer accumulates the breakdown across the
/// episode.
pub struct RewardOutput {
    pub reward: f64,
    pub done: bool,
    pub level_id: String,
    /// Only populated when the caller asks for the breakdown (the
    /// narrator path). Hot-path `compute()` callers pass
    /// `want_breakdown=false` and the vec stays empty.
    pub breakdown_delta: Vec<(&'static str, f64)>,
}

/// The enum of per-game reward functions. Boxed inside the PyO3
/// wrapper so adding a game is a single variant + match arm.
pub enum Reward {
    Zelda(ZeldaReward),
    Mario(MarioReward),
    Contra(ContraReward),
    MegaMan(MegaManReward),
    Castlevania(CastlevaniaReward),
    Metroid(MetroidReward),
    Tetris(TetrisReward),
    BubbleBobble(BubbleBobbleReward),
    /// Fallback for any ROM without a hand-crafted reward function.
    /// Uses generic progress signals: RAM churn as motion proxy,
    /// survival-per-step, time penalty, and auto-detected score
    /// bytes (addresses that monotonically increase during play).
    /// Every one of the 794 library ROMs can train on this out of
    /// the box; hand-authored rewards per game replace it as they
    /// get written.
    Generic(GenericReward),
}

impl Reward {
    pub fn reset(&mut self) {
        match self {
            Reward::Zelda(r) => r.reset(),
            Reward::Mario(r) => r.reset(),
            Reward::Contra(r) => r.reset(),
            Reward::MegaMan(r) => r.reset(),
            Reward::Castlevania(r) => r.reset(),
            Reward::Metroid(r) => r.reset(),
            Reward::Tetris(r) => r.reset(),
            Reward::BubbleBobble(r) => r.reset(),
            Reward::Generic(r) => r.reset(),
        }
    }

    pub fn compute(&mut self, ram: &[u8], action: u8, want_breakdown: bool) -> RewardOutput {
        match self {
            Reward::Zelda(r) => r.compute(ram, action, want_breakdown),
            Reward::Mario(r) => r.compute(ram, action, want_breakdown),
            Reward::Contra(r) => r.compute(ram, action, want_breakdown),
            Reward::MegaMan(r) => r.compute(ram, action, want_breakdown),
            Reward::Castlevania(r) => r.compute(ram, action, want_breakdown),
            Reward::Metroid(r) => r.compute(ram, action, want_breakdown),
            Reward::Tetris(r) => r.compute(ram, action, want_breakdown),
            Reward::BubbleBobble(r) => r.compute(ram, action, want_breakdown),
            Reward::Generic(r) => r.compute(ram, action, want_breakdown),
        }
    }

    pub fn episode_success(&self) -> bool {
        match self {
            Reward::Zelda(r) => r.episode_success(),
            Reward::Mario(r) => r.episode_success(),
            Reward::Contra(r) => r.episode_success(),
            Reward::MegaMan(r) => r.episode_success(),
            Reward::Castlevania(r) => r.episode_success(),
            Reward::Metroid(r) => r.episode_success(),
            Reward::Tetris(r) => r.episode_success(),
            Reward::BubbleBobble(r) => r.episode_success(),
            Reward::Generic(r) => r.episode_success(),
        }
    }
}

// ============================================================
// Generic reward — works on any NES ROM with no hand-authored
// RAM knowledge. Uses dynamic signals the trainer can compute
// purely from the RAM observation + step count.
// ============================================================

/// State for the fallback reward function.
///
/// **Signals combined:**
/// - **motion**: fraction of 2 KB RAM bytes that changed since the
///   last step. Proxies "something is happening" — interactive
///   screens, scrolling, enemies moving, etc.
/// - **survival**: small positive per step. Offsets the time penalty
///   so trivially-stuck games don't train to suicide.
/// - **time penalty**: small negative per step. Biases toward
///   completing the scenario rather than stalling.
/// - **score hunt**: tracks per-byte monotonic-increase frequency
///   over a warmup window (first 300 steps). Any byte that increased
///   much more than it decreased joins the "score candidates" set;
///   positive deltas on those bytes earn score_weight × delta reward.
/// - **stuck detection**: if RAM has had zero bytes change for
///   stuck_steps consecutive ticks, episode ends (probably on a
///   menu / pause screen / post-death).
#[derive(Clone)]
pub struct GenericReward {
    // Configurable weights (from profile reward_weights, defaults below).
    motion_weight: f64,
    survival_weight: f64,
    time_penalty: f64,
    score_weight: f64,
    stuck_steps: usize,
    warmup_steps: usize,

    // Dynamic state.
    last_ram: Vec<u8>,
    total: f64,
    steps: usize,
    stuck_counter: usize,
    // For each RAM byte, how many times it increased and decreased
    // during the warmup window. After warmup, we freeze the set of
    // "score candidate" addresses (increased much more than
    // decreased) and reward positive deltas on those only.
    byte_increased: Vec<u32>,
    byte_decreased: Vec<u32>,
    score_candidates: HashSet<usize>,
    score_frozen: bool,
}

impl GenericReward {
    pub fn new(
        motion_weight: f64,
        survival_weight: f64,
        time_penalty: f64,
        score_weight: f64,
        stuck_steps: usize,
        warmup_steps: usize,
    ) -> Self {
        Self {
            motion_weight,
            survival_weight,
            time_penalty,
            score_weight,
            stuck_steps,
            warmup_steps,
            last_ram: Vec::new(),
            total: 0.0,
            steps: 0,
            stuck_counter: 0,
            byte_increased: vec![0; 2048],
            byte_decreased: vec![0; 2048],
            score_candidates: HashSet::new(),
            score_frozen: false,
        }
    }

    pub fn reset(&mut self) {
        self.last_ram.clear();
        self.total = 0.0;
        self.steps = 0;
        self.stuck_counter = 0;
        self.byte_increased.iter_mut().for_each(|v| *v = 0);
        self.byte_decreased.iter_mut().for_each(|v| *v = 0);
        self.score_candidates.clear();
        self.score_frozen = false;
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let mut accum = RewardAccum::new(want_breakdown);

        // First call per episode: initialise last_ram to current state.
        // No motion signal this step — we need a baseline to diff
        // against.
        if self.last_ram.is_empty() || self.last_ram.len() != ram.len() {
            self.last_ram = ram.to_vec();
            self.steps += 1;
            accum.add("survival", self.survival_weight);
            return RewardOutput {
                reward: accum.total,
                done: false,
                level_id: "stage_1".to_string(),
                breakdown_delta: accum.breakdown,
            };
        }

        // 1. Motion signal — fraction of bytes that changed.
        let mut changed = 0usize;
        let limit = ram.len().min(self.last_ram.len()).min(2048);
        for i in 0..limit {
            let cur = ram[i];
            let prev = self.last_ram[i];
            if cur != prev {
                changed += 1;
                // Score-hunt bookkeeping during warmup.
                if !self.score_frozen {
                    if cur > prev {
                        self.byte_increased[i] =
                            self.byte_increased[i].saturating_add(1);
                    } else {
                        self.byte_decreased[i] =
                            self.byte_decreased[i].saturating_add(1);
                    }
                }
            }
        }
        let motion = changed as f64 / (limit.max(1) as f64);
        accum.add("motion", self.motion_weight * motion);

        // 2. Survival + time penalty.
        accum.add("survival", self.survival_weight);
        accum.add("time", self.time_penalty);

        // 3. Score-candidate reward once warmup is done.
        //
        // Freeze candidates at warmup_steps: any byte that increased
        // at least 4× as often as it decreased (and increased at
        // least 3 times total) is a plausible score counter. Empirically
        // covers most NES score displays without false-positiving
        // animation counters that flap up/down.
        if !self.score_frozen && self.steps >= self.warmup_steps {
            for i in 0..limit {
                let up = self.byte_increased[i];
                let down = self.byte_decreased[i];
                if up >= 3 && up >= down.saturating_mul(4) {
                    self.score_candidates.insert(i);
                }
            }
            self.score_frozen = true;
        }
        if self.score_frozen && self.score_weight != 0.0 {
            let mut score_delta = 0.0f64;
            for &addr in self.score_candidates.iter() {
                let cur = ram[addr] as i32;
                let prev = self.last_ram[addr] as i32;
                let diff = cur - prev;
                if diff > 0 {
                    score_delta += diff as f64;
                }
            }
            if score_delta > 0.0 {
                accum.add("score", self.score_weight * score_delta);
            }
        }

        // 4. Stuck detection — done when RAM is frozen long enough
        // that the game is almost certainly on a menu or in a dead-end
        // state we can't escape from.
        if changed == 0 {
            self.stuck_counter = self.stuck_counter.saturating_add(1);
        } else {
            self.stuck_counter = 0;
        }
        let done = self.stuck_counter >= self.stuck_steps;

        let n = self.last_ram.len().min(ram.len());
        self.last_ram[..n].copy_from_slice(&ram[..n]);
        self.steps += 1;
        self.total += accum.total;

        RewardOutput {
            reward: accum.total,
            done,
            level_id: "stage_1".to_string(),
            breakdown_delta: accum.breakdown,
        }
    }

    pub fn episode_success(&self) -> bool {
        // No built-in notion of "success" without game-specific
        // knowledge. Report positive cumulative reward as a
        // best-effort proxy — good enough for curriculum auto-promotion
        // until a hand-crafted reward takes over.
        self.total > 0.0
    }
}

/// Helper used by every game — accumulates a reward component into a
/// running total AND records it in the breakdown delta when asked.
struct RewardAccum {
    total: f64,
    made_progress: bool,
    breakdown: Vec<(&'static str, f64)>,
    want_breakdown: bool,
}
impl RewardAccum {
    fn new(want_breakdown: bool) -> Self {
        Self {
            total: 0.0,
            made_progress: false,
            breakdown: Vec::new(),
            want_breakdown,
        }
    }
    fn add(&mut self, signal: &'static str, amount: f64) {
        self.total += amount;
        if amount > 0.0 {
            self.made_progress = true;
        }
        if amount != 0.0 && self.want_breakdown {
            self.breakdown.push((signal, amount));
        }
    }
}

// ============================================================
// Zelda — the 443-line beast, ported
// ============================================================

#[derive(Clone)]
pub struct ZeldaReward {
    // Weights
    exploration_bonus: f64,
    rupee_weight: f64,
    heart_container: f64,
    heart_recovery: f64,
    triforce_piece: f64,
    damage_penalty: f64,
    death_penalty: f64,
    time_penalty: f64,
    motion_weight: f64,
    stagnation_penalty: f64,
    stagnation_window: usize,
    wall_bump_penalty: f64,
    enemy_kill: f64,
    dungeon_enter_bonus: f64,
    new_item_bonus: f64,
    item_tier_upgrade_bonus: f64,
    first_sword_bonus: f64,
    key_collected_bonus: f64,
    magic_key_bonus: f64,
    map_compass_bonus: f64,
    item_used_bonus: f64,
    item_cycle_bonus: f64,
    activity_timeout_steps: usize,
    // Win-chain milestone weights (progress-to-Ganon must dominate the
    // exploration bonus so the motion/explore local optimum can't win).
    all_fragments_bonus: f64,
    level9_enter_bonus: f64,
    ganon_reached_bonus: f64,
    win_bonus: f64,
    // State
    won: bool,
    all_fragments_awarded: bool,
    level9_entered: bool,
    ganon_reached: bool,
    visited: HashSet<(u8, u8)>,
    dungeons_entered: HashSet<u8>,
    prev_hearts_byte: u8,
    prev_partial_hearts: u8,
    prev_rupees: u8,
    prev_triforce: u8,
    prev_link_x: u8,
    prev_link_y: u8,
    prev_kill_count: u8,
    idle_steps: usize,
    prev_inventory: [u8; 16], // RAM $0657..$0666 (inclusive)
    prev_b_item: u8,
    prev_bombs: u8,
    prev_arrows: u8,
    prev_keys: u8,
    first_sword_awarded: bool,
    steps_since_progress: usize,
    prev_magic_key: u8,
    prev_map_bits: u8,
    prev_compass_bits: u8,
    died: bool,
    first_step: bool,
}

impl ZeldaReward {
    const RAM_LINK_X: usize = 0x0070;
    const RAM_LINK_Y: usize = 0x0084;
    const RAM_MAP_X: usize = 0x00EB;
    const RAM_MAP_Y: usize = 0x00EC;
    const RAM_HEARTS: usize = 0x066F;
    // Fill level of Link's current (topmost, partially-filled) heart.
    // Nonzero means Link still has a fraction of a heart left even when
    // the full-heart nibble of $066F already reads 0.
    const RAM_PARTIAL_HEARTS: usize = 0x0670;
    const RAM_RUPEES: usize = 0x066D;
    const RAM_TRIFORCE: usize = 0x0671;
    const RAM_DUNGEON_LEVEL: usize = 0x10;
    // Win chain (aldonunez disassembly + empirically verified against the
    // captured zelda_*.state.bin on this emulator):
    //   $0672 LastBossDefeated — 0 while Ganon lives, set to 1 the instant
    //     his death fanfare ends. Monotonic 0->1; the ONE reliable "game
    //     effectively won" flag (rescuing Zelda after is a scripted walk).
    //   $0609 Song — one-hot active track: 0x10=Ending theme (Zelda rescued
    //     / credits), 0x02=Ganon-fight theme (reached the final boss).
    // $0671==0xFF (all 8 fragments) only opens Level 9 — a MILESTONE, not a
    // win; the old episode_success() wrongly treated it as victory.
    const RAM_GANON_DEFEATED: usize = 0x0672;
    const RAM_SONG: usize = 0x0609;
    const SONG_ENDING: u8 = 0x10;
    const SONG_GANON: u8 = 0x02;
    const RAM_KILL_COUNT: usize = 0x0627;
    const RAM_INVENTORY_START: usize = 0x0657;
    const RAM_B_ITEM: usize = 0x0656;
    const RAM_SWORD: usize = 0x0657;
    const RAM_BOMBS: usize = 0x0658;
    const RAM_ARROWS: usize = 0x0659;
    const RAM_MAGIC_KEY: usize = 0x0664;
    const RAM_KEYS: usize = 0x066E;
    const RAM_MAP_BITS: usize = 0x0668;
    const RAM_COMPASS_BITS: usize = 0x0667;
    const DIRECTIONAL_MASK: u8 = 0x80 | 0x40 | 0x20 | 0x10;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        exploration_bonus: f64,
        rupee_weight: f64,
        heart_container: f64,
        heart_recovery: f64,
        triforce_piece: f64,
        damage_penalty: f64,
        death_penalty: f64,
        time_penalty: f64,
        motion_weight: f64,
        stagnation_penalty: f64,
        stagnation_window: usize,
        wall_bump_penalty: f64,
        enemy_kill: f64,
        dungeon_enter_bonus: f64,
        new_item_bonus: f64,
        item_tier_upgrade_bonus: f64,
        first_sword_bonus: f64,
        key_collected_bonus: f64,
        magic_key_bonus: f64,
        map_compass_bonus: f64,
        item_used_bonus: f64,
        item_cycle_bonus: f64,
        activity_timeout_steps: usize,
        all_fragments_bonus: f64,
        level9_enter_bonus: f64,
        ganon_reached_bonus: f64,
        win_bonus: f64,
    ) -> Self {
        Self {
            exploration_bonus,
            rupee_weight,
            heart_container,
            heart_recovery,
            triforce_piece,
            damage_penalty,
            death_penalty,
            time_penalty,
            motion_weight,
            stagnation_penalty,
            stagnation_window,
            wall_bump_penalty,
            enemy_kill,
            dungeon_enter_bonus,
            new_item_bonus,
            item_tier_upgrade_bonus,
            first_sword_bonus,
            key_collected_bonus,
            magic_key_bonus,
            map_compass_bonus,
            item_used_bonus,
            item_cycle_bonus,
            activity_timeout_steps,
            all_fragments_bonus,
            level9_enter_bonus,
            ganon_reached_bonus,
            win_bonus,
            won: false,
            all_fragments_awarded: false,
            level9_entered: false,
            ganon_reached: false,
            visited: HashSet::new(),
            dungeons_entered: HashSet::new(),
            prev_hearts_byte: 0,
            prev_partial_hearts: 0,
            prev_rupees: 0,
            prev_triforce: 0,
            prev_link_x: 0,
            prev_link_y: 0,
            prev_kill_count: 0,
            idle_steps: 0,
            prev_inventory: [0; 16],
            prev_b_item: 0,
            prev_bombs: 0,
            prev_arrows: 0,
            prev_keys: 0,
            first_sword_awarded: false,
            steps_since_progress: 0,
            prev_magic_key: 0,
            prev_map_bits: 0,
            prev_compass_bits: 0,
            died: false,
            first_step: true,
        }
    }

    pub fn reset(&mut self) {
        self.visited.clear();
        self.dungeons_entered.clear();
        self.prev_hearts_byte = 0;
        self.prev_partial_hearts = 0;
        self.prev_rupees = 0;
        self.prev_triforce = 0;
        self.prev_link_x = 0;
        self.prev_link_y = 0;
        self.prev_kill_count = 0;
        self.idle_steps = 0;
        self.prev_inventory = [0; 16];
        self.prev_b_item = 0;
        self.prev_bombs = 0;
        self.prev_arrows = 0;
        self.prev_keys = 0;
        self.first_sword_awarded = false;
        self.steps_since_progress = 0;
        self.prev_magic_key = 0;
        self.prev_map_bits = 0;
        self.prev_compass_bits = 0;
        self.won = false;
        self.all_fragments_awarded = false;
        self.level9_entered = false;
        self.ganon_reached = false;
        self.died = false;
        self.first_step = true;
    }

    fn hearts_current(byte: u8) -> u8 {
        byte & 0x0F
    }
    fn hearts_max(byte: u8) -> u8 {
        (byte >> 4) & 0x0F
    }

    pub fn compute(&mut self, ram: &[u8], action: u8, want_breakdown: bool) -> RewardOutput {
        let map_cell = (ram[Self::RAM_MAP_X], ram[Self::RAM_MAP_Y]);
        let hearts_byte = ram[Self::RAM_HEARTS];
        let partial_hearts = ram[Self::RAM_PARTIAL_HEARTS];
        let rupees = ram[Self::RAM_RUPEES];
        let triforce_bits = ram[Self::RAM_TRIFORCE];
        let link_x = ram[Self::RAM_LINK_X];
        let link_y = ram[Self::RAM_LINK_Y];
        let dungeon_level = ram[Self::RAM_DUNGEON_LEVEL];
        let kill_count = ram[Self::RAM_KILL_COUNT];
        let mut inventory = [0u8; 16];
        for (i, slot) in inventory.iter_mut().enumerate() {
            *slot = ram[Self::RAM_INVENTORY_START + i];
        }
        let b_item = ram[Self::RAM_B_ITEM];
        let bombs = ram[Self::RAM_BOMBS];
        let arrows = ram[Self::RAM_ARROWS];
        let keys = ram[Self::RAM_KEYS];
        let sword = ram[Self::RAM_SWORD];
        let magic_key = ram[Self::RAM_MAGIC_KEY];
        let map_bits = ram[Self::RAM_MAP_BITS];
        let compass_bits = ram[Self::RAM_COMPASS_BITS];

        if self.first_step {
            self.prev_hearts_byte = hearts_byte;
            self.prev_partial_hearts = partial_hearts;
            self.prev_rupees = rupees;
            self.prev_triforce = triforce_bits;
            self.prev_link_x = link_x;
            self.prev_link_y = link_y;
            self.prev_kill_count = kill_count;
            self.prev_inventory = inventory;
            self.prev_b_item = b_item;
            self.prev_bombs = bombs;
            self.prev_arrows = arrows;
            self.prev_keys = keys;
            self.first_sword_awarded = sword != 0;
            self.prev_magic_key = magic_key;
            self.prev_map_bits = map_bits;
            self.prev_compass_bits = compass_bits;
            self.visited.insert(map_cell);
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        if !self.visited.contains(&map_cell) {
            self.visited.insert(map_cell);
            acc.add("exploration", self.exploration_bonus);
        }

        let dx = (link_x as i32 - self.prev_link_x as i32).unsigned_abs();
        let dy = (link_y as i32 - self.prev_link_y as i32).unsigned_abs();
        if self.motion_weight > 0.0 {
            acc.add("motion", self.motion_weight * (dx + dy) as f64);
        }

        if dx == 0 && dy == 0 {
            self.idle_steps += 1;
            if self.idle_steps > self.stagnation_window && self.stagnation_penalty != 0.0 {
                acc.add("stagnation", self.stagnation_penalty);
            }
        } else {
            self.idle_steps = 0;
        }

        if self.wall_bump_penalty != 0.0
            && (action & Self::DIRECTIONAL_MASK) != 0
            && dx == 0
            && dy == 0
        {
            acc.add("wall_bump", self.wall_bump_penalty);
        }

        self.prev_link_x = link_x;
        self.prev_link_y = link_y;

        if rupees > self.prev_rupees {
            acc.add("rupee", self.rupee_weight * (rupees - self.prev_rupees) as f64);
        }
        self.prev_rupees = rupees;

        if dungeon_level != 0 && !self.dungeons_entered.contains(&dungeon_level) {
            self.dungeons_entered.insert(dungeon_level);
            acc.add("dungeon_enter", self.dungeon_enter_bonus);
        }

        if kill_count > self.prev_kill_count {
            acc.add(
                "enemy_kill",
                self.enemy_kill * (kill_count - self.prev_kill_count) as f64,
            );
        }
        self.prev_kill_count = kill_count;

        let prev_max = Self::hearts_max(self.prev_hearts_byte);
        let curr_max = Self::hearts_max(hearts_byte);
        if curr_max > prev_max {
            acc.add(
                "heart_container",
                self.heart_container * (curr_max - prev_max) as f64,
            );
        }
        let prev_cur = Self::hearts_current(self.prev_hearts_byte);
        let curr_cur = Self::hearts_current(hearts_byte);
        if curr_cur > prev_cur && curr_max == prev_max {
            acc.add(
                "heart_recovery",
                self.heart_recovery * (curr_cur - prev_cur) as f64,
            );
        }
        if curr_cur < prev_cur {
            acc.add(
                "damage",
                self.damage_penalty * (prev_cur - curr_cur) as f64,
            );
        }
        self.prev_hearts_byte = hearts_byte;

        // Link is dead only when BOTH the full-heart nibble ($066F low
        // nibble) AND the partial-heart byte ($0670) reach zero. A
        // remaining half heart (partial_hearts != 0) means Link is still
        // alive even though the full-heart nibble already reads 0, so it
        // must not trigger the death penalty / episode end. The previous
        // partial value is tracked so the *actual* death — which lands one
        // or more steps after the full-heart nibble first hits 0 — is
        // still detected (by then prev_cur is already 0).
        let was_alive = prev_cur > 0 || self.prev_partial_hearts > 0;
        if was_alive && curr_cur == 0 && partial_hearts == 0 {
            acc.add("death", self.death_penalty);
            self.died = true;
            done = true;
        }
        self.prev_partial_hearts = partial_hearts;

        if !self.first_sword_awarded && sword != 0 {
            acc.add("first_sword", self.first_sword_bonus);
            self.first_sword_awarded = true;
        }

        let mut newly_acquired = 0u32;
        let mut upgrades = 0u32;
        for (&prev, &cur) in self.prev_inventory.iter().zip(inventory.iter()) {
            if prev == 0 && cur != 0 {
                newly_acquired += 1;
            } else if cur > prev && prev != 0 {
                upgrades += 1;
            }
        }
        if newly_acquired > 0 {
            acc.add("new_item", self.new_item_bonus * newly_acquired as f64);
        }
        if upgrades > 0 {
            acc.add(
                "item_tier_upgrade",
                self.item_tier_upgrade_bonus * upgrades as f64,
            );
        }
        self.prev_inventory = inventory;

        if keys > self.prev_keys && self.key_collected_bonus != 0.0 {
            acc.add(
                "key_collected",
                self.key_collected_bonus * (keys - self.prev_keys) as f64,
            );
        }
        self.prev_keys = keys;

        if magic_key > self.prev_magic_key {
            acc.add("magic_key", self.magic_key_bonus);
        }
        self.prev_magic_key = magic_key;

        let new_map_bits = (map_bits & !self.prev_map_bits).count_ones() as f64;
        let new_compass_bits = (compass_bits & !self.prev_compass_bits).count_ones() as f64;
        if new_map_bits > 0.0 {
            acc.add("map", self.map_compass_bonus * new_map_bits);
        }
        if new_compass_bits > 0.0 {
            acc.add("compass", self.map_compass_bonus * new_compass_bits);
        }
        self.prev_map_bits = map_bits;
        self.prev_compass_bits = compass_bits;

        if bombs < self.prev_bombs {
            acc.add(
                "item_used",
                self.item_used_bonus * (self.prev_bombs - bombs) as f64,
            );
        }
        if arrows < self.prev_arrows {
            acc.add(
                "item_used",
                self.item_used_bonus * (self.prev_arrows - arrows) as f64,
            );
        }
        self.prev_bombs = bombs;
        self.prev_arrows = arrows;

        if b_item != self.prev_b_item && self.prev_b_item != 0 && b_item != 0 {
            acc.add("item_cycle", self.item_cycle_bonus);
        }
        self.prev_b_item = b_item;

        let new_pieces = (triforce_bits & !self.prev_triforce).count_ones() as f64;
        if new_pieces > 0.0 {
            acc.add("triforce", self.triforce_piece * new_pieces);
        }
        self.prev_triforce = triforce_bits;

        // Win-chain milestones (each fires ONCE) so progress toward Ganon is
        // the dominant attractor over exploration. Ordered by lateness:
        //   all 8 fragments ($0671==0xFF, Level 9 openable) -> +all_fragments
        //   entered Level 9 ($0010==9)                      -> +level9_enter
        //   reached the Ganon fight (song 0x02)             -> +ganon_reached
        //   Ganon defeated ($0672!=0) / ending song (0x10)  -> +win_bonus, WIN
        let ganon_defeated_byte = ram[Self::RAM_GANON_DEFEATED];
        let song = ram[Self::RAM_SONG];
        if !self.all_fragments_awarded && triforce_bits == 0xFF {
            acc.add("all_fragments", self.all_fragments_bonus);
            self.all_fragments_awarded = true;
        }
        if !self.level9_entered && dungeon_level == 9 {
            acc.add("level9_enter", self.level9_enter_bonus);
            self.level9_entered = true;
        }
        if !self.ganon_reached && (song & Self::SONG_GANON) != 0 {
            acc.add("ganon_reached", self.ganon_reached_bonus);
            self.ganon_reached = true;
        }
        // Terminal: Ganon dead (the monotonic $0672 flag) or the ending theme
        // playing (Zelda rescued). Fire the jackpot once and end the episode.
        if !self.won && (ganon_defeated_byte != 0 || (song & Self::SONG_ENDING) != 0) {
            acc.add("win", self.win_bonus);
            self.won = true;
            done = true;
        }

        acc.add("time_penalty", self.time_penalty);

        if self.activity_timeout_steps > 0 {
            if acc.made_progress {
                self.steps_since_progress = 0;
            } else {
                self.steps_since_progress += 1;
            }
            if self.steps_since_progress >= self.activity_timeout_steps {
                done = true;
            }
        }

        let level_id = format!("map_{:02x}_{:02x}", map_cell.0, map_cell.1);
        RewardOutput {
            reward: acc.total,
            done,
            level_id,
            breakdown_delta: acc.breakdown,
        }
    }

    pub fn episode_success(&self) -> bool {
        // WIN = Ganon defeated ($0672) / Zelda rescued (ending song), latched
        // in `won`. The old predicate (all 8 fragments collected) declared
        // victory ~3 dungeons + the final boss too early — collecting every
        // fragment only OPENS Level 9; it is not winning the game.
        self.won && !self.died
    }
}

// ============================================================
// Mario — Super Mario Bros. NES
// ============================================================

#[derive(Clone)]
pub struct MarioReward {
    forward_weight: f64,
    score_weight: f64,
    death_penalty: f64,
    completion_bonus: f64,
    time_penalty: f64,
    /// Multiplier on per-checkpoint bonuses. `0.0` disables the dense
    /// reward shaping entirely (recovers the prior pure-progress
    /// behavior). `1.0` is the recommended default for tile-mode
    /// training, where PPO needs strong intermediate signal to learn.
    checkpoint_scale: f64,
    /// Per-step bonus when Mario is airborne AND moving right. Targets
    /// the "needs to learn jumping" failure mode observed at
    /// depth_scalar=1275 (first pipe before staircase) where the
    /// policy converged on "run right on flat ground" but couldn't
    /// commit to the run+jump action sequence required to clear
    /// pipes. `forward_weight * vel_x` rewards horizontal speed only;
    /// `air_bonus_weight` adds a vertical signal so PPO sees positive
    /// gradient on action 4 (right+A+B = run-jump-right).
    ///
    /// Gating on vel_x>0 is critical — otherwise the agent learns to
    /// jump in place (action 5 = "A") for free reward without
    /// progressing. With the gate, the bonus only fires when Mario
    /// is genuinely jumping while running forward.
    ///
    /// 0.0 disables. Suggested values: 0.5 to 1.0. Higher than
    /// forward_weight × Mario's max speed (~30) would dominate the
    /// per-step gradient and over-bias toward jumping.
    air_bonus_weight: f64,
    /// Death-cause differentiation. The base `death_penalty` always
    /// fires on death; these add an EXTRA penalty depending on HOW
    /// Mario died, giving PPO a sharper gradient than a single
    /// undifferentiated -15:
    ///
    /// * `pit_fall_extra`: triggered when Mario was airborne with
    ///   Y > 200 at the moment of death (he was falling through a
    ///   pit). Suggested -15 to -30. Teaches the policy to commit
    ///   to jumps EARLIER on approach to gaps.
    /// * `enemy_death_extra`: triggered when an active enemy is
    ///   within 1 tile of Mario at the moment of death. Teaches
    ///   the policy that enemy contact while small is dangerous
    ///   (vs. jumping ON top of an enemy, which DOESN'T fire this
    ///   penalty because the contact frame typically has the enemy
    ///   already despawned/transitioning).
    ///
    /// Both 0.0 = backward compat (single death penalty as before).
    pit_fall_extra: f64,
    enemy_death_extra: f64,
    /// Periodic "still alive" bonus. Fires +`survival_bonus` once
    /// every `survival_block_steps` compute() calls. Both 0 disables
    /// (back-compat default). Useful for breaking plateaus where the
    /// agent needs gradient on "stay alive long enough to attempt the
    /// hard maneuver" — episodes that die early earn nothing, episodes
    /// that survive past the first 150 steps get +300, etc. Stacks
    /// across blocks: surviving 600 steps earns 4× the bonus.
    survival_bonus: f64,
    survival_block_steps: u32,
    /// Counts compute() calls within the current episode. Reset in
    /// reset() (between episodes) but NOT on level transition — the
    /// "still alive" signal is genuinely about episode duration, and
    /// resetting would make level transitions briefly erase the
    /// progress credit.
    step_count: u32,
    /// Bonus fired ONCE per successful jump landing where Mario covered
    /// `jump_clear_min_dx` X-units of forward progress while airborne.
    /// Approximates "cleared a pit / hopped a Goomba" without needing
    /// per-tile pit detection: any productive jump-and-land pattern
    /// fires it. Designed as positive-reinforcement counterpart to
    /// pit_fall_extra (which the same scheme PUNISHES). Set both:
    /// pit_fall_extra=0 + jump_clear_bonus>0 = pure carrot.
    jump_clear_bonus: f64,
    jump_clear_min_dx: u32,
    /// Was Mario on the ground at the previous compute() call. Used to
    /// detect ground→air and air→ground transitions for the bonus.
    prev_on_ground: bool,
    /// X-position at the moment Mario left the ground. Difference vs
    /// current x at landing-time = horizontal distance covered in air.
    air_start_x: u32,
    prev_x: u32,
    /// Furthest X position reached this episode. Used to gate the
    /// `forward` reward on NEW progress only — backtracking earns 0,
    /// not negative — which prevents the agent from learning "always
    /// press right out of fear of losing reward" and lets it back up
    /// to time jumps. Reset on level transition + episode start.
    max_x_visited: u32,
    /// Index of the next checkpoint not yet awarded this episode. We
    /// only fire each checkpoint once — once Mario has crossed
    /// `CHECKPOINTS[i].0`, the bonus is added and `next_checkpoint`
    /// advances. Reset on episode start + level transition.
    next_checkpoint: usize,
    prev_lives: u8,
    prev_score: u64,
    prev_world: u8,
    prev_level: u8,
    completed: bool,
    died: bool,
    first_step: bool,
}

impl MarioReward {
    const RAM_X_PAGE: usize = 0x006D;
    const RAM_X_LOW: usize = 0x0086;
    const RAM_LIVES: usize = 0x075A;
    const RAM_PLAYER_STATE: usize = 0x000E;
    const RAM_WORLD: usize = 0x075F;
    const RAM_LEVEL: usize = 0x0760;
    const RAM_FLOAT_STATE: usize = 0x001D;
    const PLAYER_STATE_DYING: u8 = 0x0B;
    const PLAYER_STATE_DEAD: u8 = 0x06;

    /// Per-level progress milestones (x-position threshold, base bonus).
    /// Bonuses are scaled by the `checkpoint_scale` weight at runtime
    /// so users can tune the strength globally without per-checkpoint
    /// YAML knobs.
    ///
    /// Why per-level tables instead of one shared list: 1-1 has a
    /// staircase at x=2700; 1-2 is a vertical underground level with
    /// no staircase at all and totally different obstacle x-positions.
    /// Sharing the table would have 1-1's bonuses fire opportunistically
    /// at random spots in 1-2 and provide no signal at the actual
    /// 1-2 obstacles. Per-level tables preserve dense gradient signal
    /// across the curriculum.
    ///
    /// `checkpoints_for(world, level)` returns the right table; the
    /// `next_checkpoint` cursor is reset on every world/level
    /// transition (already handled in `compute`).
    const LEVEL_1_1: &'static [(u32, f64)] = &[
        (350, 50.0),    // past first pipe
        (720, 100.0),   // past first pit
        (1100, 150.0),  // past first horizontal pipe
        (1640, 250.0),  // past large pit
        // 2026-05-07 — bridge the 1640→2100 dead zone. The
        // population pinned at depth ≈1956 for 50+ gens because no
        // dense signal exists between the large-pit reward and the
        // obstacle-complex reward. A genome that crosses 1640 has
        // 460 units of "blind walking" before the next reinforcement,
        // and any death in that zone (the pipe-cluster + Goombas at
        // ~1700-1900) wipes the gradient. Two thin intermediates fill
        // the gap.
        (1800, 175.0),  // past pipe-cluster + Goomba lane
        (1980, 300.0),  // approach to obstacle complex
        (2100, 400.0),  // past obstacle complex
        (2700, 600.0),  // at staircase base
        // 2026-05-09 — bridge the 2700→2900 staircase dead zone.
        // Same mechanism that broke the 1640→2100 wall: with no
        // dense signal across the multi-jump staircase, PPO has
        // nothing to gradient-descent on. Empirical evidence: 142-
        // gen run pinned at depth=2753 (one tile past 2700) for
        // 127 consecutive generations. A genome that crosses 2700
        // gets +600 once, then 199 units of reward silence until
        // 2900 — and the boosted +2000 at the top is unreachable
        // without solving the staircase jump pattern PPO can't
        // signal-find. +800 at 2820 (≈halfway up the staircase)
        // turns "make ANY progress on the staircase" into a
        // gradient-detectable event.
        (2820, 800.0),
        // 2026-05-08 — boost top-of-staircase from 1000 -> 2000.
        // Run gen 117-465 plateau'd at depth 3161 with only ~3
        // clears in last 50 gens. Doubling this checkpoint makes
        // "reach the staircase top" a much stronger attractor and
        // tightens the gradient on the multi-jump motor pattern
        // separating staircase-base (2700) from flagpole (3160).
        (2900, 2000.0),  // top of staircase, near flag
        // 2026-05-08 — new beyond-flag checkpoint. Mario at x>=3000
        // has cleared the flagpole (flag is at x≈2960; agent slides
        // off pole onto castle area at x=3050+). Firing +1500 here
        // gives the policy a final terminal landmark that's distinct
        // from the +2000 staircase-top reward, so the policy gets
        // gradient on the last 200 X-units (the actual flag-touch
        // motor pattern) rather than only on reaching-the-staircase.
        (3000, 1500.0),
    ];
    // Note: a previous revision (commit 770d258) added 5 dense
    // intermediate checkpoints at 1850/2000/2200/2400/2600 — the
    // population stalled at depth=1275 long before the dense zone.
    // The current 2-checkpoint addition is narrower: only the
    // bridge between known-reachable depth (1956) and the next
    // existing milestone (2100). If the agent never reaches 1800
    // anyway, the new checkpoints don't change anything; if it
    // does, they should pull it past the obstacle complex.

    /// 1-2 UNDERGROUND MAIN (area byte 2). The long underground half of
    /// 1-2. Obstacles: jumping pipes (~x=400), green pipe gauntlet
    /// (~x=1300), final descent to warp zone OR continuation gauntlet
    /// (~x=2300). Calibrated from emulator playthrough — not as
    /// obstacle-dense as 1-1 but still has clear chokepoints worth
    /// signaling. Mapped to area byte 2 (see `checkpoints_for`).
    const LEVEL_1_2: &'static [(u32, f64)] = &[
        (400, 50.0),    // past first jumping-pipe section
        (900, 100.0),   // past mid-level brick clusters
        // 2026-06-25 — split the 900→1300 reward gap. Standard dead-zone
        // bridge (cf. 1-1's 1640→2100 / 2700→2900): a 400-unit reward-
        // silent span gets a thin intermediate so crossing it is
        // gradient-detectable. NOTE: an A/B warm-started directly in the
        // underground showed it is learnable WITHOUT this bridge — both
        // the dense and no-shaping arms reach x~980 trivially, so the
        // greedy 1-1 policy's x~980 stall was a replay artifact, not a
        // training wall. The real binding stall is later: both arms
        // plateaued at ~2435/2618, inside the 2300→2700 gap. This 1150
        // entry is harmless density; the data-motivated future bridge is
        // ~2500 (see follow-up).
        (1150, 150.0),  // splits the 900→1300 gap (harmless density)
        (1300, 200.0),  // past green pipe gauntlet
        (1900, 350.0),  // past long-jump section
        (2300, 500.0),  // approach to warp zone / level end
        (2700, 1000.0), // near pipe to overworld / 1-3
    ];

    /// 1-3 (area byte 3). The above-ground floating-platform level. Ladder
    /// is APPROXIMATE — evenly-spaced with the dead-zone-bridge principle
    /// (no >450px reward-silent span) rather than per-obstacle calibrated,
    /// since no 1-3-clearing policy exists yet to map the real chokepoints.
    /// Refine from a clearing replay once available. Mapped to area byte 3.
    const LEVEL_1_3: &'static [(u32, f64)] = &[
        (250, 50.0),
        (600, 90.0),
        (1000, 140.0),
        (1400, 220.0),
        (1800, 340.0),
        // 2026-06-25 — bridge the 1800→2200 gap: the seed policy's
        // stochastic reach plateaued at x~1872 here (just past 1800),
        // so split the dead zone to make progress past it detectable.
        (2000, 420.0),
        (2200, 520.0),
        (2560, 900.0),  // approach to flagpole / level end
    ];

    /// 1-4 (area byte 4 — the castle: lava pits, firebars, the bridge/
    /// axe at the end). APPROXIMATE evenly-spaced ladder with the
    /// dead-zone-bridge principle; refine from a clearing replay. Mapped
    /// to area byte 4 (confirmed empirically: a 1-3-clearing policy
    /// transitions to displaylvl 3 / area byte 4 = 1-4).
    const LEVEL_1_4: &'static [(u32, f64)] = &[
        (300, 60.0),
        (700, 110.0),
        // 2026-06-26 — DENSE ramp through the x814→994 platform-hop.
        // Trajectory investigation: x814 is a jump-UP onto an elevated
        // platform-hop section (the furthest env hops platforms at
        // y~80-112; the collapsed policy stays at the bottom y~176,
        // oscillating against the step, never jumping). The 700→1100 gap
        // had NO dense reward, so the high-variance jump path competed
        // against the safe-stall with zero reinforcement → risk-aversion
        // collapse to x814. Dense, increasing checkpoints through the
        // platform-hop reward each successful hop so attempting the path
        // beats stalling.
        (840, 250.0),
        (920, 350.0),
        (1000, 450.0),
        (1100, 550.0),
        (1300, 700.0),
        (1500, 850.0),
        (1700, 1000.0),
        (1900, 1200.0),
        (2100, 1450.0),
        (2300, 1700.0),
        // 2026-06-26 — densify the final Bowser-bridge (2430→2560). The
        // agent reaches the bridge approach (~x2430) regularly but only
        // crosses into world 2 ~10% of iters; dense reward across the
        // bridge reinforces each step of the rare crossing to raise the
        // rate (toward consolidation).
        (2430, 1900.0),  // bridge approach (reached regularly)
        (2500, 2150.0),  // mid-bridge
        (2560, 2400.0),  // the axe / world-2 crossing
    ];

    /// Look up the dense-reward checkpoints for a given (world, level).
    /// Returns an empty slice for levels we haven't tuned, which means
    /// only the baseline forward_progress + completion_bonus fire on
    /// those levels. Not harmful — just less dense signal until the
    /// table is calibrated.
    fn checkpoints_for(world: u8, level: u8) -> &'static [(u32, f64)] {
        // NOTE: `level` here is the raw AREA byte ($0760), NOT the
        // displayed level number ($075C). A single displayed level can
        // span multiple area bytes. Verified empirically by replaying a
        // 1-1-clearing policy and logging ($075F, $075C, $0760):
        //   area 0 = 1-1 (full level, x up to ~3266)
        //   area 1 = 1-2 ENTRANCE (short above-ground bit, ~125 steps)
        //   area 2 = 1-2 UNDERGROUND MAIN (the long pipe-gauntlet half)
        //   area 3 = 1-3
        // The 1-2 obstacle ladder (LEVEL_1_2: pipes at 400, gauntlet at
        // 1300, warp at 2300) describes the UNDERGROUND, so it belongs on
        // area 2 — NOT area 1. It was previously mis-wired to area 1 (the
        // brief entrance), which left the underground main — where the
        // policy demonstrably stalls at x~980 — with ZERO dense signal,
        // and fired the underground bonuses on the wrong geometry.
        match (world, level) {
            (0, 0) => Self::LEVEL_1_1,
            // area 1 = 1-2 entrance: short and easy; no dense table
            // needed (forward + completion carry it).
            (0, 1) => &[],
            (0, 2) => Self::LEVEL_1_2,
            (0, 3) => Self::LEVEL_1_3,
            (0, 4) => Self::LEVEL_1_4,
            // world 2+ — not yet calibrated. Falls through to forward +
            // completion only.
            _ => &[],
        }
    }

    pub fn new(
        forward_weight: f64,
        score_weight: f64,
        death_penalty: f64,
        completion_bonus: f64,
        time_penalty: f64,
        checkpoint_scale: f64,
        air_bonus_weight: f64,
        pit_fall_extra: f64,
        enemy_death_extra: f64,
        survival_bonus: f64,
        survival_block_steps: u32,
        jump_clear_bonus: f64,
        jump_clear_min_dx: u32,
    ) -> Self {
        Self {
            forward_weight,
            score_weight,
            death_penalty,
            completion_bonus,
            time_penalty,
            checkpoint_scale,
            air_bonus_weight,
            pit_fall_extra,
            enemy_death_extra,
            survival_bonus,
            // Guard against zero (would div-by-zero on the modulo);
            // 1 means "fire every step" — degenerate but valid.
            survival_block_steps: survival_block_steps.max(1),
            step_count: 0,
            jump_clear_bonus,
            jump_clear_min_dx,
            prev_on_ground: true,
            air_start_x: 0,
            prev_x: 0,
            max_x_visited: 0,
            next_checkpoint: 0,
            prev_lives: 0,
            prev_score: 0,
            prev_world: 0,
            prev_level: 0,
            completed: false,
            died: false,
            first_step: true,
        }
    }
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.prev_on_ground = true;
        self.air_start_x = 0;
        self.prev_x = 0;
        self.max_x_visited = 0;
        self.next_checkpoint = 0;
        self.prev_lives = 0;
        self.prev_score = 0;
        self.prev_world = 0;
        self.prev_level = 0;
        self.completed = false;
        self.died = false;
        self.first_step = true;
    }

    fn read_x(ram: &[u8]) -> u32 {
        256 * ram[Self::RAM_X_PAGE] as u32 + ram[Self::RAM_X_LOW] as u32
    }
    fn read_score(ram: &[u8]) -> u64 {
        let mut score: u64 = 0;
        for d in &ram[0x07DE..0x07E4] {
            score = score * 10 + *d as u64;
        }
        score
    }

    /// Map the internal SMB area byte (`$0760`) to the displayed level
    /// label (e.g. "1-2"). `$0760` is an internal area *index*, not a
    /// 0-indexed level number: several area indices collapse onto a single
    /// displayed level and the numbering is non-contiguous, so the naive
    /// `area + 1` mislabels every level from the underground onward — the
    /// underground lives at area byte 2, which `+1` wrongly reports as
    /// "1-3". This mirrors the empirically-verified `AREA_TO_LEVEL` table
    /// the Python trainer uses so Rust and Python telemetry agree.
    fn level_label(world: u8, area: u8) -> String {
        let world_num = world as u16 + 1;
        match area {
            0 => format!("{}-1", world_num),
            1 => format!("{}-1->2", world_num),
            2 => format!("{}-2", world_num),
            5 => format!("{}-3", world_num),
            6 => format!("{}-4", world_num),
            other => format!("{}-area{}", world_num, other),
        }
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let x = Self::read_x(ram);
        let score = Self::read_score(ram);
        let lives = ram[Self::RAM_LIVES];
        let world = ram[Self::RAM_WORLD];
        let level = ram[Self::RAM_LEVEL];

        if self.first_step {
            self.prev_x = x;
            self.max_x_visited = x;
            self.prev_score = score;
            self.prev_lives = lives;
            self.prev_world = world;
            self.prev_level = level;
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        if world != self.prev_world || level != self.prev_level {
            let world_incremented = world > self.prev_world;
            self.prev_x = x;
            self.max_x_visited = x;
            // Reset checkpoint cursor on every level transition so the
            // bonuses re-fire on the new level. Without this, levels
            // 1-2+ would never produce checkpoint signal because all
            // were "consumed" during 1-1.
            self.next_checkpoint = 0;
            if world_incremented {
                // Castle-clear credit: a world-byte increment means Mario
                // beat the world's castle (x-4). SMB castles end with the
                // axe/Bowser walk, NOT a flagpole, so float_state never
                // reaches 3 and the flagpole predicate below can never
                // credit an x-4 clear. Fire the same completion bonus here
                // and KEEP the latch set so episode_success credits beating
                // the castle. Within-world level changes (x-1 -> x-2 ...)
                // are unaffected — the working 1-1/1-2/1-3 recipe never
                // increments the world byte. (F52.)
                if !self.completed {
                    acc.add("completion", self.completion_bonus);
                }
                self.completed = true;
            } else {
                // Re-arm the completion bonus for the new level so the
                // flag-touch at the end of 1-2, 1-3, ... each fires the
                // +completion reward. Without this, only the very first
                // level's clear would pay out. Paired with the dropping
                // of `done = true` on completion below — the agent now
                // plays continuously through multiple levels per episode.
                self.completed = false;
            }
            self.prev_world = world;
            self.prev_level = level;
        }
        // Reward NEW forward progress only — the delta between current x
        // and the furthest x reached this episode. Backtracking yields 0
        // (not negative), so the agent can back up to time a jump
        // without paying for the retreat. The reward magnitude is
        // unchanged when Mario only moves forward; the change kicks in
        // exactly when he reverses direction.
        let new_progress = (x as i64 - self.max_x_visited as i64).max(0);
        acc.add("forward", self.forward_weight * new_progress as f64);
        if x > self.max_x_visited {
            self.max_x_visited = x;
        }
        self.prev_x = x;

        // Air bonus: reward Mario for being airborne while moving right.
        // RAM[$001D] (float_state) is 0 when on ground, nonzero when in
        // air. RAM[$0057] is signed Mario X velocity. The gate ensures
        // we only pay for productive jumps (running through pipes,
        // clearing pits) — not stationary jumping in place.
        let on_ground = ram[Self::RAM_FLOAT_STATE] == 0;
        if self.air_bonus_weight != 0.0 {
            let vel_x = ram[0x0057] as i8 as i32;
            if !on_ground && vel_x > 0 {
                acc.add("air", self.air_bonus_weight);
            }
        }

        // Jump-clear bonus: fires once per successful airborne arc that
        // covered at least `jump_clear_min_dx` of forward progress.
        // Detect transitions: ground → air (record air_start_x), and
        // air → ground (compare current x to air_start_x). Approximates
        // "cleared an obstacle" without needing tile-level pit detection.
        if self.jump_clear_bonus != 0.0 {
            if self.prev_on_ground && !on_ground {
                // Just left the ground — record where the jump started.
                self.air_start_x = x;
            } else if !self.prev_on_ground && on_ground {
                // Just landed — credit if we covered enough horizontal
                // distance during the air period. saturating_sub guards
                // against backward-jump landings (rare but possible if
                // Mario hits a wall mid-air and lands further left).
                let dx = x.saturating_sub(self.air_start_x);
                if dx >= self.jump_clear_min_dx {
                    acc.add("jump_clear", self.jump_clear_bonus);
                }
            }
        }
        self.prev_on_ground = on_ground;

        // Periodic survival bonus. Increment first, then check the
        // boundary so step 150 (not step 0) gets the first credit.
        self.step_count = self.step_count.wrapping_add(1);
        if self.survival_bonus != 0.0
            && self.step_count.is_multiple_of(self.survival_block_steps)
        {
            acc.add("survival", self.survival_bonus);
        }

        // Dense progress checkpoints. Each crossing fires once per
        // episode and gives PPO a strong intermediate value signal at
        // the obstacle — much easier to learn from than a single
        // sparse "+2000 at the flag" reward at the trajectory tail.
        // Checkpoints are sorted by x so a single-pointer scan suffices.
        // Per-level tables: world/level lookup runs once per step
        // (constant time) so the cost is negligible vs the GPU/Pool work.
        if self.checkpoint_scale != 0.0 {
            let checkpoints = Self::checkpoints_for(world, level);
            while self.next_checkpoint < checkpoints.len() {
                let (threshold, bonus) = checkpoints[self.next_checkpoint];
                if x < threshold {
                    break;
                }
                acc.add("checkpoint", self.checkpoint_scale * bonus);
                self.next_checkpoint += 1;
            }
        }

        acc.add(
            "score",
            self.score_weight * (score as i64 - self.prev_score as i64) as f64,
        );
        self.prev_score = score;

        let player_state = ram[Self::RAM_PLAYER_STATE];
        if !self.died
            && (player_state == Self::PLAYER_STATE_DYING
                || player_state == Self::PLAYER_STATE_DEAD
                || lives < self.prev_lives)
        {
            acc.add("death", self.death_penalty);
            // Differentiate cause-of-death so PPO sees a sharper
            // gradient on each failure mode. Detection runs at the
            // moment of death state-transition (per-step compute is
            // edge-triggered by `!self.died`), so we read fresh RAM.
            //
            // Pit fall: Mario's screen-Y > 200 means he's well below
            // the normal jump arc — he was airborne and falling. The
            // floor of an SMB screen is ~Y=224; by the time the death
            // animation starts he's cleared visible level geometry.
            // Combined with !on_ground we know it wasn't an enemy
            // touching him on solid floor.
            if self.pit_fall_extra != 0.0 {
                let mario_y = ram[0x00CE] as u32;
                let on_ground = ram[0x001D] == 0;
                if mario_y > 200 && !on_ground {
                    acc.add("pit_fall", self.pit_fall_extra);
                }
            }
            // Enemy contact: scan the 5 enemy slots; if any active
            // enemy is within 1 tile (16 px) of Mario in both X and
            // Y, attribute the death to enemy contact. Computing
            // world-X for both Mario and enemy via the page+low
            // pair to handle screen wraps.
            if self.enemy_death_extra != 0.0 {
                let mario_x_world: i32 =
                    ((ram[0x006D] as i32) << 8) | (ram[0x0086] as i32);
                let mario_y_screen: i32 = ram[0x00CE] as i32;
                let mut enemy_nearby = false;
                for slot in 0..5 {
                    if ram[0x000F + slot] == 0 {
                        continue; // slot inactive
                    }
                    let ex_world: i32 = ((ram[0x006E + slot] as i32) << 8)
                        | (ram[0x0087 + slot] as i32);
                    let ey_screen: i32 = ram[0x00CF + slot] as i32;
                    let dx = (ex_world - mario_x_world).abs();
                    let dy = (ey_screen - mario_y_screen).abs();
                    if dx <= 16 && dy <= 16 {
                        enemy_nearby = true;
                        break;
                    }
                }
                if enemy_nearby {
                    acc.add("enemy_death", self.enemy_death_extra);
                }
            }
            self.died = true;
            done = true;
        }
        self.prev_lives = lives;

        let float_state = ram[Self::RAM_FLOAT_STATE];
        if !self.completed && float_state == 3 {
            acc.add("completion", self.completion_bonus);
            self.completed = true;
            // DON'T fire done — let the SMB cutscene play through and
            // the next level load naturally. Episode continues across
            // levels; level-change detection above re-arms `completed`
            // so the next level's flagpole touch fires its own bonus.
            // Episode only ends on actual Mario death (player_state
            // DYING/DEAD or lives drop, handled above).
        }

        acc.add("time_penalty", self.time_penalty);

        let level_id = Self::level_label(world, level);
        RewardOutput {
            reward: acc.total,
            done,
            level_id,
            breakdown_delta: acc.breakdown,
        }
    }

    pub fn episode_success(&self) -> bool {
        self.completed && !self.died
    }
}

// ============================================================
// Contra
// ============================================================

#[derive(Clone)]
pub struct ContraReward {
    forward_weight: f64,
    score_weight: f64,
    weapon_upgrade_bonus: f64,
    death_penalty: f64,
    time_penalty: f64,
    screen_progress_bonus: f64,
    completion_bonus: f64,
    // Stage-1-end screen number that counts as a clear. 255 = disabled
    // (must be measured from a telemetry run before it can fire).
    clear_screen: u8,
    prev_x: u8,
    prev_screen: u8,
    prev_score: u64,
    prev_weapon: u8,
    prev_lives: u8,
    max_screen: u8,
    completed: bool,
    died: bool,
    first_step: bool,
}

impl ContraReward {
    const RAM_PLAYER_X: usize = 0x0031;
    const RAM_SCREEN: usize = 0x0064;
    const RAM_PLAYER_STATE: usize = 0x002C;
    const RAM_LIVES: usize = 0x0032;
    const RAM_WEAPON: usize = 0x00AA;
    const SCORE_START: usize = 0x07E2;
    const SCORE_END: usize = 0x07E8;
    const DEAD_STATES: [u8; 2] = [0x01, 0x02];

    pub fn new(
        forward_weight: f64,
        score_weight: f64,
        weapon_upgrade_bonus: f64,
        death_penalty: f64,
        time_penalty: f64,
        screen_progress_bonus: f64,
        completion_bonus: f64,
        clear_screen: u8,
    ) -> Self {
        Self {
            forward_weight,
            score_weight,
            weapon_upgrade_bonus,
            death_penalty,
            time_penalty,
            screen_progress_bonus,
            completion_bonus,
            clear_screen,
            prev_x: 0,
            prev_screen: 0,
            prev_score: 0,
            prev_weapon: 0,
            prev_lives: 0,
            max_screen: 0,
            completed: false,
            died: false,
            first_step: true,
        }
    }
    pub fn reset(&mut self) {
        self.prev_x = 0;
        self.prev_screen = 0;
        self.prev_score = 0;
        self.prev_weapon = 0;
        self.prev_lives = 0;
        self.max_screen = 0;
        self.completed = false;
        self.died = false;
        self.first_step = true;
    }
    fn read_score(ram: &[u8]) -> u64 {
        let mut s: u64 = 0;
        for d in &ram[Self::SCORE_START..Self::SCORE_END] {
            s = s * 10 + *d as u64;
        }
        s
    }
    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let x = ram[Self::RAM_PLAYER_X];
        let screen = ram[Self::RAM_SCREEN];
        let score = Self::read_score(ram);
        let weapon = ram[Self::RAM_WEAPON];
        let lives = ram[Self::RAM_LIVES];
        let state = ram[Self::RAM_PLAYER_STATE];

        if self.first_step {
            self.prev_x = x;
            self.prev_screen = screen;
            self.prev_score = score;
            self.prev_weapon = weapon;
            self.prev_lives = lives;
            self.max_screen = screen;
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        let forward = ((screen as i32 - self.prev_screen as i32) * 256
            + (x as i32 - self.prev_x as i32))
            .max(0) as f64;
        acc.add("forward", self.forward_weight * forward);
        self.prev_x = x;
        self.prev_screen = screen;

        // Milestone bonus on each new max screen reached. Gated to the
        // high-water mark so it rewards genuine progress, not pacing
        // back and forth across a screen boundary. Also surfaces as
        // `reward_screen_progress` in metrics for progress telemetry.
        if screen > self.max_screen {
            acc.add(
                "screen_progress",
                self.screen_progress_bonus * (screen - self.max_screen) as f64,
            );
            self.max_screen = screen;
        }

        // Terminal win signal. `clear_screen == 255` (default) means
        // "disabled" — the real stage-1-end screen must be measured from
        // a telemetry run before this can fire. Drives `episode_success`
        // and the trainer's `completion`-keyed clears counter.
        if !self.completed && self.clear_screen != 255 && screen >= self.clear_screen {
            acc.add("completion", self.completion_bonus);
            self.completed = true;
        }

        let score_delta = (score as i64 - self.prev_score as i64).max(0) as f64;
        acc.add("score", self.score_weight * score_delta);
        self.prev_score = score;

        if weapon != self.prev_weapon && weapon != 0 {
            acc.add("weapon_upgrade", self.weapon_upgrade_bonus);
        }
        self.prev_weapon = weapon;

        if !self.died && (Self::DEAD_STATES.contains(&state) || lives < self.prev_lives) {
            acc.add("death", self.death_penalty);
            self.died = true;
            done = true;
        }
        self.prev_lives = lives;

        acc.add("time_penalty", self.time_penalty);

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("screen_{:02x}", screen),
            breakdown_delta: acc.breakdown,
        }
    }
    pub fn episode_success(&self) -> bool {
        self.completed && !self.died
    }
}

// ============================================================
// Mega Man
// ============================================================

#[derive(Clone)]
pub struct MegaManReward {
    forward_weight: f64,
    health_delta_weight: f64,
    boss_damage_weight: f64,
    boss_killed_bonus: f64,
    death_penalty: f64,
    time_penalty: f64,
    prev_x_page: u8,
    prev_health: u8,
    prev_boss: u8,
    prev_lives: u8,
    boss_killed: bool,
    died: bool,
    first_step: bool,
}

impl MegaManReward {
    const RAM_PLAYER_X_PAGE: usize = 0x0460;
    const RAM_PLAYER_HEALTH: usize = 0x06C0;
    const RAM_BOSS_HEALTH: usize = 0x06C1;
    const RAM_LIVES: usize = 0x00A8;
    pub fn new(
        forward_weight: f64,
        health_delta_weight: f64,
        boss_damage_weight: f64,
        boss_killed_bonus: f64,
        death_penalty: f64,
        time_penalty: f64,
    ) -> Self {
        Self {
            forward_weight,
            health_delta_weight,
            boss_damage_weight,
            boss_killed_bonus,
            death_penalty,
            time_penalty,
            prev_x_page: 0,
            prev_health: 28,
            prev_boss: 0,
            prev_lives: 0,
            boss_killed: false,
            died: false,
            first_step: true,
        }
    }
    pub fn reset(&mut self) {
        self.prev_x_page = 0;
        self.prev_health = 28;
        self.prev_boss = 0;
        self.prev_lives = 0;
        self.boss_killed = false;
        self.died = false;
        self.first_step = true;
    }
    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let x_page = ram[Self::RAM_PLAYER_X_PAGE];
        let health = ram[Self::RAM_PLAYER_HEALTH];
        let boss = ram[Self::RAM_BOSS_HEALTH];
        let lives = ram[Self::RAM_LIVES];
        if self.first_step {
            self.prev_x_page = x_page;
            self.prev_health = health;
            self.prev_boss = boss;
            self.prev_lives = lives;
            self.first_step = false;
        }
        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        acc.add(
            "forward",
            self.forward_weight
                * (x_page as i32 - self.prev_x_page as i32).max(0) as f64,
        );
        self.prev_x_page = x_page;

        if health < self.prev_health {
            acc.add(
                "health_delta",
                self.health_delta_weight * (self.prev_health - health) as f64,
            );
        }
        self.prev_health = health;

        if boss < self.prev_boss && self.prev_boss > 0 {
            acc.add(
                "boss_damage",
                self.boss_damage_weight * (self.prev_boss - boss) as f64,
            );
        }
        if boss == 0 && self.prev_boss > 0 {
            acc.add("boss_killed", self.boss_killed_bonus);
            self.boss_killed = true;
        }
        self.prev_boss = boss;

        if !self.died && (health == 0 || lives < self.prev_lives) {
            acc.add("death", self.death_penalty);
            self.died = true;
            done = true;
        }
        self.prev_lives = lives;

        acc.add("time_penalty", self.time_penalty);

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("page_{:02x}", x_page),
            breakdown_delta: acc.breakdown,
        }
    }
    pub fn episode_success(&self) -> bool {
        self.boss_killed && !self.died
    }
}

// ============================================================
// Castlevania
// ============================================================

#[derive(Clone)]
pub struct CastlevaniaReward {
    forward_weight: f64,
    health_delta_weight: f64,
    heart_weight: f64,
    boss_damage_weight: f64,
    boss_killed_bonus: f64,
    stage_cleared_bonus: f64,
    death_penalty: f64,
    time_penalty: f64,
    prev_x_screen: u8,
    prev_x: u8,
    /// Furthest combined position (screen * 256 + x) reached this
    /// stage. Same backtrack-penalty fix as MarioReward — without
    /// this, retreating to time a jump produces a negative forward
    /// reward and the agent learns "always press right out of fear
    /// of losing reward." Reset on stage transition + episode start.
    max_progress_visited: i32,
    prev_health: u8,
    prev_hearts: u8,
    prev_boss: u8,
    prev_stage: u8,
    prev_lives: u8,
    died: bool,
    first_step: bool,
    /// Latched true only on the real game-completion event — killing the
    /// final-block (Dracula) boss. episode_success reads this, so a
    /// mid-game block clear no longer counts as "beat Castlevania".
    completed: bool,
}

impl CastlevaniaReward {
    const RAM_PLAYER_HEALTH: usize = 0x0044;
    const RAM_STAGE: usize = 0x0028;
    const RAM_LIVES: usize = 0x002A;
    // Sub-weapon ammo (hearts). Was 0x0045, which is actually Simon's real HP
    // (emulator-verified: $0045 reads 64 = full health at the start, $0071
    // reads 5 = starting hearts). The old value made the "heart" bonus fire
    // on HP changes, not heart pickups.
    const RAM_HEARTS: usize = 0x0071;
    // Boss/Dracula health. Was 0x04A0, which is emulator-verified DEAD (reads
    // 0 constantly), so the final-block win could NEVER fire. $01A9 is the
    // boss-health slot (Data Crystal + tasvideos; reads 64 idle). Researched,
    // not yet Dracula-fight-verified — but strictly better than a dead byte,
    // and the stage>=18 gate below prevents any false early win.
    const RAM_BOSS_HEALTH: usize = 0x01A9;
    const RAM_X_SCREEN: usize = 0x003F;
    const RAM_X: usize = 0x0040;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        forward_weight: f64,
        health_delta_weight: f64,
        heart_weight: f64,
        boss_damage_weight: f64,
        boss_killed_bonus: f64,
        stage_cleared_bonus: f64,
        death_penalty: f64,
        time_penalty: f64,
    ) -> Self {
        Self {
            forward_weight,
            health_delta_weight,
            heart_weight,
            boss_damage_weight,
            boss_killed_bonus,
            stage_cleared_bonus,
            death_penalty,
            time_penalty,
            prev_x_screen: 0,
            prev_x: 0,
            max_progress_visited: 0,
            prev_health: 16,
            prev_hearts: 0,
            prev_boss: 0,
            prev_stage: 0,
            prev_lives: 0,
            died: false,
            first_step: true,
            completed: false,
        }
    }
    pub fn reset(&mut self) {
        self.prev_x_screen = 0;
        self.prev_x = 0;
        self.max_progress_visited = 0;
        self.prev_health = 16;
        self.prev_hearts = 0;
        self.prev_boss = 0;
        self.prev_stage = 0;
        self.prev_lives = 0;
        self.died = false;
        self.first_step = true;
        self.completed = false;
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let x_screen = ram[Self::RAM_X_SCREEN];
        let x = ram[Self::RAM_X];
        let health = ram[Self::RAM_PLAYER_HEALTH];
        let hearts = ram[Self::RAM_HEARTS];
        let boss = ram[Self::RAM_BOSS_HEALTH];
        let stage = ram[Self::RAM_STAGE];
        let lives = ram[Self::RAM_LIVES];

        let progress = x_screen as i32 * 256 + x as i32;
        if self.first_step {
            self.prev_x_screen = x_screen;
            self.prev_x = x;
            self.max_progress_visited = progress;
            self.prev_health = health;
            self.prev_hearts = hearts;
            self.prev_boss = boss;
            self.prev_stage = stage;
            self.prev_lives = lives;
            self.first_step = false;
        }
        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        if stage != self.prev_stage {
            self.prev_x_screen = x_screen;
            self.prev_x = x;
            self.max_progress_visited = progress;
        }
        // Reward only NEW forward progress beyond the furthest point
        // reached this stage. Backtracking yields 0 (not negative), so
        // the agent can retreat to time a jump or cycle a sub-weapon
        // without paying for it.
        let new_progress = (progress - self.max_progress_visited).max(0);
        acc.add("forward", self.forward_weight * new_progress as f64);
        if progress > self.max_progress_visited {
            self.max_progress_visited = progress;
        }
        self.prev_x_screen = x_screen;
        self.prev_x = x;

        if health < self.prev_health {
            acc.add(
                "health_delta",
                self.health_delta_weight * (self.prev_health - health) as f64,
            );
        }
        self.prev_health = health;

        if hearts > self.prev_hearts {
            acc.add(
                "heart",
                self.heart_weight * (hearts - self.prev_hearts) as f64,
            );
        }
        self.prev_hearts = hearts;

        if boss < self.prev_boss && self.prev_boss > 0 {
            acc.add(
                "boss_damage",
                self.boss_damage_weight * (self.prev_boss - boss) as f64,
            );
        }
        if boss == 0 && self.prev_boss > 0 {
            acc.add("boss_killed", self.boss_killed_bonus);
            // Dracula is the final boss, in the last block (stage 18).
            // Killing a boss on the final stage is the real
            // game-completion event — latch it and end the episode. A
            // boss-kill on any earlier block is only stage progress, NOT
            // a game win. If the $0028 stage encoding differs, this simply
            // never fires — strictly safer than the old predicate, which
            // reported "beat Castlevania" on merely clearing block 1.
            if stage >= 18 {
                self.completed = true;
                done = true;
            }
        }
        self.prev_boss = boss;

        if stage > self.prev_stage {
            acc.add("stage_cleared", self.stage_cleared_bonus);
        }
        self.prev_stage = stage;

        if !self.died && (health == 0 || lives < self.prev_lives) {
            acc.add("death", self.death_penalty);
            self.died = true;
            done = true;
        }
        self.prev_lives = lives;

        acc.add("time_penalty", self.time_penalty);

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("stage_{:02x}", stage),
            breakdown_delta: acc.breakdown,
        }
    }
    pub fn episode_success(&self) -> bool {
        // Beating Castlevania = defeating Dracula (final-block boss),
        // latched in compute() — not merely clearing a block or spawning
        // past block 0 under a warm-start.
        self.completed && !self.died
    }
}

// ============================================================
// Metroid
// ============================================================

#[derive(Clone)]
pub struct MetroidReward {
    exploration_bonus: f64,
    missile_weight: f64,
    energy_tank_bonus: f64, // was _energy_tank_bonus — now the filled-tank milestone
    _new_item_bonus: f64,   // item flags live in WRAM (unreachable) — see note below
    mb_hit_weight: f64,     // was _boss_damage_weight — per Mother-Brain hit
    win_bonus: f64,         // was _boss_killed_bonus — terminal (MB dead + escaped)
    health_delta_weight: f64,
    death_penalty: f64,
    time_penalty: f64,
    visited: HashSet<(u8, u8)>,
    prev_missiles: u8,
    prev_health_total: u32,
    prev_tanks: u8,
    prev_mb_hits: u8,
    mb_defeated: bool,
    won: bool,
    died: bool,
    first_step: bool,
}

impl MetroidReward {
    const RAM_HEALTH: usize = 0x0106;
    const RAM_HEALTH_TENS: usize = 0x0107;
    const RAM_MISSILES: usize = 0x0109;
    const RAM_ROOM: usize = 0x0050;
    const RAM_AREA: usize = 0x0074;
    // Win chain, using only get_ram-accessible bytes (item/equipment flags
    // live in cartridge WRAM $6000-$7FFF, which the 2 KB RAM snapshot cannot
    // reach — the old predicate read $006A, an unrelated zero-page byte that
    // is always noise). Verified reachable on this emulator; the fight/ending
    // VALUES are from the disassembly + Data Crystal, NOT yet fight-verified.
    const RAM_MB_STATE: usize = 0x0098; // Mother Brain state machine
    const RAM_MB_HITS: usize = 0x0099; // MB hit counter; dies at 32 hits
    const RAM_ENDING_MSG: usize = 0x007A; // "write ending message" flag
    const RAM_CREDITS: usize = 0x007B; // credits-rolling flag (post-escape)
    const MB_DEATH_HITS: u8 = 32;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        exploration_bonus: f64,
        missile_weight: f64,
        energy_tank_bonus: f64,
        new_item_bonus: f64,
        boss_damage_weight: f64,
        boss_killed_bonus: f64,
        health_delta_weight: f64,
        death_penalty: f64,
        time_penalty: f64,
    ) -> Self {
        Self {
            exploration_bonus,
            missile_weight,
            energy_tank_bonus,
            _new_item_bonus: new_item_bonus,
            mb_hit_weight: boss_damage_weight,
            win_bonus: boss_killed_bonus,
            health_delta_weight,
            death_penalty,
            time_penalty,
            visited: HashSet::new(),
            prev_missiles: 0,
            prev_health_total: 0,
            prev_tanks: 0,
            prev_mb_hits: 0,
            mb_defeated: false,
            won: false,
            died: false,
            first_step: true,
        }
    }
    pub fn reset(&mut self) {
        self.visited.clear();
        self.prev_missiles = 0;
        self.prev_health_total = 0;
        self.prev_tanks = 0;
        self.prev_mb_hits = 0;
        self.mb_defeated = false;
        self.won = false;
        self.died = false;
        self.first_step = true;
    }

    /// Filled energy-tank count = high nibble of the health hi-byte (BCD
    /// hundreds). Accessible without WRAM; a good progress milestone.
    fn filled_tanks(ram: &[u8]) -> u8 {
        ram[Self::RAM_HEALTH_TENS] >> 4
    }

    fn health_total(ram: &[u8]) -> u32 {
        let lo = ram[Self::RAM_HEALTH] as u32;
        let hi = ram[Self::RAM_HEALTH_TENS] as u32;
        100 * (hi >> 4) + 10 * (hi & 0xF) + (lo & 0xF)
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let area = ram[Self::RAM_AREA];
        let room = ram[Self::RAM_ROOM];
        let missiles = ram[Self::RAM_MISSILES];
        let tanks = Self::filled_tanks(ram);
        let mb_hits = ram[Self::RAM_MB_HITS];
        let health_total = Self::health_total(ram);

        if self.first_step {
            self.prev_missiles = missiles;
            self.prev_health_total = health_total;
            self.prev_tanks = tanks;
            self.prev_mb_hits = mb_hits;
            self.visited.insert((area, room));
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        let cell = (area, room);
        if !self.visited.contains(&cell) {
            self.visited.insert(cell);
            acc.add("exploration", self.exploration_bonus);
        }

        if missiles > self.prev_missiles {
            acc.add(
                "missile",
                self.missile_weight * (missiles - self.prev_missiles) as f64,
            );
        }
        self.prev_missiles = missiles;

        // Filled-energy-tank milestone (accessible proxy for major progress).
        if tanks > self.prev_tanks {
            acc.add("energy_tank", self.energy_tank_bonus * (tanks - self.prev_tanks) as f64);
        }
        self.prev_tanks = tanks;

        // Mother Brain fight: reward each new hit, and latch mb_defeated at 32
        // hits (she dies then) or once the state machine enters the death
        // chain (>=3, but NOT 8 = fight-init). prev_mb_hits guards the reset
        // between rooms where the counter clears.
        if mb_hits > self.prev_mb_hits {
            acc.add("mb_hit", self.mb_hit_weight * (mb_hits - self.prev_mb_hits) as f64);
        }
        self.prev_mb_hits = mb_hits;
        let mb_state = ram[Self::RAM_MB_STATE];
        if !self.mb_defeated
            && (mb_hits >= Self::MB_DEATH_HITS || (mb_state >= 3 && mb_state <= 7))
        {
            self.mb_defeated = true;
        }

        // Terminal WIN: the ending fires (credits rolling / ending message),
        // which only happens after escaping Zebes post-MB-death. Gate on
        // mb_defeated to make it airtight. Ends the episode.
        let ending = ram[Self::RAM_ENDING_MSG] == 1 || ram[Self::RAM_CREDITS] == 1;
        if !self.won && self.mb_defeated && ending {
            acc.add("win", self.win_bonus);
            self.won = true;
            done = true;
        }

        if health_total < self.prev_health_total {
            acc.add(
                "health_delta",
                self.health_delta_weight
                    * (self.prev_health_total - health_total) as f64,
            );
        }

        if health_total == 0 && self.prev_health_total > 0 {
            acc.add("death", self.death_penalty);
            self.died = true;
            done = true;
        }
        self.prev_health_total = health_total;

        acc.add("time_penalty", self.time_penalty);

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("area_{:02x}_room_{:02x}", area, room),
            breakdown_delta: acc.breakdown,
        }
    }
    pub fn episode_success(&self) -> bool {
        // WIN = Mother Brain defeated AND escaped (ending fired), latched in
        // `won`. The old predicate (item_flags.count_ones() >= 8) read $006A —
        // an unrelated zero-page byte (the real item flags are in unreachable
        // WRAM), so it fired on noise. This is the accessible, correct chain;
        // the fight/ending values are researched but not yet fight-verified.
        self.won && !self.died
    }
}

// ============================================================
// Factory — reads a game_profile-like weights dict.
// ============================================================

/// Convenience: weight lookup with default fallback.
// ============================================================
// Tetris (NES, Nintendo) — puzzle, axis-free
// ============================================================

#[derive(Clone)]
pub struct TetrisReward {
    line_clear_bonus: f64,
    score_weight: f64,
    top_out_penalty: f64,
    height_penalty: f64,
    hole_penalty: f64,
    survival_weight: f64,
    time_penalty: f64,
    line_goal: u32,
    board_shaping: f64,
    prev_lines: u32,
    prev_score: u64,
    start_level: u8,
    prev_level: u8,
    max_lines: u32,
    prev_agg_height: i32,
    prev_holes: i32,
    goal_reached: bool,
    topped_out: bool,
    first_step: bool,
}

impl TetrisReward {
    const RAM_SCORE: [usize; 3] = [0x0053, 0x0054, 0x0055];
    const RAM_LINES: [usize; 2] = [0x0050, 0x0051];
    const RAM_LEVEL: usize = 0x0044;
    const RAM_GAME_OVER: usize = 0x0058;
    const RAM_BOARD: usize = 0x0400;
    const BOARD_ROWS: usize = 20;
    const BOARD_COLS: usize = 10;
    /// Empty-cell sentinel on the $0400 playfield. Widely cited as 0xEF;
    /// VERIFY against a live RAM dump. Only used by the OPT-IN board
    /// shaping (board_shaping != 0), so a wrong value cannot break the
    /// core line/score/top-out signals.
    const EMPTY_CELL: u8 = 0xEF;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        line_clear_bonus: f64,
        score_weight: f64,
        top_out_penalty: f64,
        height_penalty: f64,
        hole_penalty: f64,
        survival_weight: f64,
        time_penalty: f64,
        line_goal: u32,
        board_shaping: f64,
    ) -> Self {
        Self {
            line_clear_bonus,
            score_weight,
            top_out_penalty,
            height_penalty,
            hole_penalty,
            survival_weight,
            time_penalty,
            line_goal,
            board_shaping,
            prev_lines: 0,
            prev_score: 0,
            start_level: 0,
            prev_level: 0,
            max_lines: 0,
            prev_agg_height: 0,
            prev_holes: 0,
            goal_reached: false,
            topped_out: false,
            first_step: true,
        }
    }

    pub fn reset(&mut self) {
        self.prev_lines = 0;
        self.prev_score = 0;
        self.start_level = 0;
        self.prev_level = 0;
        self.max_lines = 0;
        self.prev_agg_height = 0;
        self.prev_holes = 0;
        self.goal_reached = false;
        self.topped_out = false;
        self.first_step = true;
    }

    /// Little-endian packed BCD: addrs[0] is least significant, 2 BCD
    /// digits/byte. NES Tetris score/lines use this — do NOT reuse
    /// MarioReward's big-endian decode.
    fn read_bcd(ram: &[u8], addrs: &[usize]) -> u64 {
        let (mut val, mut mul) = (0u64, 1u64);
        for &a in addrs {
            let b = ram[a];
            val += ((b & 0x0F) as u64 + (b >> 4) as u64 * 10) * mul;
            mul *= 100;
        }
        val
    }

    /// Aggregate column height + buried holes (Dellacherie features).
    fn board_potential(ram: &[u8]) -> (i32, i32) {
        let (mut agg_h, mut holes) = (0i32, 0i32);
        for c in 0..Self::BOARD_COLS {
            let mut top: Option<usize> = None;
            for r in 0..Self::BOARD_ROWS {
                let filled = ram[Self::RAM_BOARD + r * Self::BOARD_COLS + c]
                    != Self::EMPTY_CELL;
                if filled && top.is_none() {
                    top = Some(r);
                }
                if !filled && top.is_some() {
                    holes += 1;
                }
            }
            if let Some(t) = top {
                agg_h += (Self::BOARD_ROWS - t) as i32;
            }
        }
        (agg_h, holes)
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let lines = Self::read_bcd(ram, &Self::RAM_LINES) as u32;
        let score = Self::read_bcd(ram, &Self::RAM_SCORE);
        let level = ram[Self::RAM_LEVEL];
        let game_over = ram[Self::RAM_GAME_OVER];

        if self.first_step {
            self.prev_lines = lines;
            self.prev_score = score;
            self.start_level = level;
            self.prev_level = level;
            self.max_lines = lines;
            if self.board_shaping != 0.0 {
                let (h, holes) = Self::board_potential(ram);
                self.prev_agg_height = h;
                self.prev_holes = holes;
            }
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        // Line clear — the dominant signal.
        let dl = lines.saturating_sub(self.prev_lines);
        if dl > 0 {
            acc.add("lines", self.line_clear_bonus * dl as f64);
            self.max_lines = self.max_lines.max(lines);
        }
        self.prev_lines = lines;

        // Score delta captures the multi-line (Tetris/40-100-300-1200 x
        // level) premium on top of the flat per-line bonus.
        acc.add(
            "score",
            self.score_weight * (score as i64 - self.prev_score as i64).max(0) as f64,
        );
        self.prev_score = score;

        // Board shaping (opt-in via board_shaping != 0): penalize INCREASES
        // only, so a line clear (which lowers the stack) never double-counts.
        if self.board_shaping != 0.0 {
            let (h, holes) = Self::board_potential(ram);
            let dh = (h - self.prev_agg_height).max(0) as f64;
            let dholes = (holes - self.prev_holes).max(0) as f64;
            acc.add("height", self.board_shaping * self.height_penalty * dh);
            acc.add("holes", self.board_shaping * self.hole_penalty * dholes);
            self.prev_agg_height = h;
            self.prev_holes = holes;
        }

        acc.add("survival", self.survival_weight);
        acc.add("time_penalty", self.time_penalty);

        // Success = reached the line goal OR leveled up (no "beat the game").
        if self.max_lines >= self.line_goal || level > self.start_level {
            self.goal_reached = true;
        }
        self.prev_level = level;

        // Top-out ends the episode (one-shot penalty).
        if !self.topped_out && game_over != 0 {
            acc.add("top_out", self.top_out_penalty);
            self.topped_out = true;
            done = true;
        }

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("level_{:02}", level),
            breakdown_delta: acc.breakdown,
        }
    }

    pub fn episode_success(&self) -> bool {
        self.goal_reached
    }
}

// ============================================================
// Bubble Bobble (NES) — single-screen clear-all-enemies arcade
// ============================================================

#[derive(Clone)]
pub struct BubbleBobbleReward {
    round_clear_bonus: f64,
    score_weight: f64,
    enemy_defeat_bonus: f64,
    death_penalty: f64,
    survival_weight: f64,
    time_penalty: f64,
    round_goal: u8,
    /// 0 = DISABLED. The NES Bubble Bobble RAM map documents no enemy-count
    /// byte, so per-enemy shaping is off until a verified address is set
    /// (the core loop keys off round-increment = "all enemies defeated").
    enemy_count_addr: usize,
    prev_round: u8,
    start_round: u8,
    prev_lives: u8,
    prev_score: u64,
    prev_enemy_count: u8,
    rounds_cleared: u32,
    died: bool,
    first_step: bool,
}

impl BubbleBobbleReward {
    const RAM_ROUND: usize = 0x0401;
    const RAM_LIVES: usize = 0x002E;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        round_clear_bonus: f64,
        score_weight: f64,
        enemy_defeat_bonus: f64,
        death_penalty: f64,
        survival_weight: f64,
        time_penalty: f64,
        round_goal: u8,
        enemy_count_addr: usize,
    ) -> Self {
        Self {
            round_clear_bonus,
            score_weight,
            enemy_defeat_bonus,
            death_penalty,
            survival_weight,
            time_penalty,
            round_goal,
            // Clamp to valid 2 KB RAM range: a misconfigured out-of-range
            // address is treated as disabled (0) rather than panicking on
            // `ram[addr]` in compute().
            enemy_count_addr: if enemy_count_addr < 2048 { enemy_count_addr } else { 0 },
            prev_round: 0,
            start_round: 0,
            prev_lives: 0,
            prev_score: 0,
            prev_enemy_count: 0,
            rounds_cleared: 0,
            died: false,
            first_step: true,
        }
    }

    pub fn reset(&mut self) {
        self.prev_round = 0;
        self.start_round = 0;
        self.prev_lives = 0;
        self.prev_score = 0;
        self.prev_enemy_count = 0;
        self.rounds_cleared = 0;
        self.died = false;
        self.first_step = true;
    }

    /// $0445 (millions) .. $044A (tens), digit-per-byte big-endian; returns
    /// the true score / 10 (units are always 0). Monotonic — fine as a delta.
    fn read_score(ram: &[u8]) -> u64 {
        let mut s = 0u64;
        for a in 0x0445..=0x044A {
            s = s * 10 + ram[a] as u64;
        }
        s
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let round = ram[Self::RAM_ROUND];
        let lives = ram[Self::RAM_LIVES];
        let score = Self::read_score(ram);

        if self.first_step {
            self.prev_round = round;
            self.start_round = round;
            self.prev_lives = lives;
            self.prev_score = score;
            if self.enemy_count_addr != 0 {
                self.prev_enemy_count = ram[self.enemy_count_addr];
            }
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        // Round clear = all enemies defeated ($0401 increments on screen clear).
        if round > self.prev_round {
            let adv = (round - self.prev_round) as f64;
            acc.add("round_clear", self.round_clear_bonus * adv);
            self.rounds_cleared += (round - self.prev_round) as u32;
        }
        self.prev_round = round;

        // Score delta — per-enemy-defeat + item proxy.
        acc.add(
            "score",
            self.score_weight * (score as i64 - self.prev_score as i64).max(0) as f64,
        );
        self.prev_score = score;

        // Optional enemy-defeat shaping (gated on a verified address).
        if self.enemy_count_addr != 0 {
            let c = ram[self.enemy_count_addr];
            if c < self.prev_enemy_count {
                acc.add(
                    "enemy_defeat",
                    self.enemy_defeat_bonus * (self.prev_enemy_count - c) as f64,
                );
            }
            self.prev_enemy_count = c;
        }

        acc.add("survival", self.survival_weight);
        acc.add("time_penalty", self.time_penalty);

        // First life loss ends the episode (repo single-life convention; no
        // dedicated game-over byte is documented for this ROM).
        if !self.died && lives < self.prev_lives {
            acc.add("death", self.death_penalty);
            self.died = true;
            done = true;
        }
        self.prev_lives = lives;

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("round_{:02}", round),
            breakdown_delta: acc.breakdown,
        }
    }

    pub fn episode_success(&self) -> bool {
        self.rounds_cleared >= self.round_goal as u32
    }
}

fn w(weights: &HashMap<String, f64>, key: &str, default: f64) -> f64 {
    weights.get(key).copied().unwrap_or(default)
}

pub fn build_reward(name: &str, weights: &HashMap<String, f64>) -> Option<Reward> {
    let name = name.to_lowercase();
    if name.contains("mario") {
        return Some(Reward::Mario(MarioReward::new(
            w(weights, "forward_progress", 1.0),
            w(weights, "score_delta", 0.025),
            w(weights, "death_penalty", -15.0),
            w(weights, "completion_bonus", 100.0),
            w(weights, "time_penalty", -0.01),
            // Dense progress checkpoints (default off for back-compat —
            // tile-mode profiles set 1.0 to enable).
            w(weights, "checkpoint_scale", 0.0),
            // Air bonus: per-step reward when airborne AND vel_x>0.
            // 0.0 (default) preserves existing behavior. tile-mode
            // profiles can opt in (~0.5-1.0) to give PPO direct
            // gradient on jump-while-running, addressing the
            // observed "stuck before staircase" failure mode.
            w(weights, "air_bonus", 0.0),
            // Death-cause differentiation (both default 0 = back-
            // compat). Negative values strengthen the death signal
            // for the specific failure mode; e.g. pit_fall_extra
            // -15 doubles the penalty for falling vs running into
            // an enemy, teaching PPO to commit to jumps earlier.
            w(weights, "pit_fall_extra", 0.0),
            w(weights, "enemy_death_extra", 0.0),
            // Periodic survival bonus — defaults disabled.
            w(weights, "survival_bonus", 0.0),
            w(weights, "survival_block_steps", 150.0) as u32,
            // Jump-clear bonus — defaults disabled. Fires when Mario
            // lands at least min_dx X-units past the takeoff point.
            w(weights, "jump_clear_bonus", 0.0),
            w(weights, "jump_clear_min_dx", 16.0) as u32,
        )));
    }
    if name.contains("zelda") {
        return Some(Reward::Zelda(ZeldaReward::new(
            w(weights, "exploration_bonus", 20.0),
            w(weights, "rupee_collected", 2.0),
            w(weights, "heart_container", 30.0),
            w(weights, "heart_recovery", 5.0),
            w(weights, "triforce_piece", 50.0),
            w(weights, "damage_taken", -5.0),
            w(weights, "death_penalty", -20.0),
            w(weights, "time_penalty", -0.001),
            w(weights, "motion", 0.0),
            w(weights, "stagnation", -0.05),
            w(weights, "stagnation_window", 30.0) as usize,
            w(weights, "wall_bump", -0.5),
            w(weights, "enemy_kill", 3.0),
            w(weights, "dungeon_enter", 15.0),
            w(weights, "new_item", 25.0),
            w(weights, "item_tier_upgrade", 15.0),
            w(weights, "first_sword", 100.0),
            w(weights, "key_collected", 10.0),
            w(weights, "magic_key", 30.0),
            w(weights, "map_compass", 8.0),
            w(weights, "item_used", 3.0),
            w(weights, "item_cycle", 0.5),
            w(weights, "activity_timeout_steps", 120.0) as usize,
            // Win-chain milestones. Defaults dwarf the exploration bonus
            // (128 screens x 50 = 6400) so reaching Ganon strictly dominates
            // map-wandering: all-fragments 3000, Level-9 entry 5000, Ganon
            // fight 5000, the win 20000. Tunable per profile.
            w(weights, "all_fragments", 3000.0),
            w(weights, "level9_enter", 5000.0),
            w(weights, "ganon_reached", 5000.0),
            w(weights, "win_bonus", 20000.0),
        )));
    }
    if name.contains("contra") {
        return Some(Reward::Contra(ContraReward::new(
            w(weights, "forward_progress", 0.5),
            w(weights, "score_delta", 0.01),
            w(weights, "weapon_upgrade", 5.0),
            w(weights, "death_penalty", -25.0),
            w(weights, "time_penalty", -0.005),
            w(weights, "screen_progress_bonus", 0.0),
            w(weights, "completion_bonus", 0.0),
            w(weights, "clear_screen", 255.0) as u8,
        )));
    }
    if name.contains("mega man") || name.contains("megaman") {
        return Some(Reward::MegaMan(MegaManReward::new(
            w(weights, "forward_progress", 0.1),
            w(weights, "health_delta", -2.0),
            w(weights, "boss_damage", 5.0),
            w(weights, "boss_killed", 75.0),
            w(weights, "death_penalty", -30.0),
            w(weights, "time_penalty", -0.002),
        )));
    }
    if name.contains("castlevania") {
        return Some(Reward::Castlevania(CastlevaniaReward::new(
            w(weights, "forward_progress", 0.5),
            w(weights, "health_delta", -1.5),
            w(weights, "heart_collected", 0.5),
            w(weights, "boss_damage", 3.0),
            w(weights, "boss_killed", 50.0),
            w(weights, "stage_cleared", 100.0),
            w(weights, "death_penalty", -25.0),
            w(weights, "time_penalty", -0.005),
        )));
    }
    if name.contains("metroid") {
        return Some(Reward::Metroid(MetroidReward::new(
            w(weights, "exploration_bonus", 8.0),
            w(weights, "missile_collected", 3.0),
            w(weights, "energy_tank", 200.0),   // filled-tank milestone (accessible)
            w(weights, "new_item", 30.0),       // unused: item flags are WRAM-only
            w(weights, "boss_damage", 20.0),    // per Mother-Brain hit
            w(weights, "boss_killed", 5000.0),  // TERMINAL win (MB dead + escaped)
            w(weights, "health_delta", -1.0),
            w(weights, "death_penalty", -25.0),
            w(weights, "time_penalty", -0.002),
        )));
    }
    if name.contains("tetris") {
        return Some(Reward::Tetris(TetrisReward::new(
            w(weights, "line_clear_bonus", 100.0),
            w(weights, "score_delta", 0.1),
            w(weights, "top_out_penalty", -50.0),
            w(weights, "height_penalty", -0.5),
            w(weights, "hole_penalty", -2.0),
            w(weights, "survival_weight", 0.01),
            w(weights, "time_penalty", -0.001),
            w(weights, "line_goal", 10.0) as u32,
            // Board shaping (aggregate height / holes) is opt-in like Mario's
            // checkpoint_scale — 0.0 disables it (and its unverified
            // EMPTY_CELL sentinel). Profiles set 1.0 to enable.
            w(weights, "board_shaping", 0.0),
        )));
    }
    if name.contains("bubble bobble") || name.contains("bubblebobble") {
        return Some(Reward::BubbleBobble(BubbleBobbleReward::new(
            w(weights, "round_clear_bonus", 200.0),
            w(weights, "score_delta", 0.05),
            w(weights, "enemy_defeat", 10.0),
            w(weights, "death_penalty", -25.0),
            w(weights, "survival_weight", 0.01),
            w(weights, "time_penalty", -0.002),
            w(weights, "round_goal", 1.0) as u8,
            // 0 = enemy-count shaping DISABLED until a verified address exists.
            w(weights, "enemy_count_addr", 0.0) as usize,
        )));
    }
    // Fallback for any ROM without a hand-authored reward. Uses generic,
    // axis-free progress signals (RAM-churn motion proxy, survival,
    // auto-detected monotonic score bytes, stuck detection) so a
    // brand-new game — Tetris, Bubble Bobble, anything in the library —
    // constructs a working reward instead of returning None (which the
    // Python layer turned into a hard ValueError at trainer build).
    Some(Reward::Generic(GenericReward::new(
        w(weights, "motion_weight", 0.01),
        w(weights, "survival_weight", 0.01),
        w(weights, "time_penalty", -0.001),
        w(weights, "score_weight", 0.1),
        w(weights, "stuck_steps", 240.0) as usize,
        w(weights, "warmup_steps", 60.0) as usize,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn zram() -> Vec<u8> {
        vec![0u8; 2048]
    }

    #[test]
    fn castlevania_win_only_on_final_block_boss() {
        let weights = HashMap::new();
        // A boss-kill on an EARLY block must NOT count as beating the game.
        let mut r = build_reward("castlevania", &weights).unwrap();
        let mut ram = zram();
        ram[0x0044] = 16; // player health (alive)
        ram[0x002A] = 3; // lives (alive)
        ram[0x0028] = 1; // stage = block 1
        ram[0x01A9] = 100; // boss health (real slot)
        r.compute(&ram, 0, false); // first step baselines prev_boss=100
        ram[0x01A9] = 0; // block-1 boss killed
        r.compute(&ram, 0, false);
        assert!(
            !r.episode_success(),
            "clearing block 1 must NOT count as beating Castlevania"
        );

        // A boss-kill on the FINAL block (Dracula, stage 18) IS the win.
        let mut r2 = build_reward("castlevania", &weights).unwrap();
        let mut ram2 = zram();
        ram2[0x0044] = 16;
        ram2[0x002A] = 3;
        ram2[0x0028] = 18; // final block
        ram2[0x01A9] = 100;
        r2.compute(&ram2, 0, false);
        ram2[0x01A9] = 0; // Dracula killed
        let out = r2.compute(&ram2, 0, false);
        assert!(
            r2.episode_success(),
            "defeating Dracula on the final block must be a win"
        );
        assert!(out.done, "the game-completion event ends the episode");
    }

    #[test]
    fn build_reward_falls_back_to_generic_for_unknown_game() {
        let weights = HashMap::new();
        // A game with no hand-authored reward must still construct a
        // working reward (the generic fallback), not None.
        assert!(
            matches!(build_reward("galaga", &weights), Some(Reward::Generic(_))),
            "unknown game must fall back to GenericReward"
        );
        assert!(
            matches!(build_reward("pac-man", &weights), Some(Reward::Generic(_))),
            "unknown game must fall back to GenericReward"
        );
        // Named games still dispatch to their specific reward.
        assert!(matches!(
            build_reward("super mario bros", &weights),
            Some(Reward::Mario(_))
        ));
        assert!(matches!(
            build_reward("contra", &weights),
            Some(Reward::Contra(_))
        ));
        // Tetris + Bubble Bobble now have bespoke axis-free rewards.
        assert!(matches!(
            build_reward("tetris", &weights),
            Some(Reward::Tetris(_))
        ));
        assert!(matches!(
            build_reward("bubble bobble", &weights),
            Some(Reward::BubbleBobble(_))
        ));
    }

    #[test]
    fn generic_reward_computes_without_panicking_on_arbitrary_ram() {
        let weights = HashMap::new();
        let mut r = build_reward("galaga", &weights).unwrap();
        let mut ram = zram();
        // A few steps with changing RAM must produce finite rewards.
        for i in 0..8u8 {
            ram[0x0040] = i.wrapping_mul(17);
            let out = r.compute(&ram, 0, true);
            assert!(out.reward.is_finite());
        }
    }

    #[test]
    fn tetris_line_clear_and_topout() {
        let mut weights = HashMap::new();
        weights.insert("board_shaping".to_string(), 0.0);
        weights.insert("score_delta".to_string(), 0.0);
        weights.insert("survival_weight".to_string(), 0.0);
        weights.insert("time_penalty".to_string(), 0.0);
        let mut r = build_reward("tetris", &weights).unwrap();
        let mut ram = zram();
        let _ = r.compute(&ram, 0, false); // first_step seed (lines=0)
        ram[0x0050] = 0x01; // BCD 1 line cleared
        let o1 = r.compute(&ram, 0, false);
        assert!((o1.reward - 100.0).abs() < 1e-6); // line_clear_bonus
        assert!(!r.episode_success());
        ram[0x0058] = 0x01; // top-out flag set
        let o2 = r.compute(&ram, 0, false);
        assert!(o2.done && (o2.reward + 50.0).abs() < 1e-6); // -50 once
        let o3 = r.compute(&ram, 0, false); // penalty must not re-fire
        assert!(o3.reward.abs() < 1e-6);
    }

    #[test]
    fn tetris_success_latches_at_line_goal() {
        let mut weights = HashMap::new();
        weights.insert("line_goal".to_string(), 3.0);
        let mut r = build_reward("tetris", &weights).unwrap();
        let mut ram = zram();
        let _ = r.compute(&ram, 0, false);
        ram[0x0050] = 0x03; // BCD 3 lines
        let _ = r.compute(&ram, 0, false);
        assert!(r.episode_success());
    }

    #[test]
    fn tetris_holes_counted_under_surface() {
        // Col 0: filled top cell, empty beneath = holes; EMPTY=0xEF.
        let mut ram = zram();
        for i in 0x0400..0x04C8 {
            ram[i] = 0xEF;
        }
        ram[0x0400] = 0x7B; // col 0, row 0 filled
        let (h, holes) = TetrisReward::board_potential(&ram);
        assert_eq!(holes, 19); // rows 1..19 empty under the top
        assert_eq!(h, 20); // column height = 20 - 0
    }

    #[test]
    fn bubble_bobble_round_clear_and_death() {
        let mut weights = HashMap::new();
        weights.insert("score_delta".to_string(), 0.0);
        weights.insert("survival_weight".to_string(), 0.0);
        weights.insert("time_penalty".to_string(), 0.0);
        let mut r = build_reward("bubble bobble", &weights).unwrap();
        let mut ram = zram();
        ram[0x0401] = 1; // round 1
        ram[0x002E] = 3; // 3 lives
        let _ = r.compute(&ram, 0, false); // seed
        assert!(!r.episode_success());
        ram[0x0401] = 2; // cleared round 1 -> round 2
        let o1 = r.compute(&ram, 0, false);
        assert!((o1.reward - 200.0).abs() < 1e-6);
        assert!(r.episode_success()); // rounds_cleared >= round_goal(1)
        ram[0x002E] = 2; // lost a life
        let o2 = r.compute(&ram, 0, false);
        assert!(o2.done && (o2.reward + 25.0).abs() < 1e-6);
        let o3 = r.compute(&ram, 0, false); // death penalty must not re-fire
        assert!(o3.reward.abs() < 1e-6);
    }

    #[test]
    fn bubble_bobble_optional_enemy_shaping() {
        let mut weights = HashMap::new();
        weights.insert("enemy_count_addr".to_string(), 0x0300 as f64);
        weights.insert("enemy_defeat".to_string(), 10.0);
        weights.insert("score_delta".to_string(), 0.0);
        weights.insert("survival_weight".to_string(), 0.0);
        weights.insert("time_penalty".to_string(), 0.0);
        let mut r = build_reward("bubble bobble", &weights).unwrap();
        let mut ram = zram();
        ram[0x0401] = 1;
        ram[0x002E] = 3;
        ram[0x0300] = 5; // 5 enemies
        let _ = r.compute(&ram, 0, false);
        ram[0x0300] = 3; // 2 defeated
        let o = r.compute(&ram, 0, false);
        assert!((o.reward - 20.0).abs() < 1e-6); // 2 * enemy_defeat
    }

    #[test]
    fn zelda_exploration_fires_on_new_cell() {
        let weights = HashMap::new();
        let mut r = build_reward("zelda", &weights).unwrap();
        let mut ram = zram();
        // First-step anchor.
        ram[0x00EB] = 7;
        ram[0x00EC] = 7;
        let o1 = r.compute(&ram, 0, false);
        // Move to new cell — should fire exploration bonus.
        ram[0x00EB] = 8;
        let o2 = r.compute(&ram, 0, false);
        assert!(o2.reward > o1.reward, "new cell should reward more");
    }

    #[test]
    fn metroid_win_requires_mb_defeat_and_escape_not_item_flags() {
        // Old bug: success = item_flags($006A).count_ones() >= 8 — read a
        // garbage zero-page byte. New: Mother Brain defeated + ending fired.
        let weights = HashMap::new();
        let mut r = build_reward("metroid", &weights).unwrap();
        let mut ram = zram();
        ram[0x0107] = 0x30; // health hi (3 tanks / 0 tens) — alive
        ram[0x0106] = 0x00;
        let _ = r.compute(&ram, 0, false); // anchor
        // Random noise in $006A must NOT win (the old bug).
        ram[0x006A] = 0xFF;
        let _ = r.compute(&ram, 0, false);
        assert!(!r.episode_success(), "item-flag noise must not be a win");
        // Mother Brain takes 32 hits -> defeated, but no escape yet.
        ram[0x0099] = 32;
        let _ = r.compute(&ram, 0, false);
        assert!(!r.episode_success(), "MB dead but not escaped is not a win yet");
        // Ending fires (credits) after escape -> win.
        ram[0x007B] = 1;
        let o = r.compute(&ram, 0, false);
        assert!(r.episode_success() && o.done, "MB defeated + ending = win");
    }

    #[test]
    fn metroid_ending_without_mb_defeat_does_not_win() {
        // Guard: the ending flags must be gated on MB actually being defeated
        // (a stray flag can't shortcut the win).
        let weights = HashMap::new();
        let mut r = build_reward("metroid", &weights).unwrap();
        let mut ram = zram();
        ram[0x0107] = 0x30;
        let _ = r.compute(&ram, 0, false);
        ram[0x007B] = 1; // ending flag but MB never defeated
        let _ = r.compute(&ram, 0, false);
        assert!(!r.episode_success(), "ending without MB defeat is not a win");
    }

    #[test]
    fn zelda_win_only_on_ganon_defeated_not_all_fragments() {
        // The bug this guards: collecting all 8 Triforce fragments only OPENS
        // Level 9 — it is NOT winning. Victory requires Ganon defeated ($0672).
        let weights = HashMap::new();
        let mut r = build_reward("zelda", &weights).unwrap();
        let mut ram = zram();
        ram[0x066F] = 0x33; // 3 max / 3 current hearts (alive)
        ram[0x0670] = 0xFF;
        let _ = r.compute(&ram, 0, false); // first-step anchor
        // All 8 fragments collected, but Ganon still alive.
        ram[0x0671] = 0xFF;
        let o1 = r.compute(&ram, 0, false);
        assert!(!r.episode_success(), "all fragments != win (Level 9 not beaten)");
        assert!(!o1.done, "all-fragments milestone must not end the episode");
        // Now Ganon defeated.
        ram[0x0672] = 1;
        let o2 = r.compute(&ram, 0, false);
        assert!(r.episode_success(), "Ganon defeated ($0672) is the win");
        assert!(o2.done, "the win must end the episode");
        assert!(o2.reward > 1000.0, "win jackpot should fire, got {}", o2.reward);
    }

    #[test]
    fn zelda_ending_song_also_counts_as_win() {
        let weights = HashMap::new();
        let mut r = build_reward("zelda", &weights).unwrap();
        let mut ram = zram();
        ram[0x066F] = 0x33;
        ram[0x0670] = 0xFF;
        let _ = r.compute(&ram, 0, false);
        ram[0x0609] = 0x10; // ending theme = Zelda rescued
        let o = r.compute(&ram, 0, false);
        assert!(r.episode_success() && o.done, "ending song is a win");
    }

    #[test]
    fn zelda_win_bonus_fires_once() {
        let weights = HashMap::new();
        let mut r = build_reward("zelda", &weights).unwrap();
        let mut ram = zram();
        ram[0x066F] = 0x33;
        ram[0x0670] = 0xFF;
        let _ = r.compute(&ram, 0, false);
        ram[0x0672] = 1;
        let o1 = r.compute(&ram, 0, false);
        let o2 = r.compute(&ram, 0, false); // win latched — no second jackpot
        assert!(o1.reward > o2.reward + 1000.0, "win bonus must not re-fire");
    }

    #[test]
    fn mario_death_once_only() {
        let weights = HashMap::new();
        let mut r = build_reward("mario", &weights).unwrap();
        let mut ram = zram();
        ram[0x000E] = 0x0B; // dying state
        ram[0x075A] = 1;
        let o1 = r.compute(&ram, 0, false);
        assert!(o1.done);
        let o2 = r.compute(&ram, 0, false);
        // second step should NOT re-apply -15 death penalty (guarded by died flag)
        assert!(o2.reward > o1.reward);
    }

    #[test]
    fn mario_forward_rewards_only_new_progress() {
        // Verify the visited_x_max gating: walking forward then
        // backward should reward only the forward portion, not punish
        // the backtrack. Confirms the agent can retreat to time a
        // jump without losing reward.
        let mut weights = HashMap::new();
        weights.insert("forward_progress".to_string(), 1.0);
        let mut r = build_reward("mario", &weights).unwrap();
        let mut ram = zram();

        // Initial position: x = 0x0040 (page 0, low 64).
        ram[0x006D] = 0; ram[0x0086] = 0x40;
        let _ = r.compute(&ram, 0, false);   // first_step seeds prev_x

        // Step right: x = 0x0050 → forward delta = 16, all new.
        // Reward includes a small time_penalty (~-0.01) so allow slop.
        ram[0x0086] = 0x50;
        let o1 = r.compute(&ram, 0, false);
        assert!((o1.reward - 16.0).abs() < 1.0,
                "expected forward ~+16 on new progress, got {}", o1.reward);

        // Step LEFT: x = 0x0048. max_x is still 0x50, so no new
        // progress → forward reward is 0 (NOT -8, the old behavior).
        ram[0x0086] = 0x48;
        let o2 = r.compute(&ram, 0, false);
        // time_penalty is the only non-zero contribution; forward = 0.
        // Default mario time_penalty is -0.01 (per impl); allow some slop.
        assert!(o2.reward.abs() < 1.0,
                "backtrack should yield ~0 forward reward, got {}", o2.reward);

        // Step right past previous max: x = 0x0058 → only the part
        // beyond max_x_visited (0x50) is new = 8.
        ram[0x0086] = 0x58;
        let o3 = r.compute(&ram, 0, false);
        assert!((o3.reward - 8.0).abs() < 1.0,
                "should reward only new progress beyond max (=8), got {}", o3.reward);
    }

    #[test]
    fn smb_checkpoint_tables_aligned_to_area_bytes() {
        // Regression guard for the area-byte misalignment fixed
        // 2026-06-25. `checkpoints_for` keys off the raw AREA byte
        // ($0760), verified by replaying a 1-1-clearing policy:
        //   area 0 = 1-1, area 1 = 1-2 ENTRANCE (short), area 2 = 1-2
        //   UNDERGROUND MAIN (the obstacle ladder's real home).
        // The underground ladder MUST live on area 2, not the entrance
        // (area 1) — the bug left the underground (where the policy
        // stalls at x~980) with zero dense signal.
        assert_eq!(MarioReward::checkpoints_for(0, 0), MarioReward::LEVEL_1_1);
        assert!(
            MarioReward::checkpoints_for(0, 1).is_empty(),
            "area byte 1 (1-2 entrance) must carry no dense table"
        );
        assert_eq!(
            MarioReward::checkpoints_for(0, 2),
            MarioReward::LEVEL_1_2,
            "area byte 2 (1-2 underground main) must carry the obstacle ladder"
        );
        assert!(!MarioReward::checkpoints_for(0, 2).is_empty());
    }

    #[test]
    fn smb_dense_checkpoint_fires_in_underground_not_entrance() {
        // End-to-end through compute(): the dense 1-2 ladder fires in the
        // underground main (area byte 2) and stays silent in the brief
        // entrance (area byte 1). x is packed ($006D<<8 | $0086); the
        // first checkpoint is x=400 (=0x190).
        let mut weights = HashMap::new();
        weights.insert("checkpoint_scale".to_string(), 1.0);
        weights.insert("forward_progress".to_string(), 0.0); // isolate checkpoint
        weights.insert("time_penalty".to_string(), 0.0);

        // Underground main (area byte 2): crossing x=400 fires +50.
        let mut r = build_reward("mario", &weights).unwrap();
        let mut ram = zram();
        ram[0x075F] = 0;
        ram[0x0760] = 2; // world 1, area byte 2 (underground main)
        ram[0x006D] = 0;
        ram[0x0086] = 0x10; // x=16, before first checkpoint
        let _ = r.compute(&ram, 0, false); // seed first_step
        ram[0x006D] = 1;
        ram[0x0086] = 0x90; // x=0x190=400 -> crosses checkpoint[0]
        let under = r.compute(&ram, 0, false);

        // Entrance (area byte 1): same crossing fires nothing.
        let mut r2 = build_reward("mario", &weights).unwrap();
        let mut ram2 = zram();
        ram2[0x075F] = 0;
        ram2[0x0760] = 1; // world 1, area byte 1 (entrance)
        ram2[0x006D] = 0;
        ram2[0x0086] = 0x10;
        let _ = r2.compute(&ram2, 0, false);
        ram2[0x006D] = 1;
        ram2[0x0086] = 0x90; // x=400
        let entrance = r2.compute(&ram2, 0, false);

        assert!(
            under.reward - entrance.reward >= 49.0,
            "underground (area 2) must fire the ~+50 dense checkpoint that \
             the entrance (area 1) does not: underground={} entrance={}",
            under.reward,
            entrance.reward
        );
    }

    #[test]
    fn smb_checkpoint_cursor_resets_on_area_transition() {
        // THE mechanism the area-byte fix depends on in real play: the
        // agent enters 1-2 at area byte 1 (empty table), then transitions
        // mid-episode to area byte 2 (the underground ladder). The
        // checkpoint cursor MUST reset on that 1->2 transition for the
        // LEVEL_1_2 ladder to ever fire. The other tests seed a single
        // area byte, so this transition-reset path was untested — and
        // deleting the reset clause would silently break the fix while
        // they still pass. This crosses the transition within ONE reward
        // instance and asserts the ladder fires only after it.
        let mut weights = HashMap::new();
        weights.insert("checkpoint_scale".to_string(), 1.0);
        weights.insert("forward_progress".to_string(), 0.0);
        weights.insert("time_penalty".to_string(), 0.0);
        let mut r = build_reward("mario", &weights).unwrap();
        let mut ram = zram();

        // Enter at area byte 1 (1-2 entrance), low x → seed first_step.
        ram[0x075F] = 0;
        ram[0x0760] = 1;
        ram[0x006D] = 0;
        ram[0x0086] = 0x10;
        let _ = r.compute(&ram, 0, false);
        // Cross x=400 in the entrance: empty table → nothing fires.
        ram[0x006D] = 1;
        ram[0x0086] = 0x90;
        let in_entrance = r.compute(&ram, 0, false);
        assert!(
            in_entrance.reward.abs() < 1e-6,
            "entrance (area 1) must fire no checkpoint, got {}",
            in_entrance.reward
        );
        // Transition to area byte 2 (underground), x resets low. This is
        // the world/level change that must reset the cursor.
        ram[0x0760] = 2;
        ram[0x006D] = 0;
        ram[0x0086] = 0x10;
        let _ = r.compute(&ram, 0, false);
        // Now cross x=400 in the underground: the ladder's first
        // checkpoint must fire (+50) — proving the cursor reset.
        ram[0x006D] = 1;
        ram[0x0086] = 0x90;
        let after_transition = r.compute(&ram, 0, false);
        assert!(
            (after_transition.reward - 50.0).abs() < 1e-6,
            "after the 1->2 area transition the underground ladder must \
             re-fire from x=400 (+50); cursor reset failed if this is 0. \
             got {}",
            after_transition.reward
        );
    }

    #[test]
    fn smb_checkpoint_tables_strictly_ascending() {
        // The compute() checkpoint loop is a single forward cursor that
        // breaks on the first threshold above x — correctness REQUIRES
        // each table be strictly ascending. Guards hand-inserted entries
        // (e.g. the 1150 bridge) against a future ordering typo.
        for (name, table) in [
            ("LEVEL_1_1", MarioReward::LEVEL_1_1),
            ("LEVEL_1_2", MarioReward::LEVEL_1_2),
            ("LEVEL_1_3", MarioReward::LEVEL_1_3),
            ("LEVEL_1_4", MarioReward::LEVEL_1_4),
        ] {
            for w in table.windows(2) {
                assert!(
                    w[0].0 < w[1].0,
                    "{name} checkpoint thresholds must be strictly \
                     ascending; found {} then {}",
                    w[0].0,
                    w[1].0
                );
            }
        }
        // World 1 levels 1-3 (area 3) and 1-4 (area 4) are calibrated;
        // world 2+ remains uncalibrated -> empty (forward + completion).
        assert_eq!(MarioReward::checkpoints_for(0, 3), MarioReward::LEVEL_1_3);
        assert_eq!(MarioReward::checkpoints_for(0, 4), MarioReward::LEVEL_1_4);
        assert!(MarioReward::checkpoints_for(1, 0).is_empty());
    }

    #[test]
    fn contra_progress_and_completion_signals() {
        // Isolate the new screen_progress + completion signals by zeroing
        // forward/score/time so only the milestone terms contribute.
        let mut weights = HashMap::new();
        weights.insert("forward_progress".to_string(), 0.0);
        weights.insert("score_delta".to_string(), 0.0);
        weights.insert("time_penalty".to_string(), 0.0);
        weights.insert("death_penalty".to_string(), 0.0);
        weights.insert("screen_progress_bonus".to_string(), 10.0);
        weights.insert("completion_bonus".to_string(), 100.0);
        weights.insert("clear_screen".to_string(), 5.0);
        let mut r = build_reward("contra", &weights).unwrap();
        let mut ram = zram();
        ram[0x0032] = 3; // lives (avoid spurious death)

        // First step seeds prev/max at screen 0.
        ram[0x0064] = 0;
        let _ = r.compute(&ram, 0, false);
        assert!(!r.episode_success());

        // Advance one screen → screen_progress fires (+10), no completion.
        ram[0x0064] = 1;
        let o1 = r.compute(&ram, 0, false);
        assert!((o1.reward - 10.0).abs() < 1e-6,
                "new max screen should reward +10, got {}", o1.reward);
        assert!(!r.episode_success());

        // Backtrack → no new max, zero progress reward (can't be farmed).
        ram[0x0064] = 0;
        let o2 = r.compute(&ram, 0, false);
        assert!(o2.reward.abs() < 1e-6, "backtrack should not reward, got {}", o2.reward);

        // Reach the clear screen → progress for the 1→5 jump (+40) plus
        // completion (+100); episode_success flips true.
        ram[0x0064] = 5;
        let o3 = r.compute(&ram, 0, false);
        assert!((o3.reward - 140.0).abs() < 1e-6,
                "clear should reward progress+completion = 140, got {}", o3.reward);
        assert!(r.episode_success(),
                "episode_success must be true after reaching clear_screen");

        // Completion fires once only.
        let o4 = r.compute(&ram, 0, false);
        assert!(o4.reward.abs() < 1e-6, "completion must not re-fire, got {}", o4.reward);
    }

    #[test]
    fn contra_completion_disabled_by_default() {
        // With no clear_screen weight set, the sentinel (255) keeps the
        // completion/win signal off so it can never falsely fire before
        // the real stage-end screen is measured.
        let weights = HashMap::new();
        let mut r = build_reward("contra", &weights).unwrap();
        let mut ram = zram();
        ram[0x0032] = 3;
        let _ = r.compute(&ram, 0, false);
        for s in 1u8..=254 {
            ram[0x0064] = s;
            let _ = r.compute(&ram, 0, false);
            assert!(!r.episode_success(),
                    "completion must stay disabled at screen {} when clear_screen unset", s);
        }
    }

    #[test]
    fn castlevania_forward_rewards_only_new_progress() {
        // Same backtrack-no-penalty contract as the Mario test:
        // walking forward then back then forward should reward only
        // the genuinely-new progress, not penalize the retreat.
        let mut weights = HashMap::new();
        weights.insert("forward_progress".to_string(), 1.0);
        let mut r = build_reward("castlevania", &weights).unwrap();
        let mut ram = zram();
        ram[0x0044] = 16;  // start at full health so death doesn't fire

        // Initial: x_screen=0, x=0x40 → progress = 64.
        ram[0x003F] = 0; ram[0x0040] = 0x40;
        let _ = r.compute(&ram, 0, false);

        // Walk right: x=0x60 → +32 new progress.
        ram[0x0040] = 0x60;
        let o1 = r.compute(&ram, 0, false);
        assert!((o1.reward - 32.0).abs() < 1.0,
                "expected forward ~+32, got {}", o1.reward);

        // Walk LEFT: x=0x50. max is still 0x60, so 0 forward
        // (NOT -16, the old buggy behavior).
        ram[0x0040] = 0x50;
        let o2 = r.compute(&ram, 0, false);
        assert!(o2.reward.abs() < 1.0,
                "backtrack should yield ~0 forward reward, got {}", o2.reward);

        // Walk past previous max: x=0x70 → +16 new progress
        // (only the part beyond the old max).
        ram[0x0040] = 0x70;
        let o3 = r.compute(&ram, 0, false);
        assert!((o3.reward - 16.0).abs() < 1.0,
                "should reward only NEW progress (=16), got {}", o3.reward);
    }

    #[test]
    fn mario_castle_clear_credits_completion_on_world_increment() {
        // F52: SMB castles (x-4) end with the axe/Bowser walk, not a
        // flagpole, so float_state never reaches 3. Beating 1-4 increments
        // the world byte ($075F); that transition must credit completion and
        // count as success (the flagpole predicate alone never could).
        let mut weights: HashMap<String, f64> = HashMap::new();
        weights.insert("completion_bonus".to_string(), 1000.0);
        weights.insert("time_penalty".to_string(), 0.0);
        let mut r = build_reward("mario", &weights).unwrap();
        let mut ram = zram();
        ram[0x075A] = 3; // lives (avoid the death path)
        ram[0x075F] = 0; // world byte 0 => world 1
        ram[0x0760] = 6; // area 6 => 1-4 castle
        let _ = r.compute(&ram, 0, false); // first step seeds prev_world=0
        assert!(!r.episode_success(), "castle not cleared yet");
        // Beat the castle: world byte increments to 1 (world 2), area 0 (2-1).
        ram[0x075F] = 1;
        ram[0x0760] = 0;
        let o = r.compute(&ram, 0, false);
        assert!(
            o.reward > 500.0,
            "castle clear must fire the completion bonus, got {}",
            o.reward
        );
        assert!(
            r.episode_success(),
            "beating a castle (world-byte increment) must count as success"
        );
    }

    #[test]
    fn mario_within_world_level_change_still_re_arms() {
        // Guard the F52 change: a within-world level change (1-1 -> 1-2, world
        // byte unchanged) must STILL re-arm completed=false so each flagpole
        // clear re-fires — the working 1-1/1-2/1-3 recipe must be untouched.
        let weights: HashMap<String, f64> = HashMap::new();
        let mut r = build_reward("mario", &weights).unwrap();
        let mut ram = zram();
        ram[0x075A] = 3;
        ram[0x075F] = 0; // world 1 throughout
        ram[0x0760] = 0; // 1-1
        let _ = r.compute(&ram, 0, false);
        ram[0x001D] = 3; // flagpole slide -> completion latches
        let _ = r.compute(&ram, 0, false);
        assert!(r.episode_success(), "flagpole clear latches success");
        // Advance to 1-2 (area 2), world byte unchanged -> must re-arm.
        ram[0x001D] = 0;
        ram[0x0760] = 2;
        let _ = r.compute(&ram, 0, false);
        assert!(
            !r.episode_success(),
            "within-world level change must re-arm completed (not latch)"
        );
    }

    #[test]
    fn mario_level_id_maps_area_byte_to_displayed_level() {
        // $0760 is an internal AREA index, not a 0-indexed level. The
        // underground level sits at area byte 2 and must read "1-2"; the
        // old `area + 1` reported it as "1-3", over-counting the
        // curriculum by one level from the underground onward.
        let weights: HashMap<String, f64> = HashMap::new();
        let label_for = |area: u8| -> String {
            let mut r = build_reward("mario", &weights).unwrap();
            let mut ram = zram();
            ram[0x075F] = 0; // world byte 0 -> "1-..."
            ram[0x0760] = area; // internal area index
            r.compute(&ram, 0, false).level_id
        };
        assert_eq!(label_for(0), "1-1");
        assert_eq!(label_for(1), "1-1->2");
        assert_eq!(label_for(2), "1-2");
        assert_eq!(label_for(5), "1-3");
        assert_eq!(label_for(6), "1-4");
    }

    #[test]
    fn zelda_half_heart_is_not_death() {
        // Full-heart nibble 0 with a nonzero partial-heart byte ($0670)
        // means Link still has a sliver of health — not a death. Only when
        // the partial byte also reaches 0 is Link actually dead.
        let weights = HashMap::new();
        let mut r = build_reward("zelda", &weights).unwrap();
        let mut ram = zram();
        // Anchor: 3 max containers, 1 full heart, no partial.
        ram[0x066F] = 0x31; // high nibble = max = 3, low nibble = full = 1
        ram[0x0670] = 0x00;
        let _ = r.compute(&ram, 0, false);

        // Lose the last full heart but keep half a heart -> still alive.
        ram[0x066F] = 0x30; // full = 0
        ram[0x0670] = 0x80; // half heart remaining
        let o1 = r.compute(&ram, 0, false);
        assert!(!o1.done, "half heart remaining must not be declared death");

        // Lose the final half heart -> now dead.
        ram[0x0670] = 0x00;
        let o2 = r.compute(&ram, 0, false);
        assert!(o2.done, "zero full and zero partial hearts must be death");
    }

    #[test]
    fn contra_weapon_upgrade() {
        let weights = HashMap::new();
        let mut r = build_reward("contra", &weights).unwrap();
        let mut ram = zram();
        let _ = r.compute(&ram, 0, false);
        ram[0x00AA] = 1;
        let o = r.compute(&ram, 0, false);
        assert!(o.reward >= 5.0 - 0.01);
    }
}
