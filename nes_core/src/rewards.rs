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
    PunchOut(PunchOutReward),
    KungFu(KungFuReward),
    Gradius(GradiusReward),
    Excitebike(ExcitebikeReward),
    Ghosts(GhostsReward),
    DuckTales(DuckTalesReward),
    KidIcarus(KidIcarusReward),
    DoubleDragon(DoubleDragonReward),
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
            Reward::PunchOut(r) => r.reset(),
            Reward::KungFu(r) => r.reset(),
            Reward::Gradius(r) => r.reset(),
            Reward::Excitebike(r) => r.reset(),
            Reward::Ghosts(r) => r.reset(),
            Reward::DuckTales(r) => r.reset(),
            Reward::KidIcarus(r) => r.reset(),
            Reward::DoubleDragon(r) => r.reset(),
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
            Reward::PunchOut(r) => r.compute(ram, action, want_breakdown),
            Reward::KungFu(r) => r.compute(ram, action, want_breakdown),
            Reward::Gradius(r) => r.compute(ram, action, want_breakdown),
            Reward::Excitebike(r) => r.compute(ram, action, want_breakdown),
            Reward::Ghosts(r) => r.compute(ram, action, want_breakdown),
            Reward::DuckTales(r) => r.compute(ram, action, want_breakdown),
            Reward::KidIcarus(r) => r.compute(ram, action, want_breakdown),
            Reward::DoubleDragon(r) => r.compute(ram, action, want_breakdown),
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
            Reward::PunchOut(r) => r.episode_success(),
            Reward::KungFu(r) => r.episode_success(),
            Reward::Gradius(r) => r.episode_success(),
            Reward::Excitebike(r) => r.episode_success(),
            Reward::Ghosts(r) => r.episode_success(),
            Reward::DuckTales(r) => r.episode_success(),
            Reward::KidIcarus(r) => r.episode_success(),
            Reward::DoubleDragon(r) => r.episode_success(),
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
            // A per-frame timer / animation counter increments on (almost)
            // every step (up ~= steps) — an idle agent then farms it as "score"
            // (measured: 96.8% of an idle episode's reward). A real score
            // display increments SPORADICALLY (on kills/pickups), so require
            // the byte to rise on well under half the observed steps. This
            // frequency guard excludes monotonic frame counters.
            let max_up = ((self.steps / 2).max(3)) as u32;
            for i in 0..limit {
                let up = self.byte_increased[i];
                let down = self.byte_decreased[i];
                if up >= 3 && up >= down.saturating_mul(4) && up <= max_up {
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
        // The generic fallback has NO game-specific notion of winning, so it
        // must NOT claim success. The old `total > 0.0` was trivially true
        // (the survival term 0.01 alone beats the time penalty 0.001), so it
        // reported "success" at step 1 for any game — poisoning the narrator,
        // the BC-success cache, and curriculum auto-promotion with non-wins.
        // A real win needs a hand-authored reward; until then, never succeed.
        false
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
        // `song` is a one-hot active-track byte, so test the ending track by
        // EQUALITY (not a bit-AND): the Level-9 ($20) and Ganon-fight ($02)
        // tracks are distinct values that must never satisfy the win.
        if !self.won && (ganon_defeated_byte != 0 || song == Self::SONG_ENDING) {
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
    prev_display_level: u8,
    completed: bool,
    won: bool,
    /// Durable per-episode "cleared at least one level" latch. Set on any
    /// flagpole touch OR castle clear; reset only on episode start. Unlike
    /// `completed` it is never re-armed on a level change, and unlike the old
    /// predicate it is not gated on `!died` — drives `episode_success` so a real
    /// clear followed by a later death still counts (see the note there).
    cleared_any: bool,
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
    /// Displayed level-within-world (0-3): x-1=0 .. x-4=3. Emulator-verified.
    /// The robust "was in an x-4 castle" signal — warps come from x-2 (=1),
    /// so a world increment with prev_display_level==3 is a real castle clear.
    const RAM_DISPLAY_LEVEL: usize = 0x075C;
    const DISPLAY_LEVEL_CASTLE: u8 = 3; // x-4
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

    /// World 2-1 dense ladder — calibrated from the 2026-07-16/17
    /// campaign's empirical hazard map (pit-trap ~1640, enemy pack
    /// ~1658, 4-tall piranha pipe ~1904, pit to ~1963, corridor to the
    /// staircase, flag ~3161). Without this table 2-1 trained on
    /// forward-progress alone; at gamma 0.9 the completion bonus is
    /// invisible >~200px out and the pool provably collapsed to a
    /// frozen death line at ~1891 after touching x=3300.
    const LEVEL_2_1: &'static [(u32, f64)] = &[
        (400, 50.0),    // past the opening enemies
        (800, 100.0),   // mid approach
        (1200, 150.0),  // approach to the runway
        (1421, 200.0),  // the runway line
        (1640, 275.0),  // pit-trap edge survived
        (1760, 350.0),  // past the enemy pack
        (1904, 425.0),  // at the piranha pipe
        (1990, 550.0),  // PAST the pipe+pit compound (the campaign wall)
        (2200, 600.0),  // corridor
        (2600, 675.0),  // corridor deep
        (2900, 750.0),  // staircase base
        (3100, 850.0),  // final approach
    ];

    /// Generic dense ladder for any level with no calibrated table:
    /// a checkpoint every 256px. Coarse, but it guarantees NO level
    /// ever again trains in a reward desert (the World-2 fallthrough
    /// oversight): with gamma 0.9, some forward signal must exist
    /// within every ~200px or distant progress cannot be credited.
    const GENERIC_LADDER: &'static [(u32, f64)] = &[
        (256, 40.0), (512, 80.0), (768, 120.0), (1024, 160.0),
        (1280, 200.0), (1536, 240.0), (1792, 280.0), (2048, 320.0),
        (2304, 360.0), (2560, 400.0), (2816, 440.0), (3072, 480.0),
        (3328, 520.0), (3584, 560.0), (3840, 600.0),
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
            // World 2-1: calibrated (see LEVEL_2_1). Every other
            // uncalibrated level gets the generic every-256px ladder —
            // never again a reward desert (the World-2 fallthrough that
            // cost 2-1 seventeen machine-hours).
            (1, 0) => Self::LEVEL_2_1,
            _ => Self::GENERIC_LADDER,
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
            prev_display_level: 0,
            completed: false,
            won: false,
            cleared_any: false,
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
        self.prev_display_level = 0;
        self.completed = false;
        self.won = false;
        self.cleared_any = false;
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
        let display_level = ram[Self::RAM_DISPLAY_LEVEL];

        if self.first_step {
            self.prev_x = x;
            self.max_x_visited = x;
            self.prev_score = score;
            self.prev_lives = lives;
            self.prev_world = world;
            self.prev_level = level;
            self.prev_display_level = display_level;
            // Arm the checkpoint cursor at the episode's TRUE starting
            // position: a mid-level restore (curriculum warm start,
            // archive return, backplay rung) must not lump-pay every
            // rung behind it on step 1. Rungs behind the start are
            // skipped, never paid — a level-entrance restore skips
            // nothing, so forward play from the start is unchanged.
            let checkpoints = Self::checkpoints_for(world, level);
            while self.next_checkpoint < checkpoints.len()
                && x >= checkpoints[self.next_checkpoint].0
            {
                self.next_checkpoint += 1;
            }
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
            // A LEGITIMATE castle clear = the world byte incremented AND Mario
            // was in an x-4 castle (display level $075C == 3). SMB castles end
            // with the axe/Bowser walk (no flagpole), so float_state never
            // reaches 3 and the flagpole predicate below can't credit it. But
            // a raw world-byte rise ALSO happens on warp-zone pipes (1-2's
            // warp jumps to 2-1/3-1/4-1), which must NOT count as beating a
            // castle — warps leave from x-2 ($075C==1), so gating on the
            // previous display level distinguishes them. (F52; the earlier
            // fix credited every world rise, wrongly rewarding warps.)
            if world_incremented && self.prev_display_level == Self::DISPLAY_LEVEL_CASTLE {
                if !self.won {
                    acc.add("completion", self.completion_bonus);
                }
                self.won = true; // durable success latch (survives the reset below)
                self.cleared_any = true; // durable "cleared a level" latch for curriculum/BC
            }
            // Always re-arm the flagpole-completion latch on ANY level change
            // so the next level's flag-touch fires its own bonus. Without this,
            // only the first level's clear would pay out. (Castle clears set
            // `won` above and still re-arm here, so the NEXT world's x-1
            // flagpole is not suppressed — the previous fix left it latched.)
            self.completed = false;
            self.prev_world = world;
            self.prev_level = level;
            self.prev_display_level = display_level;
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
            self.score_weight * (score as i64 - self.prev_score as i64).max(0) as f64,
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
            self.cleared_any = true; // durable clear latch — survives the per-level re-arm + a later death
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
        // Success for curriculum/BC = cleared AT LEAST ONE level this episode.
        // `cleared_any` is a durable latch set on every flagpole touch AND every
        // castle clear; unlike `completed` it is never re-armed on a level change
        // and, unlike the old predicate, it is NOT gated on `!died`. The normal
        // way a long SMB episode ends is a death in a LATER level, so the old
        // `(completed || won) && !died` scored every real clear-then-die as a
        // failure — `completed` was already re-armed to false by the level change
        // and the trailing death negated the result, so the curriculum could
        // never advance in tile mode even while agents cleared 1-1 into 1-2/1-3.
        self.cleared_any
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
    // Fine-scroll position within the current screen — pairs with
    // RAM_SCREEN exactly as this formula's `screen*256 + x` assumes:
    // verified live, $0065 wraps 255->0 on every $0064 increment. WAS
    // 0x0031 (confirmed live: constant 0 across 2000 frames of varied
    // real play — the same dead byte configs/contra.yaml's solve.progress
    // and ram_mapping.player_x were already independently corrected away
    // from; this struct's own copy was never updated). Before this fix,
    // `forward` reduced to `screen_delta*256` — every within-screen
    // pixel of movement (the vast majority of real gameplay) scored zero
    // forward reward; only full-screen-boundary crossings counted.
    const RAM_PLAYER_X: usize = 0x0065;
    const RAM_SCREEN: usize = 0x0064;
    // NOT the same "corrected" byte configs/contra.yaml's ram_mapping
    // uses (0x0090) — that byte was independently verified THIS SESSION
    // to free-run a 0->1->2 animation cycle even after lives hits 0,
    // making it unusable for death detection. Left at the historical
    // (also-nonfunctional but harmless) 0x002C: death is already caught
    // correctly below via the `lives < self.prev_lives` OR-clause alone.
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
        // Reaching the measured stage-end screen (`completed`) is a durable clear
        // latch that is never re-armed. It does NOT fire `done`, so the run can
        // continue past the clear and end on a later death — the old
        // `completed && !died` therefore mis-scored a genuine clear-then-die as a
        // failure (the same died-negation-after-durable-clear shape as the Mario
        // tile-mode curriculum bug). A real stage clear stays a success
        // regardless of any subsequent death.
        self.completed
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
        // NOT the game win. `boss_killed` latches on ANY single boss-HP meter
        // ($06C1) hitting 0 — one Robot Master, or even a stage mini-boss that
        // shares the meter — so it fired on the first kill, wildly overstating
        // "beat Mega Man" (which needs all 6 Masters + Wily). Return false
        // until a verified all-Masters ($009A mask) + Wily predicate exists;
        // `boss_killed` remains a shaping signal. The addresses ($06C0 health,
        // $06C1 boss, $00A8 lives) are the Mega Man 2 map, which is CORRECT:
        // DEFAULT_ROMS maps `megaman` -> Mega Man 2 (USA).nes. (An empirical
        // pass on Mega Man 1 found these dead — MM1 uses $006A/$00A6/$06C1 —
        // but the default pipeline is MM2; supporting MM1 would need a
        // ROM-hash-gated address map, not overwriting these.) There is no
        // reliable all-Robot-Masters flag reachable in RAM, so keeping this
        // false is correct — a false "win" poisons curriculum/eval; a missing
        // one only forgoes a bonus.
        false
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
            // Killing the boss on Dracula's stage is the game-completion
            // event; a boss-kill on any earlier stage is only progress. The
            // stage index $0028 is an internal stage counter (0x00-0x15 = 22
            // internal stages, NOT the 18 displayed blocks). Dracula's stage
            // is 0x12 (18) — proven by the game's own code: LoadStage indexes a
            // 22-entry StageDataTable and the ending is a special-case
            // `cmp #$12`. (An earlier pass used `>= 17` from a wrong
            // "0-indexed block 18 = 17" inference; that only caught Dracula by
            // luck. Not yet fight-verified live — no reachable Dracula state.)
            const DRACULA_STAGE: u8 = 0x12;
            if stage == DRACULA_STAGE {
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
    /// the true score / 10 (units are always 0). These are HUD TILE bytes, and
    /// a blank cell reads as a non-digit tile (0x27), NOT 0 — summed raw that
    /// makes a true score of 0 read ~3,999,990, and HUD tile transitions
    /// produce huge farmable deltas (measured +216k in one step). Clamp each
    /// byte to a real digit; treat any non-digit tile as 0. (Address/tile map
    /// unverified — if the real digit tiles are offset this reads 0, which is
    /// safe: no reward rather than garbage. Prefer score_delta small until
    /// verified against a live score.)
    fn read_score(ram: &[u8]) -> u64 {
        let mut s = 0u64;
        for a in 0x0445..=0x044A {
            let d = ram[a] as u64;
            s = s * 10 + if d <= 9 { d } else { 0 };
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

// ============================================================
// Punch-Out!! (NES, Nintendo) — boxing, single-bout KO/TKO
// ============================================================
//
// All addresses VERIFIED-LIVE on the Mike Tyson's Punch-Out!! (Japan, USA)
// (Rev A) dump (md5 c119a5a9…, mapper 9 / MMC2) from the captured Glass Joe
// opening-bell state, driving the emulator with real inputs:
//   $0398 opponent HP  — 0..0x60; drops on every LANDED Mac punch, REFILLS
//                        (increase) when a downed opponent gets back up.
//                        Mirror at $0399 tracks it 1:1.
//   $0392 Little Mac HP — 0..0x60; drops when Mac is hit. Mirror at $0391.
//   $03D1 opponent-down flag — rises 0->1 the frame the OPPONENT is knocked
//                        down (opp-specific: stayed 0 while Mac was down).
//   $03D0 Mac-down flag — rises 0->1 the frame LITTLE MAC is knocked down.
//   $0342 stars — banked star punches (earned by clean counters).
//   $0001 win latch — 0 all through the bout AND during a loss; goes nonzero
//                        ONLY at the winning KO/TKO (0 even on a NON-final
//                        knockdown), so it is a never-false-positive win flag.
//   $000A losses — career loss counter; increments the frame Mac is TKO'd.
// Damage is counted on HP DECREASES only, so a get-up refill (an increase)
// never scores and the between-knockdown bounce can't be farmed. The win and
// loss are keyed off dedicated latches, never off HP==0 (which occurs on
// every ordinary knockdown), so the milestone strictly dominates shaping.

#[derive(Clone)]
pub struct PunchOutReward {
    // Weights
    opp_damage: f64,
    knockdown_bonus: f64,
    star_bonus: f64,
    mac_damage: f64,
    mac_knockdown: f64,
    win_bonus: f64,
    loss_penalty: f64,
    time_penalty: f64,
    // State
    prev_opp_hp: i32,
    prev_mac_hp: i32,
    prev_opp_down: u8,
    prev_mac_down: u8,
    prev_stars: u8,
    start_matchid: u8,
    start_losses: u8,
    won: bool,
    lost: bool,
    first_step: bool,
}

impl PunchOutReward {
    const RAM_OPP_HP: usize = 0x0398; // opponent current health (0..0x60); mirror $0399
    const RAM_MAC_HP: usize = 0x0392; // Little Mac current health (0..0x60); mirror $0391
    const RAM_OPP_DOWN: usize = 0x03D1; // opponent knocked-down flag (0 up / !=0 down)
    const RAM_MAC_DOWN: usize = 0x03D0; // Little Mac knocked-down flag
    const RAM_STARS: usize = 0x0342; // banked star punches
    const RAM_MATCH_ID: usize = 0x0001; // win latch: 0 during bout, !=0 at the winning KO/TKO
    const RAM_LOSSES: usize = 0x000A; // career loss counter (increments on a Mac TKO)
    const RAM_OPP_ID: usize = 0x0002; // opponent index (Glass Joe = 0) — for level_id only

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        opp_damage: f64,
        knockdown_bonus: f64,
        star_bonus: f64,
        mac_damage: f64,
        mac_knockdown: f64,
        win_bonus: f64,
        loss_penalty: f64,
        time_penalty: f64,
    ) -> Self {
        Self {
            opp_damage,
            knockdown_bonus,
            star_bonus,
            mac_damage,
            mac_knockdown,
            win_bonus,
            loss_penalty,
            time_penalty,
            prev_opp_hp: 0,
            prev_mac_hp: 0,
            prev_opp_down: 0,
            prev_mac_down: 0,
            prev_stars: 0,
            start_matchid: 0,
            start_losses: 0,
            won: false,
            lost: false,
            first_step: true,
        }
    }

    pub fn reset(&mut self) {
        self.prev_opp_hp = 0;
        self.prev_mac_hp = 0;
        self.prev_opp_down = 0;
        self.prev_mac_down = 0;
        self.prev_stars = 0;
        self.start_matchid = 0;
        self.start_losses = 0;
        self.won = false;
        self.lost = false;
        self.first_step = true;
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let opp_hp = ram[Self::RAM_OPP_HP] as i32;
        let mac_hp = ram[Self::RAM_MAC_HP] as i32;
        let opp_down = ram[Self::RAM_OPP_DOWN];
        let mac_down = ram[Self::RAM_MAC_DOWN];
        let stars = ram[Self::RAM_STARS];
        let matchid = ram[Self::RAM_MATCH_ID];
        let losses = ram[Self::RAM_LOSSES];

        if self.first_step {
            self.prev_opp_hp = opp_hp;
            self.prev_mac_hp = mac_hp;
            self.prev_opp_down = opp_down;
            self.prev_mac_down = mac_down;
            self.prev_stars = stars;
            self.start_matchid = matchid;
            self.start_losses = losses;
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        // Opponent damage — the dense signal. Reward HP DECREASES only, so a
        // downed opponent's get-up refill (an increase) never scores.
        let opp_dmg = (self.prev_opp_hp - opp_hp).max(0) as f64;
        acc.add("opp_damage", self.opp_damage * opp_dmg);
        self.prev_opp_hp = opp_hp;

        // Knockdown dealt — rising edge of the opponent-down flag (fires once
        // per knockdown even though the flag stays set through the count).
        if self.prev_opp_down == 0 && opp_down != 0 {
            acc.add("knockdown", self.knockdown_bonus);
        }
        self.prev_opp_down = opp_down;

        // Star banked (clean counter) — reward increments.
        if stars > self.prev_stars {
            acc.add("star", self.star_bonus * (stars - self.prev_stars) as f64);
        }
        self.prev_stars = stars;

        // Damage taken — Mac HP decreases only (dodging avoids this penalty).
        let mac_dmg = (self.prev_mac_hp - mac_hp).max(0) as f64;
        acc.add("mac_damage", self.mac_damage * mac_dmg);
        self.prev_mac_hp = mac_hp;

        // Knockdown taken — rising edge of the Mac-down flag.
        if self.prev_mac_down == 0 && mac_down != 0 {
            acc.add("mac_knockdown", self.mac_knockdown);
        }
        self.prev_mac_down = mac_down;

        acc.add("time_penalty", self.time_penalty);

        // WIN — the match-id latch goes nonzero at the winning KO/TKO. It is 0
        // through the whole bout (including non-final knockdowns) and 0 during a
        // loss, so this can only mark a real match win. Latch + end the episode.
        if !self.won && matchid > self.start_matchid {
            acc.add("win", self.win_bonus);
            self.won = true;
            done = true;
        }

        // LOSE — the career loss counter increments when Mac is TKO'd.
        if !self.lost && losses > self.start_losses {
            acc.add("loss", self.loss_penalty);
            self.lost = true;
            done = true;
        }

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("opp_{:02}", ram[Self::RAM_OPP_ID]),
            breakdown_delta: acc.breakdown,
        }
    }

    pub fn episode_success(&self) -> bool {
        self.won
    }
}

// ============================================================
// Kung Fu (NES, 1985 — Nintendo's port of Irem's "Kung-Fu Master" /
// Famicom "Spartan X"). Side-scrolling beat-'em-up: Thomas ascends the
// Devil's Temple one floor at a time, defeating each floor's boss (which
// increments the floor byte $0058 and opens the door up) to rescue Sylvia
// on the 5th floor. RAM verified live on this ROM (see configs/kung-fu.yaml).
// ============================================================

#[derive(Clone)]
pub struct KungFuReward {
    forward_progress: f64,
    score_weight: f64,
    floor_clear_bonus: f64,
    game_clear_bonus: f64,
    /// Opt-in boss-HP shaping (default 0 = disabled). $04A5 is cross-sourced
    /// as the boss/enemy-energy byte but was never seen behaving as boss HP in
    /// live play (it idles at 0x30 and only hits 0 on the death reset), so it
    /// stays off until verified — mirrors Tetris board_shaping / BubbleBobble
    /// enemy_count_addr. When enabled it is guarded against the death-frame
    /// reset so 0x30->0 can never pay out as boss damage.
    boss_damage: f64,
    health_penalty: f64,
    death_penalty: f64,
    time_penalty: f64,
    survival_weight: f64,
    floor_goal: u8,
    game_clear_floor: u8,

    // Dynamic state.
    start_floor: u8,
    prev_floor: u8,
    floors_cleared: u32,
    max_world_x: u32,
    prev_score: u64,
    prev_health: u8,
    prev_lives: u8,
    prev_boss_hp: u8,
    died: bool,
    game_cleared: bool,
    first_step: bool,
}

impl KungFuReward {
    const RAM_FLOOR: usize = 0x0058;    // current floor: 0=1F .. 4=5F; +1 per floor clear
    const RAM_LIVES: usize = 0x005C;    // remaining lives (boots at 3)
    const RAM_HEALTH: usize = 0x04A6;   // player energy bar; drains + damage; underflows 0->255 at death
    const RAM_BOSS_HP: usize = 0x04A5;  // cross-sourced boss/enemy energy (opt-in shaping only)
    const RAM_PX_HI: usize = 0x0081;    // player world-X high byte
    const RAM_PX_LO: usize = 0x0090;    // player world-X low byte
    /// Score bytes: ONE DECIMAL DIGIT PER BYTE, big-endian ($0531 most
    /// significant). One Gripper KO = +100 pts = +1 on $0534 (hundreds);
    /// read_score() returns score/10 (the ones digit is always 0).
    const RAM_SCORE: [usize; 5] = [0x0531, 0x0532, 0x0533, 0x0534, 0x0535];
    /// The energy bar tops out near 0x30; a single-step loss can never exceed a
    /// full bar. Capping the counted drop (and excluding the 0xFF underflow
    /// sentinel) keeps the death/respawn reset from paying out as "damage".
    const HEALTH_DROP_CAP: u8 = 64;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        forward_progress: f64,
        score_weight: f64,
        floor_clear_bonus: f64,
        game_clear_bonus: f64,
        boss_damage: f64,
        health_penalty: f64,
        death_penalty: f64,
        time_penalty: f64,
        survival_weight: f64,
        floor_goal: u8,
        game_clear_floor: u8,
    ) -> Self {
        Self {
            forward_progress,
            score_weight,
            floor_clear_bonus,
            game_clear_bonus,
            boss_damage,
            health_penalty,
            death_penalty,
            time_penalty,
            survival_weight,
            floor_goal,
            game_clear_floor,
            start_floor: 0,
            prev_floor: 0,
            floors_cleared: 0,
            max_world_x: 0,
            prev_score: 0,
            prev_health: 0,
            prev_lives: 0,
            prev_boss_hp: 0,
            died: false,
            game_cleared: false,
            first_step: true,
        }
    }

    pub fn reset(&mut self) {
        self.start_floor = 0;
        self.prev_floor = 0;
        self.floors_cleared = 0;
        self.max_world_x = 0;
        self.prev_score = 0;
        self.prev_health = 0;
        self.prev_lives = 0;
        self.prev_boss_hp = 0;
        self.died = false;
        self.game_cleared = false;
        self.first_step = true;
    }

    fn world_x(ram: &[u8]) -> u32 {
        ram[Self::RAM_PX_HI] as u32 * 256 + ram[Self::RAM_PX_LO] as u32
    }

    /// Big-endian digit-per-byte decode of the 5 score bytes. Each byte holds
    /// one decimal digit (0-9); any non-digit (a glitch/uninitialised frame)
    /// reads as 0 so a corrupt frame yields no reward rather than garbage.
    /// Returns score/10 (the ones digit is always 0 and not stored), which is
    /// monotonic — a digit rollover never makes it decrease.
    fn read_score(ram: &[u8]) -> u64 {
        let mut s = 0u64;
        for &a in Self::RAM_SCORE.iter() {
            let d = ram[a] as u64;
            s = s * 10 + if d <= 9 { d } else { 0 };
        }
        s
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let floor = ram[Self::RAM_FLOOR];
        let lives = ram[Self::RAM_LIVES];
        let health = ram[Self::RAM_HEALTH];
        let boss_hp = ram[Self::RAM_BOSS_HP];
        let wx = Self::world_x(ram);
        let score = Self::read_score(ram);

        if self.first_step {
            self.start_floor = floor;
            self.prev_floor = floor;
            self.max_world_x = wx;
            self.prev_score = score;
            self.prev_health = health;
            self.prev_lives = lives;
            self.prev_boss_hp = boss_hp;
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        let floor_advanced = floor > self.prev_floor;

        // Forward progress: reward only a NEW maximum world-X, so pacing back
        // and forth can't farm it. Skipped on the floor-advance step (world-X
        // resets to the left edge of the next floor).
        if !floor_advanced && wx > self.max_world_x {
            acc.add("forward", self.forward_progress * (wx - self.max_world_x) as f64);
            self.max_world_x = wx;
        }

        // Score delta = enemies hit/defeated (Grippers, knife-throwers, ...).
        // Positive only; read_score is monotonic so a digit rollover never
        // pays negative.
        acc.add(
            "score",
            self.score_weight * (score as i64 - self.prev_score as i64).max(0) as f64,
        );
        self.prev_score = score;

        // Floor clear ($0058 increments) = that floor's boss defeated + door to
        // the next floor. The dominant milestone: floor_clear_bonus dwarfs a
        // whole floor's forward+score shaping, so clearing strictly beats
        // camping. Terminal game_clear when the top floor is reached (Sylvia).
        if floor_advanced {
            let adv = (floor - self.prev_floor) as u32;
            acc.add("floor_clear", self.floor_clear_bonus * adv as f64);
            self.floors_cleared += adv;
            self.max_world_x = 0; // earn progress from scratch on the new floor
            if !self.game_cleared && floor >= self.game_clear_floor {
                acc.add("game_clear", self.game_clear_bonus);
                self.game_cleared = true;
                done = true;
            }
        }
        self.prev_floor = floor;

        // Optional boss-HP shaping (opt-in; $04A5 unverified). Skipped on the
        // death step (lives just dropped) so the 0x30->0 reset can't be
        // mistaken for boss damage.
        if self.boss_damage != 0.0 && lives >= self.prev_lives && boss_hp < self.prev_boss_hp {
            acc.add(
                "boss_damage",
                self.boss_damage * (self.prev_boss_hp - boss_hp) as f64,
            );
        }
        self.prev_boss_hp = boss_hp;

        // Damage / drain penalty on the energy bar ($04A6), decreases only.
        // The bar underflows 0->255 at death (an INCREASE -> excluded by the
        // `<` test) and refills on a new life/floor; excluding the 0xFF
        // sentinel and capping the per-step drop keeps those resets from
        // paying out as spurious damage.
        if health < self.prev_health
            && self.prev_health != 0xFF
            && (self.prev_health - health) <= Self::HEALTH_DROP_CAP
        {
            acc.add(
                "health",
                self.health_penalty * (self.prev_health - health) as f64,
            );
        }
        self.prev_health = health;

        acc.add("survival", self.survival_weight);
        acc.add("time_penalty", self.time_penalty);

        // First life lost ends the episode (one-shot death penalty).
        if !self.died && lives < self.prev_lives {
            acc.add("death", self.death_penalty);
            self.died = true;
            done = true;
        }
        self.prev_lives = lives;

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("floor_{:02}", floor),
            breakdown_delta: acc.breakdown,
        }
    }

    /// A REAL win: cleared at least `floor_goal` floors (each clear = a floor
    /// boss defeated, i.e. $0058 incremented). Default goal 1 = clear the first
    /// floor; raise to 4 to require reaching the 5th floor, and pair with
    /// game_clear_floor for the full Sylvia rescue.
    pub fn episode_success(&self) -> bool {
        self.floors_cleared >= self.floor_goal as u32
    }
}

// ============================================================
// Gradius (NES) — horizontal auto-scrolling shoot-'em-up
// ============================================================

/// Reward for Gradius (Konami, 1986). The stage scrolls automatically, so the
/// ship's on-screen X is NOT stage progress — staying alive IS progress (the
/// world advances past the ship while it lives). The dominant dense signal is
/// therefore SURVIVAL, layered with the game's signature power-up mechanic
/// (collect capsules -> advance the meter -> spend for Speed/Missile/Double/
/// Laser/Option/Shield) and, on top, a stage-clear (Big-Core boss defeated)
/// milestone that strictly dominates everything else.
///
/// Empirically VERIFIED-LIVE on this emulator from the repo start-state
/// (`roms/Gradius (USA)_start.state.bin`):
///   * $00A4 player X — RIGHT raises it (0x50->0x77), LEFT lowers it (->0x29).
///   * $00A5 player Y — DOWN raises it (0x60->0x87), UP lowers it (->0x39).
///   * $0020 lives — starts 3, decremented to 2 on a collision death.
///   * $0100 status — 1 while alive, flips to 2 during the death/explosion
///     animation (leads the $0020 decrement by ~120 frames).
/// Power-up bytes ($0040 Speed / $0041 Missile / $0042 meter selector /
/// $0044 Weapon / $0045 Options / $0046 Shield) are the datacrystal RAM-map
/// values, cross-checked against fresh-ship base == 0; scripted play could not
/// grab a capsule to delta-verify them live, so they feed INCREASE-only shaping
/// (a wrong address is a no-op, never a farmable bonus).
///
/// Stage number: undocumented on datacrystal, and every stable-looking zero-page
/// candidate ($0004/$000A/$001E/$002A/$002B/$0037 == 1 at stage 1) proved to
/// change mid-stage under live probing, so NO byte is trustworthy as the stage
/// index yet. `stage_addr` is therefore DISABLED by default (0): the stage-clear
/// win stays conservatively unreachable — never a false positive — until the
/// real byte is read off a genuine stage-1->2 transition and set in the profile.
/// This mirrors Contra's `clear_screen == 255` sentinel.
#[derive(Clone)]
pub struct GradiusReward {
    survival_weight: f64,
    powerup_bonus: f64,
    capsule_bonus: f64,
    stage_clear_bonus: f64,
    death_penalty: f64,
    time_penalty: f64,
    /// Address of the current-stage byte. 0 == DISABLED (the never-false-
    /// positive default): with no verified stage byte, the stage-clear win
    /// cannot fire. Set to the real address (verified from a live stage
    /// transition) to enable Big-Core-defeated win detection.
    stage_addr: usize,
    prev_lives: u8,
    prev_power: i32,
    prev_sel: u8,
    max_stage: u8,
    stages_cleared: u32,
    died: bool,
    first_step: bool,
}

impl GradiusReward {
    const RAM_LIVES: usize = 0x0020;
    const RAM_STATUS: usize = 0x0100;
    const RAM_SPEED: usize = 0x0040;
    const RAM_MISSILE: usize = 0x0041;
    const RAM_SEL: usize = 0x0042;
    const RAM_WEAPON: usize = 0x0044;
    const RAM_OPTIONS: usize = 0x0045;
    const RAM_SHIELD: usize = 0x0046;
    /// $0100 == this value means the Vic Viper is dead/exploding.
    const DEAD_STATUS: u8 = 0x02;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        survival_weight: f64,
        powerup_bonus: f64,
        capsule_bonus: f64,
        stage_clear_bonus: f64,
        death_penalty: f64,
        time_penalty: f64,
        stage_addr: usize,
    ) -> Self {
        Self {
            survival_weight,
            powerup_bonus,
            capsule_bonus,
            stage_clear_bonus,
            death_penalty,
            time_penalty,
            // Clamp to valid 2 KB RAM range: an out-of-range address is treated
            // as DISABLED (0) rather than panicking on `ram[addr]` in compute().
            stage_addr: if stage_addr < 2048 { stage_addr } else { 0 },
            prev_lives: 0,
            prev_power: 0,
            prev_sel: 0,
            max_stage: 0,
            stages_cleared: 0,
            died: false,
            first_step: true,
        }
    }

    pub fn reset(&mut self) {
        self.prev_lives = 0;
        self.prev_power = 0;
        self.prev_sel = 0;
        self.max_stage = 0;
        self.stages_cleared = 0;
        self.died = false;
        self.first_step = true;
    }

    /// Composite "power level" = sum of the applied upgrades. Increases only
    /// when the ship actually gets stronger (spend a capsule); resets to 0 on
    /// death. Weapon is a mode (0=Normal,1=Laser,2=Double), so it contributes a
    /// flat +1 once armed rather than its raw value.
    fn power_level(ram: &[u8]) -> i32 {
        ram[Self::RAM_SPEED] as i32
            + ram[Self::RAM_MISSILE] as i32
            + ram[Self::RAM_OPTIONS] as i32
            + ram[Self::RAM_SHIELD] as i32
            + if ram[Self::RAM_WEAPON] != 0 { 1 } else { 0 }
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let lives = ram[Self::RAM_LIVES];
        let status = ram[Self::RAM_STATUS];
        let sel = ram[Self::RAM_SEL];
        let power = Self::power_level(ram);
        // stage_addr == 0 => stage detection DISABLED (win unreachable).
        let stage = if self.stage_addr != 0 {
            ram[self.stage_addr]
        } else {
            0
        };

        if self.first_step {
            self.prev_lives = lives;
            self.prev_sel = sel;
            self.prev_power = power;
            self.max_stage = stage;
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        // SURVIVAL — the dominant dense signal: every alive step is the stage
        // scrolling forward past the ship.
        acc.add("survival", self.survival_weight);
        acc.add("time_penalty", self.time_penalty);

        // Capsule pickup: the power-up meter selector ($0042) advances by 1 each
        // time a capsule is collected. Reward positive deltas only, so spending
        // the meter (reset to 0) and the death-reset never subtract.
        if sel > self.prev_sel {
            acc.add("capsule", self.capsule_bonus * (sel - self.prev_sel) as f64);
        }
        self.prev_sel = sel;

        // Power-up applied: composite Speed/Missile/Double-or-Laser/Options/
        // Shield level went up (a capsule was spent). Increase-only, so the
        // death-reset to 0 costs nothing (death is penalized once, below).
        let dpow = (power - self.prev_power).max(0) as f64;
        if dpow > 0.0 {
            acc.add("powerup", self.powerup_bonus * dpow);
        }
        self.prev_power = power;

        // STAGE CLEAR (THE win = Big-Core boss defeated -> next stage loads).
        // Fires only on a strict increase of a real stage byte, which cannot
        // happen mid-stage — so with a correctly-set `stage_addr` it is never a
        // false positive, and with the default (0) it simply never fires.
        if self.stage_addr != 0 && stage > self.max_stage {
            acc.add(
                "stage_clear",
                self.stage_clear_bonus * (stage - self.max_stage) as f64,
            );
            self.stages_cleared += (stage - self.max_stage) as u32;
            self.max_stage = stage;
        }

        // Death: the ship exploded ($0100 -> dead state) or a life was lost.
        // First death ends the episode (single-life convention, as Contra /
        // Bubble Bobble). Losing a life wipes all power-ups — captured here as
        // the episode simply ending.
        if !self.died && (status == Self::DEAD_STATUS || lives < self.prev_lives) {
            acc.add("death", self.death_penalty);
            self.died = true;
            done = true;
        }
        self.prev_lives = lives;

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("stage_{:02}", 1 + self.stages_cleared),
            breakdown_delta: acc.breakdown,
        }
    }

    pub fn episode_success(&self) -> bool {
        // A REAL win = at least one genuine stage transition (Big Core beaten).
        // Never `total > 0`. With `stage_addr` disabled this is always false.
        self.stages_cleared >= 1
    }
}

// ============================================================
// Excitebike (NES, Nintendo) — solo time-trial (Selection A / Design A).
// No enemies, no health, no time limit. The bike races a fixed track;
// the ONLY way the race ends is by crossing the FINISH LINE (verified
// live: idling forever never times out, the clock just wraps 0-59).
// Win = finish the track. Turbo (B) is faster but overheats the engine,
// which stalls the bike until it cools.
// ============================================================

#[derive(Clone)]
pub struct ExcitebikeReward {
    // Weights
    forward_progress: f64,
    section_bonus: f64,
    speed_bonus: f64,
    crash_penalty: f64,
    overheat_penalty: f64,
    completion_bonus: f64,
    time_penalty: f64,
    // State
    prev_section_dist: i32,
    prev_section: u8,
    start_section: u8,
    prev_bike_status: u8,
    prev_temp: u8,
    finished: bool,
    first_step: bool,
}

impl ExcitebikeReward {
    /// Section index 0..3. VERIFIED-LIVE: increments by exactly 1 at every
    /// track-section boundary (when RAM_SECTION_DIST resets 129->~2), climbs
    /// 0->1->2->3, then resets to 0 as the race restarts. Reaching the final
    /// section == the finish line has been crossed.
    const RAM_SECTION: usize = 0x03A4;
    /// Distance within the current section, 0..~129, RESETS to ~0 at each
    /// section boundary. VERIFIED-LIVE: advances only while moving forward
    /// (stays put under NOOP). Used for dense forward-motion shaping via its
    /// positive delta; the reset discontinuity is absorbed by `max(0, .)`.
    const RAM_SECTION_DIST: usize = 0x00ED;
    /// Bike speed 0..48 (A tops at 47, turbo B at 48). VERIFIED-LIVE.
    const RAM_SPEED: usize = 0x00F3;
    /// Engine temperature. VERIFIED-LIVE: starts at 8, climbs ~1/20f while
    /// holding turbo (B), cools fast otherwise; at 32 the engine OVERHEATS
    /// and speed collapses to ~1 until it cools. (datacrystal lists 0x03E3
    /// for this, but on the Japan/USA dump 0x03E3 stays 0 — 0x03B6 is the
    /// real gauge, confirmed by a sustained-turbo overheat/stall/cooldown run.)
    const RAM_TEMP: usize = 0x03B6;
    /// Bike status: 0 = racing normally, 2 = crash/tumble, 4 = remounting
    /// after an overheat stall. VERIFIED-LIVE. Crash penalty fires on the
    /// edge into 2 only (overheat is penalized separately, so the 4 state is
    /// not double-counted).
    const RAM_BIKE_STATUS: usize = 0x00F2;

    /// Final section index: crossing into it == the finish line. VERIFIED-LIVE
    /// as the terminal section (never observed as 4; the race resets after).
    const FINAL_SECTION: u8 = 3;
    /// Temperature at which the engine overheats and stalls. VERIFIED-LIVE.
    const OVERHEAT_THRESHOLD: u8 = 32;
    /// Bike-status value for a crash/tumble. VERIFIED-LIVE.
    const CRASH_STATUS: u8 = 2;
    /// Top speed, for normalizing the speed shaping term. VERIFIED-LIVE.
    const MAX_SPEED: f64 = 48.0;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        forward_progress: f64,
        section_bonus: f64,
        speed_bonus: f64,
        crash_penalty: f64,
        overheat_penalty: f64,
        completion_bonus: f64,
        time_penalty: f64,
    ) -> Self {
        Self {
            forward_progress,
            section_bonus,
            speed_bonus,
            crash_penalty,
            overheat_penalty,
            completion_bonus,
            time_penalty,
            prev_section_dist: 0,
            prev_section: 0,
            start_section: 0,
            prev_bike_status: 0,
            prev_temp: 0,
            finished: false,
            first_step: true,
        }
    }

    pub fn reset(&mut self) {
        self.prev_section_dist = 0;
        self.prev_section = 0;
        self.start_section = 0;
        self.prev_bike_status = 0;
        self.prev_temp = 0;
        self.finished = false;
        self.first_step = true;
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let section = ram[Self::RAM_SECTION];
        let section_dist = ram[Self::RAM_SECTION_DIST] as i32;
        let speed = ram[Self::RAM_SPEED];
        let temp = ram[Self::RAM_TEMP];
        let bike_status = ram[Self::RAM_BIKE_STATUS];

        if self.first_step {
            self.prev_section_dist = section_dist;
            self.prev_section = section;
            self.start_section = section;
            self.prev_bike_status = bike_status;
            self.prev_temp = temp;
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        // Dense forward motion. Only POSITIVE deltas count, so a section
        // boundary (section_dist 129 -> ~2) contributes 0 rather than a
        // spurious negative — the boundary is rewarded by section_bonus.
        let dd = (section_dist - self.prev_section_dist).max(0) as f64;
        acc.add("forward", self.forward_progress * dd);
        self.prev_section_dist = section_dist;

        // Section milestone — a chunk of the track completed.
        if section > self.prev_section {
            acc.add("section", self.section_bonus);
        }

        // Small throttle-holding shaping, normalized to <= speed_bonus/step.
        acc.add("speed", self.speed_bonus * (speed as f64 / Self::MAX_SPEED));

        // Crash: one-shot on the edge into a tumble (prev != 2 && now == 2).
        if self.prev_bike_status != Self::CRASH_STATUS && bike_status == Self::CRASH_STATUS {
            acc.add("crash", self.crash_penalty);
        }
        self.prev_bike_status = bike_status;

        // Overheat: one-shot on crossing the overheat threshold (which stalls
        // the bike). Recoverable (it cools when turbo is released), so it is a
        // penalty, not a terminal.
        if self.prev_temp < Self::OVERHEAT_THRESHOLD && temp >= Self::OVERHEAT_THRESHOLD {
            acc.add("overheat", self.overheat_penalty);
        }
        self.prev_temp = temp;

        acc.add("time_penalty", self.time_penalty);

        // Completion = crossing the finish line. VERIFIED-LIVE: the section
        // index reaching FINAL_SECTION fires the instant the bike crosses the
        // line (still at speed), the race then stops and resets. Guarded by
        // `section > start_section` so a corrupt start-state already at the
        // final section can never false-positive. Terminal + one-shot.
        if !self.finished && section >= Self::FINAL_SECTION && section > self.start_section {
            acc.add("completion", self.completion_bonus);
            self.finished = true;
            done = true;
        }
        self.prev_section = section;

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("section_{}", section),
            breakdown_delta: acc.breakdown,
        }
    }

    pub fn episode_success(&self) -> bool {
        self.finished
    }
}

// ============================================================
// Ghosts'n Goblins (NES, Capcom/Micronics) — brutal side-scroll
// action platformer. Arthur runs right through a stage, throwing
// weapons; 2-hit armor (armored -> underwear -> death), instant
// pits/water, a stage timer, and multiple lives. Win = clear a
// stage (the infamous must-loop-twice true ending is out of reach
// for a per-stage trainer, so a stage clear is the practical win).
// ============================================================

#[derive(Clone)]
pub struct GhostsReward {
    // Weights
    forward_weight: f64,
    kill_weight: f64,
    armor_loss_penalty: f64,
    death_penalty: f64,
    stage_clear_bonus: f64,
    survival_weight: f64,
    time_penalty: f64,
    // Win config
    clear_min_progress: i32,
    reset_threshold: i32,
    stage_addr: usize,
    score_addr: usize,
    // State
    prev_lives: u8,
    prev_armor: u8,
    prev_score: u8,
    start_stage: u8,
    max_stage: u8,
    max_world_x: i32,
    prev_world_x: i32,
    cleared: bool,
    died: bool,
    first_step: bool,
}

impl GhostsReward {
    // VERIFIED-LIVE on this ROM (emulator, from the shipped Stage-1 start
    // state). See notes in configs/ghosts-n-goblins.yaml.
    /// Player-1 lives. Reads 2 in the start state; decrements by 1 on each
    /// death (VERIFIED 2->1). Cross-sourced: the Ghosts'n Goblins Game Genie
    /// "infinite lives P1" code targets $0715 (P2 is $0716).
    const RAM_LIVES: usize = 0x0715;
    /// Game/player mode. 0x08 while in ACTIVE gameplay; 0x0A on the title,
    /// during the death sequence, and the attract demo's non-play frames
    /// (VERIFIED: 8 during play, flips to 0x0A on death). Used as a hard
    /// "we are really playing" gate so no clear can be reported off a menu,
    /// banner, or death frame.
    const RAM_MODE: usize = 0x0028;
    /// Armor flag: 1 = armored (Arthur in his knight suit, can survive one
    /// hit), 0 = underwear (next hit kills). VERIFIED: holds 1 until the
    /// exact frame Arthur's sprite turns from silver to red on his first
    /// hit, then holds 0 (matches the game's two-hit health model).
    const RAM_ARMOR: usize = 0x0585;
    /// Horizontal world / camera position, signed 16-bit little-endian.
    /// VERIFIED to respond to horizontal input (holding LEFT at spawn drives
    /// it to 0xFFF8 = -8; RIGHT increases it). Near 0 at the stage start.
    /// The exact per-stage scale is NOT verified (the opening is too hard to
    /// traverse under scripted play), so it drives dense shaping + the
    /// conservative Path-B win, both tunable, never a hard-coded milestone.
    const RAM_SCROLL_LO: usize = 0x005D;
    const RAM_SCROLL_HI: usize = 0x005E;
    /// $0028 value that means "active gameplay".
    const MODE_PLAY: u8 = 0x08;
    /// Plausibility clamp for the optional stage-byte win path: a real stage
    /// clear advances the index by 1 (sometimes 2 across an area split), never
    /// by a large jump — a large jump means the wrong byte was configured.
    const MAX_STAGE_JUMP: u8 = 2;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        forward_weight: f64,
        kill_weight: f64,
        armor_loss_penalty: f64,
        death_penalty: f64,
        stage_clear_bonus: f64,
        survival_weight: f64,
        time_penalty: f64,
        clear_min_progress: i32,
        reset_threshold: i32,
        stage_addr: usize,
        score_addr: usize,
    ) -> Self {
        Self {
            forward_weight,
            kill_weight,
            armor_loss_penalty,
            death_penalty,
            stage_clear_bonus,
            survival_weight,
            time_penalty,
            clear_min_progress,
            reset_threshold,
            // Clamp optional addresses to the 2 KB RAM range: a misconfigured
            // out-of-range address is treated as disabled (0), never a panic
            // on `ram[addr]` in compute() (mirrors BubbleBobble).
            stage_addr: if stage_addr < 2048 { stage_addr } else { 0 },
            score_addr: if score_addr < 2048 { score_addr } else { 0 },
            prev_lives: 0,
            prev_armor: 0,
            prev_score: 0,
            start_stage: 0,
            max_stage: 0,
            max_world_x: 0,
            prev_world_x: 0,
            cleared: false,
            died: false,
            first_step: true,
        }
    }

    pub fn reset(&mut self) {
        self.prev_lives = 0;
        self.prev_armor = 0;
        self.prev_score = 0;
        self.start_stage = 0;
        self.max_stage = 0;
        self.max_world_x = 0;
        self.prev_world_x = 0;
        self.cleared = false;
        self.died = false;
        self.first_step = true;
    }

    /// Signed 16-bit world/camera x from $005D (lo) / $005E (hi). Near spawn
    /// the value is a small negative (Arthur nudged against the left edge), so
    /// it is decoded as signed and callers clamp to >= 0 for progress.
    fn world_x(ram: &[u8]) -> i32 {
        let raw = ram[Self::RAM_SCROLL_LO] as i32 | ((ram[Self::RAM_SCROLL_HI] as i32) << 8);
        if raw >= 0x8000 { raw - 0x10000 } else { raw }
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let lives = ram[Self::RAM_LIVES];
        let armor = ram[Self::RAM_ARMOR];
        let mode = ram[Self::RAM_MODE];
        let wx = Self::world_x(ram).max(0);
        let stage = if self.stage_addr != 0 { ram[self.stage_addr] } else { 0 };
        let score = if self.score_addr != 0 { ram[self.score_addr] } else { 0 };

        if self.first_step {
            self.prev_lives = lives;
            self.prev_armor = armor;
            self.prev_score = score;
            self.start_stage = stage;
            self.max_stage = stage;
            self.max_world_x = wx;
            self.prev_world_x = wx;
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        // Death = first life lost (VERIFIED: $0715 decrements on death). Ends
        // the episode (single-life convention, like Contra/Castlevania). The
        // death frame contributes ONLY the death penalty and wipes the
        // progress tracker, so the respawn's scroll-reset can never be
        // mistaken for a stage clear (Path B below).
        if !self.died && lives < self.prev_lives {
            acc.add("death", self.death_penalty);
            self.died = true;
            done = true;
            self.max_world_x = 0;
        }
        self.prev_lives = lives;

        if !self.died {
            // Forward progress — backtrack-safe: reward only NEW furthest
            // world-x (retreating to time a jump costs nothing), mirroring
            // MarioReward / CastlevaniaReward's max_progress_visited.
            let dprog = (wx - self.max_world_x).max(0);
            acc.add("forward", self.forward_weight * dprog as f64);

            // Kills / score proxy — OPT-IN (score_addr != 0). No clean score
            // byte was isolated on this ROM (the HUD score lives in a tile
            // buffer, not a single BCD byte), so this is disabled by default;
            // when a verified byte is supplied, positive deltas only, so a
            // display wrap can never pay negative or farm.
            if self.score_addr != 0 {
                let ds = (score as i32 - self.prev_score as i32).max(0);
                acc.add("kills", self.kill_weight * ds as f64);
            }

            // Armor loss (armored 1 -> underwear 0): one-shot penalty per
            // transition (the 1->0 edge fires once; staying underwear does
            // not re-charge).
            if armor < self.prev_armor {
                acc.add("armor_loss", self.armor_loss_penalty);
            }

            acc.add("survival", self.survival_weight);
            acc.add("time_penalty", self.time_penalty);

            // Stage clear — the dominant terminal signal. Two conservative,
            // never-false-positive paths, both gated on active gameplay:
            if !self.cleared && mode == Self::MODE_PLAY {
                // Path A (opt-in via a VERIFIED stage byte, stage_addr != 0):
                // a strict, small increment of the stage index. Disabled by
                // default because no stage byte is verified on this ROM yet.
                let path_a = self.stage_addr != 0
                    && stage > self.start_stage
                    && stage <= self.start_stage.saturating_add(Self::MAX_STAGE_JUMP)
                    && stage > self.max_stage;
                // Path B (verified-address-only, default): the "area/stage
                // reload" signature — deep forward progress (>= clear_min_
                // progress) followed by world-x collapsing to ~0 (< reset_
                // threshold) while STILL alive and in active play. Traversing
                // a whole area then seeing the next one load at x~0 is a real
                // clear; a death cannot trigger it (death drops a life, wipes
                // max_world_x, and ends the episode above). The thresholds are
                // tunable — set clear_min_progress very high to disable Path B
                // if a stage byte (Path A) is verified and preferred.
                let path_b = self.max_world_x >= self.clear_min_progress
                    && wx < self.reset_threshold;
                if path_a || path_b {
                    acc.add("stage_clear", self.stage_clear_bonus);
                    self.cleared = true;
                    done = true;
                    if stage > self.max_stage {
                        self.max_stage = stage;
                    }
                }
            }

            if wx > self.max_world_x {
                self.max_world_x = wx;
            }
        }

        self.prev_armor = armor;
        self.prev_score = score;
        self.prev_world_x = wx;

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("stage_{:02x}_x{}", self.max_stage, self.max_world_x),
            breakdown_delta: acc.breakdown,
        }
    }

    pub fn episode_success(&self) -> bool {
        // A real stage clear (latched in compute), and not a death. Never
        // `total > 0` — a positive-reward episode that merely inched forward
        // or lost armor without clearing is NOT a win.
        self.cleared && !self.died
    }
}

#[derive(Clone)]
pub struct DuckTalesReward {
    forward_progress: f64,
    treasure_weight: f64,
    level_clear_bonus: f64,
    damage_penalty: f64,
    death_penalty: f64,
    survival_weight: f64,
    time_penalty: f64,
    /// Single-step money jump (in dollars) that identifies a boss's main
    /// treasure ($1,000,000) versus the largest gem (a $50,000 red diamond).
    win_treasure_value: u64,
    level_goal: u32,
    prev_x: u8,
    prev_money: u64,
    prev_hp: u8,
    levels_cleared: u32,
    cleared: bool,
    died: bool,
    first_step: bool,
}

impl DuckTalesReward {
    // Screen-relative X ($0008): RIGHT increments, LEFT decrements; wraps on
    // scroll. VERIFIED-LIVE. Weak local forward-progress nudge (non-linear game).
    const RAM_X: usize = 0x0008;
    // Hit points = the HUD hearts ($00DD): 3 at the Amazon start, rock-stable,
    // and equal to the on-screen heart count. VERIFIED-LIVE (value = hearts).
    const RAM_HP: usize = 0x00DD;
    // Current level index ($00F6): 0=Amazon 1=Transylvania 2=African Mines
    // 3=Himalayas 4=Moon. VERIFIED-LIVE (=0 in the Amazon; persists in play).
    const RAM_LEVEL: usize = 0x00F6;
    // Money HUD digit field: 7 tiles $0324..=$032A, most-significant first.
    // VERIFIED-LIVE ($0 -> $2000 on a gem, updated atomically in one frame).
    const RAM_MONEY: usize = 0x0324;
    const MONEY_LEN: usize = 7;
    // Reject screen-wrap / level-transition jumps in the X progress term.
    const MAX_FORWARD_DX: i32 = 32;
    // Cap the money-shaping delta at the largest single gem ($50,000 red
    // diamond) so a treasure pickup never swamps the level-clear bonus.
    const TREASURE_CAP: u64 = 50_000;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        forward_progress: f64,
        treasure_weight: f64,
        level_clear_bonus: f64,
        damage_penalty: f64,
        death_penalty: f64,
        survival_weight: f64,
        time_penalty: f64,
        win_treasure_value: u64,
        level_goal: u32,
    ) -> Self {
        Self {
            forward_progress,
            treasure_weight,
            level_clear_bonus,
            damage_penalty,
            death_penalty,
            survival_weight,
            time_penalty,
            win_treasure_value,
            level_goal,
            prev_x: 0,
            prev_money: 0,
            prev_hp: 0,
            levels_cleared: 0,
            cleared: false,
            died: false,
            first_step: true,
        }
    }

    pub fn reset(&mut self) {
        self.prev_x = 0;
        self.prev_money = 0;
        self.prev_hp = 0;
        self.levels_cleared = 0;
        self.cleared = false;
        self.died = false;
        self.first_step = true;
    }

    /// Money read from the 7-tile HUD digit field $0324..=$032A (verified
    /// live: $0 -> $2000 on a gem pickup, updated atomically in one frame).
    /// Digit tiles are the raw values 0-9; blank/leading tiles are 0x24 (36)
    /// and any non-digit is treated as 0. Max display is $9,999,999.
    fn read_money(ram: &[u8]) -> u64 {
        let mut v = 0u64;
        for a in Self::RAM_MONEY..(Self::RAM_MONEY + Self::MONEY_LEN) {
            let t = ram[a];
            v = v * 10 + if t <= 9 { t as u64 } else { 0 };
        }
        v
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let x = ram[Self::RAM_X];
        let hp = ram[Self::RAM_HP];
        let level = ram[Self::RAM_LEVEL];
        let money = Self::read_money(ram);

        if self.first_step {
            self.prev_x = x;
            self.prev_money = money;
            self.prev_hp = hp;
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        // Forward progress: reward only small rightward screen-X deltas.
        // $0008 is screen-relative and wraps on scroll, so a large jump
        // (wrap / level transition) is ignored; the game is non-linear so
        // this is a weak local nudge, not the main signal.
        let dx = x as i32 - self.prev_x as i32;
        if dx > 0 && dx <= Self::MAX_FORWARD_DX {
            acc.add("forward", self.forward_progress * dx as f64);
        }
        self.prev_x = x;

        // Treasure / money — positive deltas only, capped for shaping.
        let dm = money.saturating_sub(self.prev_money);
        if dm > 0 {
            let shaped = dm.min(Self::TREASURE_CAP) as f64;
            acc.add("treasure", self.treasure_weight * shaped);

            // Level clear = the boss's main treasure ($1,000,000) claimed in
            // a single pickup. The biggest gem is a $50,000 red diamond, so a
            // single-step jump this large can ONLY be a main treasure — i.e.
            // the level's boss was defeated. Per-step (not cumulative), so
            // gem-farming can never reach it. Latched once per episode.
            if !self.cleared && dm >= self.win_treasure_value {
                acc.add("level_clear", self.level_clear_bonus);
                self.levels_cleared += 1;
                self.cleared = true;
                done = true;
            }
        }
        self.prev_money = money;

        // Damage / death via HP ($00DD = the HUD hearts). A drop is damage;
        // reaching 0 from a positive value is death (single-life episode
        // convention). Skipped on the clear frame so a return-to-map screen
        // transition can never stack a spurious death onto a win.
        if !self.cleared {
            if !self.died && self.prev_hp > 0 && hp == 0 {
                acc.add("death", self.death_penalty);
                self.died = true;
                done = true;
            } else {
                let dhp = self.prev_hp as i32 - hp as i32;
                if dhp > 0 {
                    acc.add("damage", self.damage_penalty * dhp as f64);
                }
            }
        }
        self.prev_hp = hp;

        acc.add("survival", self.survival_weight);
        acc.add("time_penalty", self.time_penalty);

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("level_{:02}", level),
            breakdown_delta: acc.breakdown,
        }
    }

    pub fn episode_success(&self) -> bool {
        self.levels_cleared >= self.level_goal
    }
}

// ============================================================
// Kid Icarus (NES, Nintendo, 1986) — vertical-scroll platformer
// ============================================================

#[derive(Clone)]
pub struct KidIcarusReward {
    // Weights
    score_weight: f64,
    health_delta_weight: f64,
    stage_cleared_bonus: f64,
    boss_damage_weight: f64,
    boss_killed_bonus: f64,
    death_penalty: f64,
    time_penalty: f64,
    // Opt-in vertical-climb shaping (like Tetris `board_shaping` /
    // BubbleBobble `enemy_count_addr`). DISABLED by default: no verified
    // vertical-scroll accumulator exists on this ROM. When enabled it
    // rewards positive deltas of the byte at `altitude_addr` (must be a
    // monotonic climb counter where larger = higher). altitude_addr == 0
    // (default) turns it off and its unverified address can never break
    // the core score/health/stage/boss signals.
    altitude_weight: f64,
    altitude_addr: usize,
    // State
    prev_score: u64,
    prev_health: u8,
    prev_stage: u8,
    start_stage: u8,
    prev_boss: u8,
    prev_altitude: i32,
    cleared_any: bool,
    boss_killed: bool,
    died: bool,
    first_step: bool,
}

impl KidIcarusReward {
    // VERIFIED-LIVE on this ROM from the Stage-1 start state:
    //   $00A6 reads 7 at spawn and decrements 7->6->..->0 as Pit takes
    //   damage ($0728 invincibility jumps to 127 and counts down on each
    //   hit); it reaches 0 on death (fall off the bottom or HP depleted).
    const RAM_HEALTH: usize = 0x00A6;
    // Score: 3-byte LITTLE-ENDIAN BINARY (NOT BCD). VERIFIED-LIVE: a kill
    // advanced $0131 0->100->200, then it wrapped to 44 with a carry of 1
    // into $0132 (= 300 total). Increases on enemy kills / heart pickups.
    const RAM_SCORE: [usize; 3] = [0x0131, 0x0132, 0x0133];
    // Stage index for vertical-scrolling levels (Data Crystal). VERIFIED
    // STABLE: held 0 across 300k+ frames of play and did NOT change on
    // death, so a strict increase beyond the episode's start stage is an
    // unambiguous stage clear (never a false positive). The 0->1 increment
    // itself is cross-sourced, not reached under blind play.
    const RAM_STAGE: usize = 0x0130;
    // Fortress boss health (Data Crystal). Reads 0 at Stage-1 spawn, so the
    // >0 -> 0 kill event can never latch from the start state.
    const RAM_BOSS: usize = 0x006B;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        score_weight: f64,
        health_delta_weight: f64,
        stage_cleared_bonus: f64,
        boss_damage_weight: f64,
        boss_killed_bonus: f64,
        death_penalty: f64,
        time_penalty: f64,
        altitude_weight: f64,
        altitude_addr: usize,
    ) -> Self {
        Self {
            score_weight,
            health_delta_weight,
            stage_cleared_bonus,
            boss_damage_weight,
            boss_killed_bonus,
            death_penalty,
            time_penalty,
            altitude_weight,
            // Clamp to the 2 KB RAM range: an out-of-range address is treated
            // as disabled (0) rather than panicking on ram[addr] in compute().
            altitude_addr: if altitude_addr < 2048 { altitude_addr } else { 0 },
            prev_score: 0,
            prev_health: 7,
            prev_stage: 0,
            start_stage: 0,
            prev_boss: 0,
            prev_altitude: 0,
            cleared_any: false,
            boss_killed: false,
            died: false,
            first_step: true,
        }
    }

    pub fn reset(&mut self) {
        self.prev_score = 0;
        self.prev_health = 7;
        self.prev_stage = 0;
        self.start_stage = 0;
        self.prev_boss = 0;
        self.prev_altitude = 0;
        self.cleared_any = false;
        self.boss_killed = false;
        self.died = false;
        self.first_step = true;
    }

    /// Score is little-endian BINARY: addrs[0] least significant.
    fn read_score(ram: &[u8]) -> u64 {
        let mut val: u64 = 0;
        for (i, &a) in Self::RAM_SCORE.iter().enumerate() {
            val += (ram[a] as u64) << (8 * i as u64);
        }
        val
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let score = Self::read_score(ram);
        let health = ram[Self::RAM_HEALTH];
        let stage = ram[Self::RAM_STAGE];
        let boss = ram[Self::RAM_BOSS];
        let altitude = if self.altitude_addr != 0 {
            ram[self.altitude_addr] as i32
        } else {
            0
        };

        if self.first_step {
            self.prev_score = score;
            self.prev_health = health;
            self.prev_stage = stage;
            self.start_stage = stage;
            self.prev_boss = boss;
            self.prev_altitude = altitude;
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        // Combat / heart progress (kills climbing the tower). VERIFIED-LIVE.
        let score_delta = (score as i64 - self.prev_score as i64).max(0) as f64;
        acc.add("score", self.score_weight * score_delta);
        self.prev_score = score;

        // Opt-in vertical-climb shaping (disabled unless altitude_addr set).
        if self.altitude_addr != 0 {
            let dz = (altitude - self.prev_altitude).max(0) as f64;
            acc.add("altitude", self.altitude_weight * dz);
            self.prev_altitude = altitude;
        }

        // Damage penalty (HP lost). VERIFIED-LIVE.
        if health < self.prev_health {
            acc.add(
                "health_delta",
                self.health_delta_weight * (self.prev_health - health) as f64,
            );
        }
        self.prev_health = health;

        // Stage clear — the dominant vertical-progress milestone. $0130 only
        // increases on a real stage advance (verified stable otherwise), so
        // each increment is a cleared sub-stage; the latch requires advancing
        // strictly beyond the episode's start stage (safe under warm-starts).
        if stage > self.prev_stage {
            acc.add(
                "stage_cleared",
                self.stage_cleared_bonus * (stage - self.prev_stage) as f64,
            );
            if stage > self.start_stage {
                self.cleared_any = true;
            }
        }
        self.prev_stage = stage;

        // Fortress boss. boss_damage is dense per HP removed; boss_killed
        // latches the fortress clear. Gated on prev_boss > 0 so a spawn-time
        // 0 (no boss loaded) can never count as a kill.
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

        // Death: HP depleted OR fell off the bottom of the vertical scroll —
        // both drive $00A6 to 0 (VERIFIED-LIVE for HP depletion). One-shot.
        if !self.died && health == 0 {
            acc.add("death", self.death_penalty);
            self.died = true;
            done = true;
        }

        acc.add("time_penalty", self.time_penalty);

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("stage_{:02x}", stage),
            breakdown_delta: acc.breakdown,
        }
    }

    pub fn episode_success(&self) -> bool {
        // A real Kid Icarus win = clearing a stage ($0130 advanced beyond the
        // episode's start stage) OR defeating a fortress boss ($006B >0 -> 0).
        // Both are gated so they can NEVER latch from the start state, and the
        // stage byte is verified stable (no drift, no change on death), so
        // neither path can false-positive.
        self.cleared_any || self.boss_killed
    }
}

// ============================================================
// Double Dragon (NES, Technos/Tradewest, 1988) — beat-'em-up, mission-clear win
// ============================================================
//
// All addresses VERIFIED-LIVE on the Double Dragon (USA) dump (md5 79aa8195…,
// MMC1 / mapper 1) from the mashed Mode-A 1P Mission-1 start (title sequence
// NOOP~120, SELECT~6, NOOP~20, START~8, NOOP~1000 — the generic masher fails;
// SELECT arms the mode menu, START alone at the title does nothing), driving
// the emulator with real inputs:
//   $03B4 player health — 64 full; ROCK-STEADY during safe walking, drops ONLY
//                         in combat, hits 0 exactly at death, and drives the
//                         on-screen bar tiles at $0469+. Cross-sourced:
//                         GameHacking "player energy" = address 948 = $03B4.
//   $0043 lives         — starts 2 (Mode A). Counts DOWN per death 2->1->0
//                         (each respawn refills health to 64, x to 72); the
//                         death AT zero wraps 0->255 = GAME OVER. Cross-sourced:
//                         retro data.json addr 67, GameFAQs "$0043".
//   $005A player x       — 72 start; +RIGHT / -LEFT; saturates ~234 at a
//                         camera-locked enemy gate (mirror $037E).
//   $0042 heart-progress — rises per ENEMY DEFEATED (0 while passive, 1..2 as
//                         kills land); fills a heart then bumps $0040. The
//                         camera stays locked until the wave is cleared, so
//                         this is the dense, NON-FARMABLE gate-progression key.
//   $0040 hearts        — heart count / move unlock; bumps when $0042 fills.
//                         Cross-sourced: GG "Start With All Hearts 0040:07".
//   $0044-$0046 score   — little-endian; rises only when a strike lands.
//                         Cross-sourced: retro data.json addr 68 type <d3.
//   $0030 mission/scene  — VERIFIED constant (=1) through ALL of Mission 1 incl.
//   [win key]             combat + two deaths; transitions 0->1 at Mission-1
//                         start. Its INCREMENT could not be observed (clearing
//                         Mission 1 needs real combat, out of a scripted probe's
//                         reach), so the mission-CLEAR increment is cross-
//                         sourced-by-structure. episode_success keys on an
//                         INCREASE of this byte — it never false-positives on
//                         score/motion/damage, and if $0030 is not the counter
//                         the win simply never fires (safe under-trigger). Made
//                         a parameter (mission_addr) so a verified address can
//                         replace it without a recompile.
//   $03B2 coarse-scroll  — advances in steps as the world scrolls (opt-in
//                         cross-section progress, section_scale; default off
//                         because it jitters under the combat-lock).
// Health is counted on DECREASES only (respawn/section refills never score);
// forward progress is a per-section high-water (knockback/backtracking can't be
// farmed); each life lost is penalized but only GAME OVER ends the episode; and
// the mission-clear bonus dominates every shaping term.

#[derive(Clone)]
pub struct DoubleDragonReward {
    // Weights
    forward_weight: f64,
    enemy_defeat: f64,
    heart_earned: f64,
    score_weight: f64,
    health_penalty: f64,
    death_penalty: f64,
    mission_clear_bonus: f64,
    time_penalty: f64,
    section_scale: f64,
    mission_addr: usize,
    // State
    prev_hp: i32,
    prev_lives: u8,
    prev_hprog: u8,
    prev_hearts: u8,
    prev_score: u64,
    max_x: i32,
    prev_section: i32,
    start_mission: u8,
    mission_cleared: bool,
    game_over: bool,
    first_step: bool,
}

impl DoubleDragonReward {
    const RAM_HEALTH: usize = 0x03B4; // player life bar, 0..64 (VERIFIED-LIVE + GameHacking addr 948)
    const RAM_LIVES: usize = 0x0043; // lives; 2 start, counts down, 0->255 = game over
    const RAM_PLAYER_X: usize = 0x005A; // player screen-x, 72 start (mirror $037E)
    const RAM_HEART_PROGRESS: usize = 0x0042; // rises per enemy defeated
    const RAM_HEARTS: usize = 0x0040; // heart count (move unlock)
    const RAM_SCORE: [usize; 3] = [0x0044, 0x0045, 0x0046]; // little-endian score
    const RAM_SCROLL_COARSE: usize = 0x03B2; // coarse world-scroll / screen column
    const GAME_OVER_MIN: u8 = 200; // lives wraps to 255 on the death at 0 life
    const SECTION_RESET_DX: i32 = 48; // x drop this big = section reload / respawn

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        forward_weight: f64,
        enemy_defeat: f64,
        heart_earned: f64,
        score_weight: f64,
        health_penalty: f64,
        death_penalty: f64,
        mission_clear_bonus: f64,
        time_penalty: f64,
        mission_addr: usize,
        section_scale: f64,
    ) -> Self {
        Self {
            forward_weight,
            enemy_defeat,
            heart_earned,
            score_weight,
            health_penalty,
            death_penalty,
            mission_clear_bonus,
            time_penalty,
            section_scale,
            mission_addr,
            prev_hp: 0,
            prev_lives: 0,
            prev_hprog: 0,
            prev_hearts: 0,
            prev_score: 0,
            max_x: 0,
            prev_section: 0,
            start_mission: 0,
            mission_cleared: false,
            game_over: false,
            first_step: true,
        }
    }

    pub fn reset(&mut self) {
        self.prev_hp = 0;
        self.prev_lives = 0;
        self.prev_hprog = 0;
        self.prev_hearts = 0;
        self.prev_score = 0;
        self.max_x = 0;
        self.prev_section = 0;
        self.start_mission = 0;
        self.mission_cleared = false;
        self.game_over = false;
        self.first_step = true;
    }

    /// Little-endian score at $0044-$0046. Only positive deltas are used (a
    /// strike-landed counter), so the exact encoding is not load-bearing.
    fn read_score(ram: &[u8]) -> u64 {
        (ram[Self::RAM_SCORE[0]] as u64)
            + ((ram[Self::RAM_SCORE[1]] as u64) << 8)
            + ((ram[Self::RAM_SCORE[2]] as u64) << 16)
    }

    pub fn compute(&mut self, ram: &[u8], _action: u8, want_breakdown: bool) -> RewardOutput {
        let hp = ram[Self::RAM_HEALTH] as i32;
        let lives = ram[Self::RAM_LIVES];
        let x = ram[Self::RAM_PLAYER_X] as i32;
        let hprog = ram[Self::RAM_HEART_PROGRESS];
        let hearts = ram[Self::RAM_HEARTS];
        let score = Self::read_score(ram);
        let section = ram[Self::RAM_SCROLL_COARSE] as i32;
        let mission = if self.mission_addr < ram.len() {
            ram[self.mission_addr]
        } else {
            0
        };

        if self.first_step {
            self.prev_hp = hp;
            self.prev_lives = lives;
            self.prev_hprog = hprog;
            self.prev_hearts = hearts;
            self.prev_score = score;
            self.max_x = x;
            self.prev_section = section;
            self.start_mission = mission;
            self.first_step = false;
        }

        let mut acc = RewardAccum::new(want_breakdown);
        let mut done = false;

        // Forward progress: reward new rightward high-water. A big backward jump
        // (x drops past SECTION_RESET_DX) is a section reload / respawn, so reset
        // the water there; small knockback never resets it, so backtracking and
        // getting-hit-bounces can't be farmed.
        if x + Self::SECTION_RESET_DX < self.max_x {
            self.max_x = x;
        }
        let fwd = (x - self.max_x).max(0) as f64;
        acc.add("forward", self.forward_weight * fwd);
        if x > self.max_x {
            self.max_x = x;
        }

        // Enemies defeated — heart-progress ($0042) rises per enemy killed and
        // the camera stays locked until the wave is cleared, so this is the
        // dense, NON-FARMABLE gate-progression signal. Reward INCREASES only
        // (the fill-to-heart reset is a decrease and is ignored; the heart bump
        // is captured below).
        let dkill = hprog.saturating_sub(self.prev_hprog);
        if dkill > 0 {
            acc.add("enemy_defeat", self.enemy_defeat * dkill as f64);
        }
        self.prev_hprog = hprog;

        // Heart earned ($0040) — a full heart-progress cycle unlocks a move.
        if hearts > self.prev_hearts {
            acc.add(
                "heart_earned",
                self.heart_earned * (hearts - self.prev_hearts) as f64,
            );
        }
        self.prev_hearts = hearts;

        // Score delta — minor shaping; only rises when a strike lands.
        acc.add(
            "score",
            self.score_weight * (score as i64 - self.prev_score as i64).max(0) as f64,
        );
        self.prev_score = score;

        // Health lost ($03B4) — penalize DECREASES only, so the respawn/section
        // refill (an increase) never scores.
        let dmg = (self.prev_hp - hp).max(0) as f64;
        acc.add("health_loss", self.health_penalty * dmg);
        self.prev_hp = hp;

        // Optional cross-section coarse-scroll progress ($03B2), high-water. OFF
        // by default (section_scale 0) because the byte jitters under the
        // combat-lock; a verified screen-high byte can arm it later.
        if self.section_scale != 0.0 {
            if section + Self::SECTION_RESET_DX < self.prev_section {
                self.prev_section = section;
            }
            let ds = (section - self.prev_section).max(0) as f64;
            acc.add("section", self.section_scale * ds);
            if section > self.prev_section {
                self.prev_section = section;
            }
        }

        acc.add("time_penalty", self.time_penalty);

        // Death / game-over. Lives ($0043) start at 2 and count DOWN per death
        // (2->1->0), respawning each time; the death AT zero wraps 0->255 = GAME
        // OVER. Penalize each life lost; end the episode only on game over. A
        // legitimate extra-life pickup (lives increases but stays < 200) is
        // neither a death nor a game over.
        let life_lost = lives < self.prev_lives;
        let game_over = lives >= Self::GAME_OVER_MIN && self.prev_lives < Self::GAME_OVER_MIN;
        if life_lost || game_over {
            acc.add("death", self.death_penalty);
        }
        if game_over && !self.game_over {
            self.game_over = true;
            done = true;
        }
        self.prev_lives = lives;

        // WIN — mission cleared. The mission/scene counter ($0030) is stable
        // through an entire mission (verified constant across combat + deaths),
        // so an INCREASE marks a real mission clear and never false-positives on
        // score / motion / damage. Latch + end the episode. Guarded to lives <
        // GAME_OVER_MIN so a game-over frame can never masquerade as a clear.
        if !self.mission_cleared && mission > self.start_mission && lives < Self::GAME_OVER_MIN {
            acc.add("mission_clear", self.mission_clear_bonus);
            self.mission_cleared = true;
            done = true;
        }

        RewardOutput {
            reward: acc.total,
            done,
            level_id: format!("mission_{:02}", mission),
            breakdown_delta: acc.breakdown,
        }
    }

    pub fn episode_success(&self) -> bool {
        self.mission_cleared
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
    // "punch-out" only (NOT bare "punch"): the library also ships "Power Punch
    // II", whose RAM layout differs — it must fall through to Generic.
    if name.contains("punch-out") || name.contains("punch out") || name.contains("punchout") {
        return Some(Reward::PunchOut(PunchOutReward::new(
            w(weights, "opp_damage", 0.5), // per opp-HP unit lost (landed punches)
            w(weights, "knockdown_bonus", 25.0), // each opponent knockdown
            w(weights, "star_bonus", 2.0), // each star banked (clean counter)
            w(weights, "mac_damage", -0.3), // per Mac-HP unit lost
            w(weights, "mac_knockdown", -20.0), // each knockdown Mac takes
            w(weights, "win_bonus", 500.0), // TERMINAL: match won (dominates shaping)
            w(weights, "loss_penalty", -50.0), // TERMINAL: Mac TKO'd (losses++)
            w(weights, "time_penalty", -0.002),
        )));
    }
    if name.contains("kung fu") {
        return Some(Reward::KungFu(KungFuReward::new(
            w(weights, "forward_progress", 0.05),
            w(weights, "score_delta", 0.05),
            w(weights, "floor_clear_bonus", 500.0),
            w(weights, "game_clear_bonus", 5000.0),
            // Boss-HP shaping OFF by default: $04A5 is cross-sourced but was
            // never seen behaving as boss HP in live play (idles at 0x30, only
            // 0 on the death reset). Set >0 once verified to add per-hit boss
            // shaping (guarded against the death-frame reset in compute()).
            w(weights, "boss_damage", 0.0),
            w(weights, "health_penalty", -0.05),
            w(weights, "death_penalty", -25.0),
            w(weights, "time_penalty", -0.005),
            w(weights, "survival_weight", 0.01),
            // floors_cleared >= floor_goal == a real stage-clear win (default
            // 1 = clear Floor 1). game_clear_floor = the $0058 value that means
            // the game is beaten / Sylvia rescued (terminal jackpot).
            w(weights, "floor_goal", 1.0) as u8,
            w(weights, "game_clear_floor", 5.0) as u8,
        )));
    }
    if name.contains("gradius") {
        return Some(Reward::Gradius(GradiusReward::new(
            // SURVIVAL is the dense progress signal for an auto-scroller; over a
            // full 2500-step episode it tops out near 0.05*2500 = 125, so the
            // stage-clear win (1000) strictly dominates it and the power-ups.
            w(weights, "survival_weight", 0.05),
            w(weights, "powerup_bonus", 5.0),
            w(weights, "capsule_bonus", 2.0),
            w(weights, "stage_clear_bonus", 1000.0),
            w(weights, "death_penalty", -25.0),
            w(weights, "time_penalty", -0.002),
            // 0 = stage-clear WIN detection DISABLED (never-false-positive
            // default): Gradius's stage-number byte is undocumented and no
            // empirical candidate proved stable mid-stage, so the win stays
            // unreachable until the real address is verified off a live
            // stage-1->2 transition and set here (as Contra's clear_screen).
            w(weights, "stage_addr", 0.0) as usize,
        )));
    }
    if name.contains("excitebike") || name.contains("excite bike") {
        return Some(Reward::Excitebike(ExcitebikeReward::new(
            // Per unit of within-section distance (0x00ED). A full clean race
            // accrues ~265 distance total, so at 1.0 the shaping sum stays well
            // under the completion bonus.
            w(weights, "forward_progress", 1.0),
            // Milestone each time a track section is completed (0x03A4 +1);
            // ~3 fire per race.
            w(weights, "section_bonus", 25.0),
            // Small throttle-holding shaping, <= this per step (normalized).
            w(weights, "speed_bonus", 0.02),
            // One-shot on a crash/tumble (0x00F2 -> 2).
            w(weights, "crash_penalty", -8.0),
            // One-shot on the engine overheating and stalling (0x03B6 >= 32).
            w(weights, "overheat_penalty", -5.0),
            // TERMINAL: crossing the finish line (0x03A4 reaches the final
            // section). Dominates the entire shaping sum (~365) so finishing
            // strictly beats forever-accruing distance; there is no time limit
            // and the episode ends here, so laps cannot be farmed.
            w(weights, "completion_bonus", 1000.0),
            w(weights, "time_penalty", -0.005),
        )));
    }
    if name.contains("ghosts") {
        return Some(Reward::Ghosts(GhostsReward::new(
            w(weights, "forward_progress", 0.05),
            w(weights, "kill_weight", 0.05),
            w(weights, "armor_hit_penalty", -8.0),
            w(weights, "death_penalty", -25.0),
            w(weights, "stage_cleared", 500.0),
            w(weights, "survival_weight", 0.01),
            w(weights, "time_penalty", -0.005),
            // Path-B "area reload" win: deep world-x then a collapse to ~0.
            // clear_min_progress must be tuned to a real Stage-1 length once a
            // stage transition is reachable; keep it above spawn noise so it
            // can never false-fire. Set very high to disable Path B.
            w(weights, "clear_min_progress", 1500.0) as i32,
            w(weights, "reset_threshold", 200.0) as i32,
            // 0 = optional VERIFIED-stage-byte win path DISABLED (no stage byte
            // is verified on this ROM yet). 0 = score/kill shaping DISABLED (no
            // clean score byte isolated). Set to a verified address to enable.
            w(weights, "stage_addr", 0.0) as usize,
            w(weights, "score_addr", 0.0) as usize,
        )));
    }
    if name.contains("ducktales") {
        return Some(Reward::DuckTales(DuckTalesReward::new(
            w(weights, "forward_progress", 0.01),
            w(weights, "treasure_weight", 0.0005),
            // Dominant win: a level boss's $1,000,000 main treasure = a clear.
            w(weights, "level_clear_bonus", 2000.0),
            w(weights, "damage_penalty", -5.0),
            w(weights, "death_penalty", -25.0),
            w(weights, "survival_weight", 0.01),
            w(weights, "time_penalty", -0.001),
            // Single-step money jump marking a boss's main treasure
            // ($1,000,000) vs the largest gem (a $50,000 red diamond).
            w(weights, "win_treasure_value", 500000.0) as u64,
            w(weights, "level_goal", 1.0) as u32,
        )));
    }
    if name.contains("kid icarus") {
        return Some(Reward::KidIcarus(KidIcarusReward::new(
            w(weights, "score_delta", 0.02),
            w(weights, "health_delta", -3.0),
            w(weights, "stage_cleared", 300.0),
            w(weights, "boss_damage", 5.0),
            w(weights, "boss_killed", 500.0),
            w(weights, "death_penalty", -25.0),
            w(weights, "time_penalty", -0.005),
            // Opt-in vertical-climb shaping. DISABLED by default (weight 0 /
            // addr 0): no verified vertical-scroll accumulator exists on this
            // ROM (Pit's on-screen Y lives in OAM; blind play never scrolled
            // the camera to expose a monotonic altitude byte). Set both to
            // enable once a climb byte (larger = higher) is verified.
            w(weights, "altitude_weight", 0.0),
            w(weights, "altitude_addr", 0.0) as usize,
        )));
    }
    // Base Double Dragon only. The library also ships "Battletoads-Double
    // Dragon", "Double Dragon II", and "Double Dragon III" — different RAM maps
    // — so exclude them (they fall through to Generic until authored).
    if name.contains("double dragon")
        && !name.contains("battletoads")
        && !name.contains(" ii")
    {
        return Some(Reward::DoubleDragon(DoubleDragonReward::new(
            w(weights, "forward_progress", 0.5), // per new-ground px (per-section high-water)
            w(weights, "enemy_defeat", 25.0),    // per enemy killed ($0042 rise) — dense gate signal
            w(weights, "heart_earned", 15.0),    // per heart / move unlock ($0040 rise)
            w(weights, "score_delta", 0.01),     // tiny strike-landed shaping ($0044-46)
            w(weights, "health_penalty", -0.2),  // per life-bar unit lost ($03B4 drop)
            w(weights, "death_penalty", -30.0),  // per life lost + on game over
            w(weights, "mission_clear", 2000.0), // TERMINAL win — dominates all shaping
            w(weights, "time_penalty", -0.01),
            // 0x0030 best-effort mission/scene counter (win key). Replace with a
            // verified address post assisted Mission-1 clear; no recompile needed.
            w(weights, "mission_addr", 48.0) as usize,
            // Opt-in coarse-scroll ($03B2) cross-section progress; 0 = disabled.
            w(weights, "section_scale", 0.0),
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
        // Dracula's internal stage $0028 is 0x12 (18) per the game's
        // StageDataTable. A boss-kill on any earlier stage (0x11 = 17) must
        // NOT count as beating the game.
        let mut r = build_reward("castlevania", &weights).unwrap();
        let mut ram = zram();
        ram[0x0044] = 16; // player health (alive)
        ram[0x002A] = 3; // lives (alive)
        ram[0x0028] = 0x11; // stage 17 (one before Dracula) — NOT a win
        ram[0x01A9] = 100; // boss health (real slot)
        r.compute(&ram, 0, false); // first step baselines prev_boss=100
        ram[0x01A9] = 0; // a boss killed on stage 17
        r.compute(&ram, 0, false);
        assert!(
            !r.episode_success(),
            "a boss-kill before Dracula's stage must NOT beat Castlevania"
        );

        // A boss-kill on DRACULA'S stage ($0028 == 0x12) IS the win.
        let mut r2 = build_reward("castlevania", &weights).unwrap();
        let mut ram2 = zram();
        ram2[0x0044] = 16;
        ram2[0x002A] = 3;
        ram2[0x0028] = 0x12; // Dracula's internal stage (18)
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
    fn punchout_win_only_on_match_id_latch() {
        // A knockdown is NOT a win: only the match-id latch ($0001) going
        // 0 -> nonzero (the winning KO/TKO) counts as beating the opponent.
        let weights = HashMap::new();
        let mut r = build_reward("punch-out", &weights).unwrap();
        let mut ram = zram();
        ram[0x0398] = 0x60; // opponent full HP
        ram[0x0392] = 0x60; // Little Mac full HP
        ram[0x0001] = 0; // match-id latch clear
        ram[0x000A] = 0; // no losses
        r.compute(&ram, 0, false); // seed baselines

        // Non-final opponent knockdown: opp HP hits 0 and the down-flag rises,
        // but the match-id latch is still 0 -> must NOT be a win.
        ram[0x0398] = 0;
        ram[0x03D1] = 1; // opponent-down flag
        let kd = r.compute(&ram, 0, false);
        assert!(!kd.done, "a knockdown mid-bout must not end the episode");
        assert!(
            !r.episode_success(),
            "knocking the opponent down is not yet a match win"
        );

        // The winning KO/TKO: the match-id latch goes 0 -> nonzero.
        ram[0x0001] = 1;
        let win = r.compute(&ram, 0, false);
        assert!(win.done, "the match-win latch ends the episode");
        assert!(
            r.episode_success(),
            "match-id latch 0->nonzero is the real win"
        );
    }

    #[test]
    fn punchout_loss_is_never_success() {
        // The career loss counter ($000A) incrementing (Mac TKO'd) ends the
        // episode as a LOSS, never a success.
        let weights = HashMap::new();
        let mut r = build_reward("punch-out", &weights).unwrap();
        let mut ram = zram();
        ram[0x0398] = 0x60;
        ram[0x0392] = 0x60;
        r.compute(&ram, 0, false);
        ram[0x000A] = 1; // Mac is TKO'd
        let out = r.compute(&ram, 0, false);
        assert!(out.done, "a loss ends the episode");
        assert!(!r.episode_success(), "a loss is never a success");
    }

    #[test]
    fn punchout_opp_damage_knockdown_and_refill() {
        let mut weights = HashMap::new();
        weights.insert("opp_damage".to_string(), 1.0);
        weights.insert("knockdown_bonus".to_string(), 10.0);
        weights.insert("mac_damage".to_string(), 0.0);
        weights.insert("time_penalty".to_string(), 0.0);
        let mut r = build_reward("punch-out", &weights).unwrap();
        let mut ram = zram();
        ram[0x0398] = 0x60; // opp HP 96
        ram[0x0392] = 0x60; // Mac HP 96
        r.compute(&ram, 0, false); // seed

        // Land punches: opp HP 96 -> 91 = 5 damage.
        ram[0x0398] = 0x5B;
        let o1 = r.compute(&ram, 0, false);
        assert!((o1.reward - 5.0).abs() < 1e-6, "5 HP lost * 1.0 = 5.0");

        // Knock the opponent down: opp HP 91 -> 0 (91 damage) + rising-edge
        // knockdown bonus (10).
        ram[0x0398] = 0;
        ram[0x03D1] = 1;
        let o2 = r.compute(&ram, 0, false);
        assert!((o2.reward - (91.0 + 10.0)).abs() < 1e-6);

        // Opponent gets up: HP refills (an INCREASE — must not score) and the
        // down-flag clears (must not re-fire the knockdown bonus).
        ram[0x0398] = 0x60;
        ram[0x03D1] = 0;
        let o3 = r.compute(&ram, 0, false);
        assert!(
            o3.reward.abs() < 1e-6,
            "a get-up HP refill and flag reset must score nothing"
        );
    }

#[test]
fn kung_fu_floor_clear_is_the_win() {
    // Floor clear ($0058 increments) is BOTH the dominant reward and the real
    // win; camping/score can never trip episode_success.
    let mut weights = HashMap::new();
    weights.insert("survival_weight".to_string(), 0.0);
    weights.insert("time_penalty".to_string(), 0.0);
    weights.insert("health_penalty".to_string(), 0.0);
    weights.insert("score_delta".to_string(), 0.0);
    weights.insert("forward_progress".to_string(), 0.0);
    let mut r = build_reward("kung fu", &weights).unwrap();
    let mut ram = vec![0u8; 2048];
    ram[0x0058] = 0; // floor 1
    ram[0x005C] = 3; // 3 lives
    ram[0x04A6] = 48; // full energy
    let _ = r.compute(&ram, 0, false); // seed
    assert!(!r.episode_success(), "no clear yet -> not a win");

    // Clear floor 1 -> floor byte increments to 1.
    ram[0x0058] = 1;
    let o1 = r.compute(&ram, 0, false);
    assert!(
        (o1.reward - 500.0).abs() < 1e-6,
        "a floor clear pays floor_clear_bonus"
    );
    assert!(
        r.episode_success(),
        "one floor cleared == a real stage-clear win"
    );
    assert!(!o1.done, "an intermediate floor clear does not end the episode");
}

#[test]
fn kung_fu_game_clear_is_terminal_and_dominates() {
    // Synthetic: IF the floor byte reaches game_clear_floor (5th-floor boss
    // beaten), the terminal jackpot fires and dominates score/motion.
    let weights = HashMap::new(); // defaults: floor_clear 500, game_clear 5000, floor 5
    let mut r = build_reward("kung fu", &weights).unwrap();
    let mut ram = vec![0u8; 2048];
    ram[0x0058] = 4; // on the 5th/top floor
    ram[0x005C] = 3;
    ram[0x04A6] = 48;
    let _ = r.compute(&ram, 0, false); // seed
    ram[0x0058] = 5; // final boss beaten -> Sylvia rescued
    let o = r.compute(&ram, 0, false);
    assert!(o.done, "reaching the top floor ends the episode");
    assert!(o.reward > 5000.0, "game_clear jackpot dominates (>> score/motion)");
    assert!(r.episode_success());
}

#[test]
fn kung_fu_no_false_win_on_death_and_no_underflow_reward() {
    // Deaths must NOT increment floors_cleared, and the energy-bar underflow
    // (0->255) must pay no spurious reward.
    let mut weights = HashMap::new();
    weights.insert("survival_weight".to_string(), 0.0);
    weights.insert("time_penalty".to_string(), 0.0);
    weights.insert("forward_progress".to_string(), 0.0);
    weights.insert("score_delta".to_string(), 0.0);
    weights.insert("health_penalty".to_string(), -1.0);
    let mut r = build_reward("kung fu", &weights).unwrap();
    let mut ram = vec![0u8; 2048];
    ram[0x0058] = 0;
    ram[0x005C] = 3;
    ram[0x04A6] = 5;
    let _ = r.compute(&ram, 0, false); // seed
    // Take one point of damage -> health_penalty * 1.
    ram[0x04A6] = 4;
    let od = r.compute(&ram, 0, false);
    assert!(
        (od.reward + 1.0).abs() < 1e-6,
        "1 energy lost -> health_penalty*1"
    );
    // Energy underflows 0->255 at death; that is an INCREASE -> no reward.
    ram[0x04A6] = 0xFF;
    let ou = r.compute(&ram, 0, false);
    assert!(ou.reward.abs() < 1e-6, "the 0->255 underflow must not pay out");
    // Life lost -> one-shot death penalty + done; floor never advanced.
    ram[0x005C] = 2;
    let ox = r.compute(&ram, 0, false);
    assert!(
        ox.done && (ox.reward + 25.0).abs() < 1e-6,
        "death: -25 once, ends the episode"
    );
    let oy = r.compute(&ram, 0, false); // penalty must not re-fire
    assert!(oy.reward.abs() < 1e-6);
    assert!(!r.episode_success(), "dying is never a win");
}

#[test]
fn kung_fu_forward_progress_rewards_only_new_ground() {
    // Forward reward fires on a NEW max world-X only; retreating (needed to
    // dodge) earns nothing, so it can't be farmed by pacing.
    let mut weights = HashMap::new();
    weights.insert("survival_weight".to_string(), 0.0);
    weights.insert("time_penalty".to_string(), 0.0);
    weights.insert("health_penalty".to_string(), 0.0);
    weights.insert("score_delta".to_string(), 0.0);
    weights.insert("forward_progress".to_string(), 1.0);
    let mut r = build_reward("kung fu", &weights).unwrap();
    let mut ram = vec![0u8; 2048];
    ram[0x005C] = 3;
    ram[0x04A6] = 48;
    ram[0x0081] = 0;
    ram[0x0090] = 10; // world_x = 10
    let _ = r.compute(&ram, 0, false); // seed max_world_x = 10
    ram[0x0090] = 30; // advance to 30 -> +20 new ground
    let o1 = r.compute(&ram, 0, false);
    assert!(
        (o1.reward - 20.0).abs() < 1e-6,
        "new ground pays forward_progress * dx"
    );
    ram[0x0090] = 20; // retreat -> not a new max -> nothing
    let o2 = r.compute(&ram, 0, false);
    assert!(o2.reward.abs() < 1e-6, "retreating earns nothing (non-farmable)");
}

// --- Gradius ------------------------------------------------------------
    fn gradius_ram_alive() -> Vec<u8> {
        let mut ram = zram();
        ram[0x0020] = 3; // lives
        ram[0x0100] = 1; // status = alive
        ram
    }

    #[test]
    fn gradius_win_only_on_real_stage_transition() {
        // With a stage byte configured, the win fires ONLY when that byte
        // strictly increases (Big Core beaten -> next stage loads). Use an
        // arbitrary safe address (0x0300) as the stand-in stage byte.
        let mut weights = HashMap::new();
        weights.insert("stage_addr".to_string(), 0x0300 as f64);
        let mut r = build_reward("gradius", &weights).unwrap();
        let mut ram = gradius_ram_alive();
        ram[0x0300] = 1; // stage 1 at baseline
        r.compute(&ram, 0, false); // first step: start_stage = max_stage = 1
        // Surviving in-stage (no stage change) is NOT a win, even though the
        // survival term makes the running total positive.
        let out = r.compute(&ram, 0, false);
        assert!(out.reward > 0.0, "survival should net positive per step");
        assert!(
            !r.episode_success(),
            "staying alive in a stage must NOT count as clearing it"
        );
        // Stage byte 1 -> 2: a genuine stage clear.
        ram[0x0300] = 2;
        let out = r.compute(&ram, 0, true);
        assert!(
            out.breakdown_delta.iter().any(|(k, v)| *k == "stage_clear" && *v >= 1000.0),
            "stage transition must pay the dominant stage_clear bonus"
        );
        assert!(
            r.episode_success(),
            "a real stage transition IS a Gradius win"
        );
    }

    #[test]
    fn gradius_no_false_win_when_stage_addr_disabled() {
        // Default profile => stage_addr disabled (0). Even a run that stays
        // alive for many steps (total >> 0) must never report success.
        let weights = HashMap::new();
        let mut r = build_reward("gradius", &weights).unwrap();
        let ram = gradius_ram_alive();
        let mut total = 0.0;
        for _ in 0..50 {
            total += r.compute(&ram, 0, false).reward;
        }
        assert!(total > 0.0, "survival accrues positive reward");
        assert!(
            !r.episode_success(),
            "episode_success must never be the `total > 0` proxy"
        );
    }

    #[test]
    fn gradius_death_penalizes_and_ends_episode() {
        let weights = HashMap::new();
        let mut r = build_reward("gradius", &weights).unwrap();
        let mut ram = gradius_ram_alive();
        r.compute(&ram, 0, false); // baseline: 3 lives, alive
        // Lose a life (3 -> 2): first death ends the episode with a penalty.
        ram[0x0020] = 2;
        let out = r.compute(&ram, 0, true);
        assert!(out.done, "losing a life ends the episode");
        assert!(
            out.breakdown_delta.iter().any(|(k, v)| *k == "death" && *v < 0.0),
            "death must apply a negative penalty"
        );
        assert!(!r.episode_success(), "dying is not a win");
    }

    #[test]
    fn gradius_rewards_power_up_collection() {
        let weights = HashMap::new();
        let mut r = build_reward("gradius", &weights).unwrap();
        let mut ram = gradius_ram_alive();
        r.compute(&ram, 0, false); // baseline: no upgrades
        // Spend a capsule for a Speed-up ($0040: 0 -> 1) => powerup reward.
        ram[0x0040] = 1;
        let out = r.compute(&ram, 0, true);
        assert!(
            out.breakdown_delta.iter().any(|(k, v)| *k == "powerup" && *v > 0.0),
            "applying a power-up must be rewarded"
        );
        // The death-reset of that upgrade ($0040 -> 0) must NOT be a penalty.
        ram[0x0040] = 0;
        let out = r.compute(&ram, 0, true);
        assert!(
            !out.breakdown_delta.iter().any(|(k, _)| *k == "powerup"),
            "losing power-ups (a decrease) must not add or subtract powerup reward"
        );
    }

#[test]
    fn excitebike_dispatches_and_wins_only_on_finish() {
        let weights = HashMap::new();
        assert!(matches!(
            build_reward("excitebike", &weights),
            Some(Reward::Excitebike(_))
        ));

        let mut r = build_reward("excitebike", &weights).unwrap();
        let mut ram = zram();
        // Start-state baseline: section 0, at the line.
        ram[0x03A4] = 0; // section index
        ram[0x00ED] = 0; // section distance
        ram[0x03B6] = 8; // engine temp (cold)
        let _ = r.compute(&ram, 0, false); // first_step seeds baselines

        // Race sections 1 then 2 — NOT a win, no terminal.
        for sec in 1u8..=2 {
            ram[0x03A4] = sec;
            ram[0x00ED] = 5;
            let out = r.compute(&ram, 0, false);
            assert!(!out.done, "mid-track section boundary must not end the episode");
            assert!(
                !r.episode_success(),
                "reaching section {sec} is NOT finishing the track"
            );
        }

        // Cross into the final section (0x03A4 == 3) == finish line crossed.
        ram[0x03A4] = 3;
        let out = r.compute(&ram, 0, true);
        assert!(out.done, "crossing the finish line must end the episode");
        assert!(
            r.episode_success(),
            "reaching the final section is a real win (finished the track)"
        );
        assert!(
            out.breakdown_delta.iter().any(|(k, v)| *k == "completion" && *v >= 1000.0),
            "the completion bonus must fire and dominate"
        );

        // Terminal + one-shot: staying at the finish must not re-fire.
        let out2 = r.compute(&ram, 0, false);
        assert!(!out2.done && out2.reward.abs() < 1.0);
    }

    #[test]
    fn excitebike_shaping_forward_crash_and_no_false_win() {
        let mut weights = HashMap::new();
        weights.insert("forward_progress".to_string(), 1.0);
        weights.insert("section_bonus".to_string(), 0.0);
        weights.insert("speed_bonus".to_string(), 0.0);
        weights.insert("crash_penalty".to_string(), -8.0);
        weights.insert("overheat_penalty".to_string(), 0.0);
        weights.insert("time_penalty".to_string(), 0.0);
        let mut r = build_reward("excitebike", &weights).unwrap();
        let mut ram = zram();
        ram[0x00ED] = 10; // section distance baseline
        let _ = r.compute(&ram, 0, false); // seed

        // Forward motion: +10 units of distance -> +10 reward.
        ram[0x00ED] = 20;
        let o1 = r.compute(&ram, 0, false);
        assert!((o1.reward - 10.0).abs() < 1e-6);

        // A section reset (129 -> 2) must NOT produce a negative forward term.
        ram[0x00ED] = 129;
        let _ = r.compute(&ram, 0, false);
        ram[0x00ED] = 2;
        let o2 = r.compute(&ram, 0, false);
        assert!(o2.reward.abs() < 1e-6, "section reset must clamp to 0, not go negative");

        // Crash edge (0 -> 2) fires once; holding 2 does not re-penalize.
        ram[0x00F2] = 2;
        let o3 = r.compute(&ram, 0, false);
        assert!((o3.reward + 8.0).abs() < 1e-6);
        let o4 = r.compute(&ram, 0, false);
        assert!(o4.reward.abs() < 1e-6, "a held crash status must not re-fire the penalty");

        // Never reached the final section -> not a win.
        assert!(!r.episode_success());
    }

// Place inside the existing `#[cfg(test)] mod tests { ... }` block in
// nes_core/src/rewards.rs (reuses that module's `zram()` -> vec![0u8; 2048]).

    fn gng_set_wx(ram: &mut [u8], wx: i32) {
        let v = (wx as i64 & 0xFFFF) as u16;
        ram[0x005D] = (v & 0xFF) as u8;
        ram[0x005E] = (v >> 8) as u8;
    }

    #[test]
    fn ghosts_win_on_area_reload_after_deep_progress() {
        // Path B: deep world-x, then a collapse to ~0 while alive & playing =
        // the next area/stage loaded = a real clear. Uses only verified bytes.
        let mut r = GhostsReward::new(
            0.05, 0.05, -8.0, -25.0, 500.0, 0.01, -0.005, 1000, 200, 0, 0,
        );
        let mut ram = zram();
        ram[0x0028] = 0x08; // mode = active gameplay
        ram[0x0715] = 2; // lives
        ram[0x0585] = 1; // armored
        gng_set_wx(&mut ram, 0);
        r.compute(&ram, 0, false); // first step baselines

        for x in (100..=1200).step_by(100) {
            gng_set_wx(&mut ram, x);
            r.compute(&ram, 0, false);
        }
        assert!(!r.episode_success(), "must not win merely by advancing");

        gng_set_wx(&mut ram, 0); // area reload: scroll collapses
        let out = r.compute(&ram, 0, false);
        assert!(out.done, "clear must end the episode");
        assert!(r.episode_success(), "deep progress + reload = stage clear");
        assert!(
            out.reward > 400.0,
            "stage_clear_bonus must dominate (got {})",
            out.reward
        );
    }

    #[test]
    fn ghosts_no_win_on_death_respawn() {
        // Dying deep in a stage then respawning (scroll back to 0) must NOT be
        // read as a clear: the life-loss ends the episode and wipes progress.
        let mut r = GhostsReward::new(
            0.05, 0.05, -8.0, -25.0, 500.0, 0.01, -0.005, 1000, 200, 0, 0,
        );
        let mut ram = zram();
        ram[0x0028] = 0x08;
        ram[0x0715] = 2;
        ram[0x0585] = 1;
        gng_set_wx(&mut ram, 0);
        r.compute(&ram, 0, false);
        for x in (100..=1200).step_by(100) {
            gng_set_wx(&mut ram, x);
            r.compute(&ram, 0, false);
        }
        ram[0x0715] = 1; // died
        gng_set_wx(&mut ram, 0); // respawn scroll reset (same frame)
        let out = r.compute(&ram, 0, false);
        assert!(out.done, "death ends the episode");
        assert!(!r.episode_success(), "death is never a win");
        assert!(out.reward < 0.0, "death frame must be net-negative");
    }

    #[test]
    fn ghosts_armor_loss_penalized_once() {
        // Path B disabled (huge clear_min_progress); survival/time zeroed so
        // the reward on the transition is exactly the armor penalty.
        let mut r = GhostsReward::new(
            0.05, 0.05, -8.0, -25.0, 500.0, 0.0, 0.0, 1_000_000, 200, 0, 0,
        );
        let mut ram = zram();
        ram[0x0028] = 0x08;
        ram[0x0715] = 2;
        ram[0x0585] = 1; // armored
        gng_set_wx(&mut ram, 0);
        r.compute(&ram, 0, false);

        ram[0x0585] = 0; // armored -> underwear
        let out = r.compute(&ram, 0, false);
        assert!((out.reward - (-8.0)).abs() < 1e-9, "one-shot armor penalty");

        let out2 = r.compute(&ram, 0, false); // stays underwear
        assert!(out2.reward.abs() < 1e-9, "no repeat armor penalty");
        assert!(!r.episode_success());
    }

// Paste into the `mod tests` block in nes_core/src/rewards.rs (it already has
// `use super::*;`). Verifies the win predicate + the damage shaping term on
// synthetic RAM.

/// Write the 7-tile HUD money field $0324-$032A (MSB first, digit tiles 0-9).
fn dt_set_money(ram: &mut [u8], dollars: u64) {
    let s = format!("{:07}", dollars.min(9_999_999));
    for (i, c) in s.bytes().enumerate() {
        ram[0x0324 + i] = c - b'0';
    }
}

#[test]
fn ducktales_level_clear_only_on_main_treasure() {
    let weights = HashMap::new();
    let mut r = build_reward("DuckTales (USA)", &weights).unwrap();
    let mut ram = vec![0u8; 2048];
    ram[0x00DD] = 3; // alive, so HP never ends the episode here
    dt_set_money(&mut ram, 0);
    r.compute(&ram, 0, false); // first step baselines money = 0

    // A $50,000 red diamond (the largest gem) is NOT a level clear.
    dt_set_money(&mut ram, 50_000);
    let out = r.compute(&ram, 0, false);
    assert!(!r.episode_success(), "a $50k gem must NOT clear a level");
    assert!(!out.done, "a gem pickup does not end the episode");

    // The boss's $1,000,000 main treasure, claimed in one pickup, IS the clear.
    dt_set_money(&mut ram, 1_050_000); // +1,000,000 single-step jump
    let out = r.compute(&ram, 0, false);
    assert!(r.episode_success(), "the $1,000,000 main treasure = level clear");
    assert!(out.done, "clearing a level ends the episode");
}

#[test]
fn ducktales_cumulative_gems_never_win() {
    // Gems that SUM past the threshold must not win: each single-step delta
    // stays at/below the $50k red-diamond cap.
    let weights = HashMap::new();
    let mut r = build_reward("ducktales", &weights).unwrap();
    let mut ram = vec![0u8; 2048];
    ram[0x00DD] = 3;
    r.compute(&ram, 0, false);
    let mut total = 0u64;
    for _ in 0..40 {
        total += 50_000; // a red diamond each step -> $2,000,000 cumulative
        dt_set_money(&mut ram, total);
        let out = r.compute(&ram, 0, false);
        assert!(!out.done, "gem-by-gem farming must never end the episode");
    }
    assert!(!r.episode_success(), "cumulative gem money is NOT a level clear");
}

#[test]
fn ducktales_damage_penalty_and_death() {
    let weights = HashMap::new();
    let mut r = build_reward("ducktales", &weights).unwrap();
    let mut ram = vec![0u8; 2048];
    ram[0x00DD] = 3;
    r.compute(&ram, 0, false); // baseline hp = 3

    ram[0x00DD] = 2; // took a hit
    let out = r.compute(&ram, 0, false);
    assert!(out.reward < 0.0, "losing a heart must be penalized");
    assert!(!r.episode_success(), "a hit is not a win");

    ram[0x00DD] = 0; // out of hearts -> death
    let out2 = r.compute(&ram, 0, false);
    assert!(out2.done, "HP reaching 0 ends the episode");
    assert!(!r.episode_success(), "dying is not a win");
}

#[test]
    fn kid_icarus_win_only_on_stage_clear_or_boss() {
        let weights = HashMap::new();
        // Stage clear: $0130 advancing beyond the episode's start stage wins.
        let mut r = build_reward("kid icarus", &weights).unwrap();
        let mut ram = zram();
        ram[0x00A6] = 7; // alive
        ram[0x0130] = 0; // start stage
        r.compute(&ram, 0, false); // baseline start_stage = 0
        assert!(!r.episode_success(), "no win at the start of a stage");
        ram[0x0130] = 1; // reached the next sub-stage
        let out = r.compute(&ram, 0, false);
        assert!(r.episode_success(), "advancing the stage index is a stage clear");
        assert!(out.reward > 100.0, "the stage-clear milestone dominates score/motion");

        // Death alone is NOT a win, and ends the episode.
        let mut r2 = build_reward("kid icarus", &weights).unwrap();
        let mut ram2 = zram();
        ram2[0x00A6] = 7;
        ram2[0x0130] = 0;
        r2.compute(&ram2, 0, false);
        ram2[0x00A6] = 0; // fell off the bottom / HP depleted
        let out2 = r2.compute(&ram2, 0, false);
        assert!(out2.done, "hp==0 ends the episode");
        assert!(!r2.episode_success(), "dying is never a win");

        // Fortress boss defeat ($006B >0 -> 0) wins; a spawn-time 0 does not.
        let mut r3 = build_reward("kid icarus", &weights).unwrap();
        let mut ram3 = zram();
        ram3[0x00A6] = 7;
        ram3[0x0130] = 3;
        ram3[0x006B] = 0; // boss=0 at spawn
        r3.compute(&ram3, 0, false);
        assert!(!r3.episode_success(), "boss=0 at spawn must not count as a kill");
        ram3[0x006B] = 40; // boss appears
        r3.compute(&ram3, 0, false);
        ram3[0x006B] = 0; // boss defeated
        r3.compute(&ram3, 0, false);
        assert!(r3.episode_success(), "defeating a fortress boss is a real win");

        // Pure score farming (no stage advance) is never a win.
        let mut r4 = build_reward("kid icarus", &weights).unwrap();
        let mut ram4 = zram();
        ram4[0x00A6] = 7;
        ram4[0x0130] = 0;
        r4.compute(&ram4, 0, false);
        ram4[0x0131] = 250; // farmed kills, no stage advance
        r4.compute(&ram4, 0, false);
        assert!(!r4.episode_success(), "score farming is not a win");
    }

    #[test]
    fn kid_icarus_score_and_damage_shaping() {
        let mut weights = HashMap::new();
        weights.insert("score_delta".to_string(), 0.1);
        weights.insert("health_delta".to_string(), -2.0);
        weights.insert("time_penalty".to_string(), 0.0);
        let mut r = build_reward("kid icarus", &weights).unwrap();
        let mut ram = zram();
        ram[0x00A6] = 7;
        ram[0x0131] = 0;
        ram[0x0132] = 0;
        ram[0x0133] = 0;
        r.compute(&ram, 0, false); // baseline
        // Score is little-endian BINARY: 300 => lo 44 + mid 1*256.
        ram[0x0131] = 100; // +100 for a kill
        let out = r.compute(&ram, 0, false);
        assert!((out.reward - 10.0).abs() < 1e-9, "score +100 * 0.1 = +10");
        ram[0x00A6] = 6; // damage 7 -> 6
        let out2 = r.compute(&ram, 0, false);
        assert!((out2.reward - (-2.0)).abs() < 1e-9, "losing 1 HP * -2.0 = -2");
    }

// Drop into `mod tests` (uses the existing zram()). Mirrors the punchout tests.

#[test]
fn double_dragon_win_only_on_mission_increment() {
    // Killing enemies and walking right is NOT a mission clear — only the
    // mission/scene counter ($0030) INCREASING above its start value is.
    let weights = HashMap::new();
    let mut r = build_reward("double dragon", &weights).unwrap();
    let mut ram = zram();
    ram[0x03B4] = 64; // full health
    ram[0x0043] = 2; // 2 lives
    ram[0x005A] = 72; // start x
    ram[0x0030] = 1; // mission 1
    r.compute(&ram, 0, false); // seed baselines

    ram[0x0042] = 3; // 3 enemies defeated
    ram[0x005A] = 200; // walked right
    let mid = r.compute(&ram, 0, false);
    assert!(!mid.done, "combat / forward progress must not end the episode");
    assert!(!r.episode_success(), "killing enemies is not a mission clear");

    ram[0x0030] = 2; // mission counter 1 -> 2: the real clear
    let win = r.compute(&ram, 0, false);
    assert!(win.done, "a mission increment ends the episode");
    assert!(r.episode_success(), "mission 1->2 is the real win");
}

#[test]
fn double_dragon_game_over_only_on_wrap_never_success() {
    // Lives count DOWN per death (respawn each) and only the 0->255 wrap is
    // game over; game over is never a success.
    let weights = HashMap::new();
    let mut r = build_reward("double dragon", &weights).unwrap();
    let mut ram = zram();
    ram[0x03B4] = 64;
    ram[0x0043] = 2;
    ram[0x0030] = 1;
    r.compute(&ram, 0, false); // seed lives=2

    ram[0x0043] = 1; // 2 -> 1: penalized, NOT done (respawns)
    let d1 = r.compute(&ram, 0, false);
    assert!(!d1.done, "losing one of several lives must not end the episode");
    assert!(d1.reward < 0.0, "a lost life is penalized");

    ram[0x0043] = 0; // 1 -> 0: still alive on the last life
    let d2 = r.compute(&ram, 0, false);
    assert!(!d2.done, "the last life is still in play");

    ram[0x0043] = 255; // 0 -> 255 (death at zero): GAME OVER
    let go = r.compute(&ram, 0, false);
    assert!(go.done, "the 0->255 wrap is game over");
    assert!(!r.episode_success(), "game over is never a mission clear");
}

#[test]
fn double_dragon_enemy_defeat_and_health_are_delta_gated() {
    // enemy_defeat scores $0042 INCREASES; the fill-to-heart reset (a decrease)
    // must not score, and the health refill on that frame must not penalize.
    let mut weights = HashMap::new();
    weights.insert("forward_progress".to_string(), 0.0);
    weights.insert("time_penalty".to_string(), 0.0);
    weights.insert("enemy_defeat".to_string(), 25.0);
    weights.insert("health_penalty".to_string(), -1.0);
    let mut r = build_reward("double dragon", &weights).unwrap();
    let mut ram = zram();
    ram[0x03B4] = 64;
    ram[0x0043] = 2;
    ram[0x0030] = 1;
    r.compute(&ram, 0, false); // seed

    ram[0x0042] = 2; // 2 kills (+50)
    ram[0x03B4] = 54; // took 10 damage (-10)
    let o1 = r.compute(&ram, 0, false);
    assert!((o1.reward - (50.0 - 10.0)).abs() < 1e-6, "2 kills*25 - 10 dmg = 40");

    ram[0x0042] = 0; // heart fills: $0042 resets 2->0 (decrease, must not score)
    ram[0x0040] = 1; // a heart earned
    ram[0x03B4] = 64; // health refills (increase, no penalty)
    let o2 = r.compute(&ram, 0, false);
    assert!(o2.reward >= 0.0, "a heart-fill reset must never produce a penalty");
    assert!(!o2.done);
}

#[test]
fn double_dragon_forward_is_high_water_not_farmable() {
    let mut weights = HashMap::new();
    weights.insert("forward_progress".to_string(), 1.0);
    weights.insert("time_penalty".to_string(), 0.0);
    let mut r = build_reward("double dragon", &weights).unwrap();
    let mut ram = zram();
    ram[0x03B4] = 64;
    ram[0x0043] = 2;
    ram[0x0030] = 1;
    ram[0x005A] = 72;
    r.compute(&ram, 0, false); // seed max_x=72

    ram[0x005A] = 120; // +48 new ground
    let a = r.compute(&ram, 0, false);
    assert!((a.reward - 48.0).abs() < 1e-6, "advancing 48 px earns 48");

    ram[0x005A] = 100; // small knockback — no reward, no high-water reset
    let b = r.compute(&ram, 0, false);
    assert!(b.reward.abs() < 1e-6, "backtracking earns nothing");

    ram[0x005A] = 120; // re-cover ground — still nothing (high-water)
    let c = r.compute(&ram, 0, false);
    assert!(c.reward.abs() < 1e-6, "re-covering ground can't be farmed");

    ram[0x005A] = 130; // new ground beyond the high-water
    let d = r.compute(&ram, 0, false);
    assert!((d.reward - 10.0).abs() < 1e-6, "only new ground scores");
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
    fn mario_score_drop_is_clamped_not_penalized() {
        // Regression guard: SMB's on-screen score is decoded digit-by-digit
        // from RAM $07DE-$07E3 and rolls over from 999999 back to 000000
        // once the 6-digit counter overflows during a long/high-scoring
        // episode. Before this fix, the "score" term was the only
        // score-delta reward in the file that was NOT clamped to
        // non-negative, so a rollover (or any transient misread) produced
        // an unbounded negative spike (~0.025 * -999900) with no
        // corresponding death/termination. Every other hand-authored
        // game's score term (Contra, Castlevania, Metroid, KungFu,
        // KidIcarus, DoubleDragon) clamps with `.max(0)`; Mario must too.
        let mut weights = HashMap::new();
        weights.insert("score_delta".to_string(), 1.0);
        weights.insert("forward_progress".to_string(), 0.0);
        weights.insert("time_penalty".to_string(), 0.0);
        weights.insert("checkpoint_scale".to_string(), 0.0);

        let mut r = build_reward("mario", &weights).unwrap();
        let mut ram = zram();
        // Keep x, lives, and player_state constant throughout so the
        // score term is the only non-zero contribution.
        ram[0x006D] = 0;
        ram[0x0086] = 0x40;
        let _ = r.compute(&ram, 0, false); // first_step seeds prev_score=0

        // Score climbs to 999900 (digits 9,9,9,9,0,0 at $07DE-$07E3).
        ram[0x07DE] = 9;
        ram[0x07DF] = 9;
        ram[0x07E0] = 9;
        ram[0x07E1] = 9;
        ram[0x07E2] = 0;
        ram[0x07E3] = 0;
        let up = r.compute(&ram, 0, false);
        assert!(
            (up.reward - 999900.0).abs() < 1.0,
            "expected the positive score delta to pass through unclamped, got {}",
            up.reward
        );

        // Score wraps back to 000000 — a real NES rollover, or any
        // transient misread of the digit window. The agent is still
        // alive (no death/lives-drop), so this must NOT punch a huge
        // negative reward through.
        ram[0x07DE] = 0;
        ram[0x07DF] = 0;
        ram[0x07E0] = 0;
        ram[0x07E1] = 0;
        ram[0x07E2] = 0;
        ram[0x07E3] = 0;
        let wrap = r.compute(&ram, 0, false);
        assert!(
            wrap.reward >= 0.0,
            "a decreasing/wrapped score must be clamped to a non-negative \
             reward contribution, got {} (unclamped would be ~-999900)",
            wrap.reward
        );
        assert!(!wrap.done, "a score wraparound alone must not end the episode");
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
        // 2-1 has its own calibrated table; every other uncalibrated
        // level gets the generic every-256px ladder (never again a
        // reward desert — the World-2 fallthrough that cost 2-1
        // seventeen machine-hours).
        assert_eq!(MarioReward::checkpoints_for(0, 3), MarioReward::LEVEL_1_3);
        assert_eq!(MarioReward::checkpoints_for(0, 4), MarioReward::LEVEL_1_4);
        assert_eq!(MarioReward::checkpoints_for(1, 0), MarioReward::LEVEL_2_1);
        assert_eq!(
            MarioReward::checkpoints_for(2, 0),
            MarioReward::GENERIC_LADDER
        );
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
    fn contra_clear_then_death_still_success() {
        // Contra's completion (reaching the measured stage-end screen) is a
        // durable latch that does NOT end the episode, so a genuine clear can be
        // followed by a later death. That must still count as success — the same
        // died-negation-after-durable-clear shape fixed for Mario in ISSUE-7.
        let mut weights = HashMap::new();
        weights.insert("forward_progress".to_string(), 0.0);
        weights.insert("score_delta".to_string(), 0.0);
        weights.insert("time_penalty".to_string(), 0.0);
        weights.insert("death_penalty".to_string(), 0.0);
        weights.insert("completion_bonus".to_string(), 100.0);
        weights.insert("clear_screen".to_string(), 5.0);
        let mut r = build_reward("contra", &weights).unwrap();
        let mut ram = zram();
        ram[0x0032] = 3; // lives
        ram[0x0064] = 0; // seed at screen 0
        let _ = r.compute(&ram, 0, false);
        // Reach the clear screen -> completion latches, success true.
        ram[0x0064] = 5;
        let _ = r.compute(&ram, 0, false);
        assert!(r.episode_success(), "reaching clear_screen must latch success");
        // Die afterwards (lives drop): the episode ends but the clear stands.
        ram[0x0032] = 2;
        let o = r.compute(&ram, 0, false);
        assert!(o.done, "a life lost must end the episode");
        assert!(
            r.episode_success(),
            "a real stage clear followed by a later death is still a success"
        );
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
        ram[0x075C] = 3; // display level 3 => x-4 castle (emulator-verified)
        let _ = r.compute(&ram, 0, false); // first step seeds prev state
        assert!(!r.episode_success(), "castle not cleared yet");
        // Beat the castle: world byte increments to 1 (world 2), level resets 0.
        ram[0x075F] = 1;
        ram[0x075C] = 0;
        let o = r.compute(&ram, 0, false);
        assert!(
            o.reward > 500.0,
            "castle clear must fire the completion bonus, got {}",
            o.reward
        );
        assert!(
            r.episode_success(),
            "beating a castle (world++ from x-4) must count as success"
        );
    }

    #[test]
    fn mario_warp_pipe_world_jump_is_not_a_castle_clear() {
        // F52 regression: a warp-zone pipe (1-2, display level 1) jumps the
        // world byte WITHOUT beating a castle. It must NOT fire completion or
        // count as success (the earlier fix wrongly credited every world rise).
        let mut weights: HashMap<String, f64> = HashMap::new();
        weights.insert("completion_bonus".to_string(), 1000.0);
        weights.insert("time_penalty".to_string(), 0.0);
        weights.insert("forward_progress".to_string(), 0.0);
        let mut r = build_reward("mario", &weights).unwrap();
        let mut ram = zram();
        ram[0x075A] = 3;
        ram[0x075F] = 0; // world 1
        ram[0x075C] = 1; // display level 1 => x-2 (the warp-zone level)
        let _ = r.compute(&ram, 0, false);
        // Warp: world jumps to 3 (world 4), from x-2 — NOT a castle clear.
        ram[0x075F] = 3;
        ram[0x075C] = 0;
        let o = r.compute(&ram, 0, false);
        assert!(o.reward < 500.0, "a warp must not fire the completion bonus, got {}", o.reward);
        assert!(!r.episode_success(), "a warp-zone world jump is not a win");
    }

    #[test]
    fn mario_within_world_level_change_still_re_arms() {
        // Guard the F52 change: a within-world level change (1-1 -> 1-2, world
        // byte unchanged) must STILL re-arm completed=false so each flagpole
        // clear re-fires its own bonus — the working 1-1/1-2/1-3 recipe must be
        // untouched. Re-arm is now verified via the reward output (the 1-2
        // flagpole re-fires completion), NOT via episode_success, which is a
        // durable per-episode latch (`cleared_any`) that must survive the change.
        let mut weights: HashMap<String, f64> = HashMap::new();
        weights.insert("completion_bonus".to_string(), 1000.0);
        weights.insert("time_penalty".to_string(), 0.0);
        weights.insert("forward_progress".to_string(), 0.0);
        let mut r = build_reward("mario", &weights).unwrap();
        let mut ram = zram();
        ram[0x075A] = 3;
        ram[0x075F] = 0; // world 1 throughout
        ram[0x0760] = 0; // 1-1
        let _ = r.compute(&ram, 0, false);
        ram[0x001D] = 3; // flagpole slide -> completion latches
        let o1 = r.compute(&ram, 0, false);
        assert!(o1.reward > 500.0, "1-1 flagpole must fire completion, got {}", o1.reward);
        assert!(r.episode_success(), "flagpole clear latches success");
        // Advance to 1-2 (area 2), world byte unchanged -> `completed` re-arms
        // for reward purposes, but the durable success latch must persist.
        ram[0x001D] = 0;
        ram[0x0760] = 2;
        let _ = r.compute(&ram, 0, false);
        assert!(
            r.episode_success(),
            "durable clear latch must survive the level change (curriculum can advance)"
        );
        // Touch the 1-2 flagpole -> completion must fire AGAIN, proving `completed`
        // re-armed (the working per-level bonus recipe is untouched).
        ram[0x001D] = 3;
        let o2 = r.compute(&ram, 0, false);
        assert!(
            o2.reward > 500.0,
            "re-armed `completed` must let the 1-2 flagpole re-fire completion, got {}",
            o2.reward
        );
    }

    #[test]
    fn mario_clear_then_level_change_then_death_is_success() {
        // ISSUE-7: an agent that clears 1-1 (flagpole), transitions to 1-2 (which
        // re-arms `completed`), then dies in 1-2 — the normal way a long SMB
        // episode ends — must STILL report episode_success. The old
        // `(completed || won) && !died` predicate scored every such real clear as
        // a failure, so the ga_ppo tile-mode curriculum (keyed on the rolling
        // stage_success_rate) could never advance despite agents reaching 1-2/1-3.
        let mut weights: HashMap<String, f64> = HashMap::new();
        weights.insert("completion_bonus".to_string(), 1000.0);
        weights.insert("time_penalty".to_string(), 0.0);
        let mut r = build_reward("mario", &weights).unwrap();
        let mut ram = zram();
        ram[0x075A] = 3; // lives
        ram[0x075F] = 0; // world 1
        ram[0x0760] = 0; // 1-1
        let _ = r.compute(&ram, 0, false); // seed prev state
        // Clear 1-1 via the flagpole slide.
        ram[0x001D] = 3;
        let _ = r.compute(&ram, 0, false);
        assert!(r.episode_success(), "flagpole clear must latch success");
        // Transition to 1-2 (area 2): re-arms `completed` to false.
        ram[0x001D] = 0;
        ram[0x0760] = 2;
        let _ = r.compute(&ram, 0, false);
        // Die in 1-2: lives drop -> death latches and the episode ends.
        ram[0x075A] = 2;
        let o = r.compute(&ram, 0, false);
        assert!(o.done, "a life lost must end the episode");
        assert!(
            r.episode_success(),
            "a real 1-1 clear followed by a later death is a curriculum success"
        );
    }

    #[test]
    fn mario_death_without_clear_is_not_success() {
        // Negative case: an episode that never touches a flagpole or beats a
        // castle and ends in death must NOT report success — `cleared_any` stays
        // false, so the durable latch cannot false-positive from a plain death.
        let weights: HashMap<String, f64> = HashMap::new();
        let mut r = build_reward("mario", &weights).unwrap();
        let mut ram = zram();
        ram[0x075A] = 3; // lives
        let _ = r.compute(&ram, 0, false); // seed prev state
        assert!(!r.episode_success(), "no clear yet -> not a success");
        // Die with no prior clear.
        ram[0x075A] = 2;
        let o = r.compute(&ram, 0, false);
        assert!(o.done, "a life lost must end the episode");
        assert!(!r.episode_success(), "death without any clear is not a success");
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

    #[test]
    fn contra_forward_reward_tracks_fine_scroll_not_the_dead_byte() {
        // RAM_PLAYER_X was 0x0031 (verified live: constant 0 across 2000
        // frames of real varied play) until this fix moved it to 0x0065,
        // the fine-scroll register that empirically wraps 255->0 exactly
        // on every $0064 screen increment — matching the `screen*256 + x`
        // formula's own assumption. Before the fix this test's within-
        // screen movement would have scored zero forward reward.
        let mut weights = HashMap::new();
        weights.insert("forward_progress".to_string(), 1.0);
        weights.insert("score_delta".to_string(), 0.0);
        weights.insert("time_penalty".to_string(), 0.0);
        weights.insert("death_penalty".to_string(), 0.0);
        let mut r = build_reward("contra", &weights).unwrap();
        let mut ram = zram();
        ram[0x0032] = 3; // lives
        ram[0x0064] = 0; // screen
        ram[0x0065] = 10; // fine-scroll x
        let _ = r.compute(&ram, 0, false); // seeds prev_x/prev_screen

        // Within-screen movement (no screen-boundary crossing) must reward.
        ram[0x0065] = 40;
        let o1 = r.compute(&ram, 0, false);
        assert!((o1.reward - 30.0).abs() < 1e-6,
                "within-screen fine-scroll movement should reward +30, got {}",
                o1.reward);

        // A screen-boundary crossing (fine wraps low, screen increments)
        // must still reward the full 256*screen_delta + new-x amount.
        ram[0x0064] = 1;
        ram[0x0065] = 5;
        let o2 = r.compute(&ram, 0, false);
        assert!((o2.reward - (256.0 + 5.0 - 40.0)).abs() < 1e-6,
                "screen-boundary crossing should reward 256+5-40=221, got {}",
                o2.reward);
    }
}
