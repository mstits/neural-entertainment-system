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
    // State
    visited: HashSet<(u8, u8)>,
    dungeons_entered: HashSet<u8>,
    prev_hearts_byte: u8,
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
    const RAM_RUPEES: usize = 0x066D;
    const RAM_TRIFORCE: usize = 0x0671;
    const RAM_DUNGEON_LEVEL: usize = 0x10;
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
            visited: HashSet::new(),
            dungeons_entered: HashSet::new(),
            prev_hearts_byte: 0,
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

        if curr_cur == 0 && prev_cur > 0 {
            acc.add("death", self.death_penalty);
            self.died = true;
            done = true;
        }

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
        self.prev_triforce.count_ones() == 8 && !self.died
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
        (2100, 400.0),  // past obstacle complex
        (2700, 600.0),  // at staircase base
        (2900, 1000.0), // top of staircase, near flag
    ];

    /// 1-2 is the underground bonus-room level. Obstacles: jumping
    /// pipes (~x=400), green pipe gauntlet (~x=1300), final descent
    /// to warp zone OR continuation gauntlet (~x=2300). Calibrated
    /// from emulator playthrough — not as obstacle-dense as 1-1 but
    /// still has clear chokepoints worth signaling.
    const LEVEL_1_2: &'static [(u32, f64)] = &[
        (400, 50.0),    // past first jumping-pipe section
        (900, 100.0),   // past mid-level brick clusters
        (1300, 200.0),  // past green pipe gauntlet
        (1900, 350.0),  // past long-jump section
        (2300, 500.0),  // approach to warp zone / level end
        (2700, 1000.0), // near pipe to overworld / 1-3
    ];

    /// Look up the dense-reward checkpoints for a given (world, level).
    /// Returns an empty slice for levels we haven't tuned, which means
    /// only the baseline forward_progress + completion_bonus fire on
    /// those levels. Not harmful — just less dense signal until the
    /// table is calibrated.
    fn checkpoints_for(world: u8, level: u8) -> &'static [(u32, f64)] {
        match (world, level) {
            (0, 0) => Self::LEVEL_1_1,
            (0, 1) => Self::LEVEL_1_2,
            // 1-3, 1-4, world 2+ — TODO: calibrate per level. Falls
            // through to empty (forward + completion only).
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
    ) -> Self {
        Self {
            forward_weight,
            score_weight,
            death_penalty,
            completion_bonus,
            time_penalty,
            checkpoint_scale,
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
            self.prev_x = x;
            self.max_x_visited = x;
            // Reset checkpoint cursor on every level transition so the
            // bonuses re-fire on the new level. Without this, levels
            // 1-2+ would never produce checkpoint signal because all
            // were "consumed" during 1-1.
            self.next_checkpoint = 0;
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
            self.died = true;
            done = true;
        }
        self.prev_lives = lives;

        let float_state = ram[Self::RAM_FLOAT_STATE];
        if !self.completed && float_state == 3 {
            acc.add("completion", self.completion_bonus);
            self.completed = true;
            done = true;
        }

        acc.add("time_penalty", self.time_penalty);

        let level_id = format!("{}-{}", world as u16 + 1, level as u16 + 1);
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
    prev_x: u8,
    prev_screen: u8,
    prev_score: u64,
    prev_weapon: u8,
    prev_lives: u8,
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
    ) -> Self {
        Self {
            forward_weight,
            score_weight,
            weapon_upgrade_bonus,
            death_penalty,
            time_penalty,
            prev_x: 0,
            prev_screen: 0,
            prev_score: 0,
            prev_weapon: 0,
            prev_lives: 0,
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
        false
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
}

impl CastlevaniaReward {
    const RAM_PLAYER_HEALTH: usize = 0x0044;
    const RAM_STAGE: usize = 0x0028;
    const RAM_LIVES: usize = 0x002A;
    const RAM_HEARTS: usize = 0x0045;
    const RAM_BOSS_HEALTH: usize = 0x04A0;
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
        self.prev_stage > 0 && !self.died
    }
}

// ============================================================
// Metroid
// ============================================================

#[derive(Clone)]
pub struct MetroidReward {
    exploration_bonus: f64,
    missile_weight: f64,
    _energy_tank_bonus: f64,
    new_item_bonus: f64,
    _boss_damage_weight: f64,
    _boss_killed_bonus: f64,
    health_delta_weight: f64,
    death_penalty: f64,
    time_penalty: f64,
    visited: HashSet<(u8, u8)>,
    prev_missiles: u8,
    prev_health_total: u32,
    prev_item_flags: u8,
    died: bool,
    first_step: bool,
}

impl MetroidReward {
    const RAM_HEALTH: usize = 0x0106;
    const RAM_HEALTH_TENS: usize = 0x0107;
    const RAM_MISSILES: usize = 0x0109;
    const RAM_ROOM: usize = 0x0050;
    const RAM_AREA: usize = 0x0074;
    const RAM_ITEM_FLAGS: usize = 0x006A;

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
            _energy_tank_bonus: energy_tank_bonus,
            new_item_bonus,
            _boss_damage_weight: boss_damage_weight,
            _boss_killed_bonus: boss_killed_bonus,
            health_delta_weight,
            death_penalty,
            time_penalty,
            visited: HashSet::new(),
            prev_missiles: 0,
            prev_health_total: 0,
            prev_item_flags: 0,
            died: false,
            first_step: true,
        }
    }
    pub fn reset(&mut self) {
        self.visited.clear();
        self.prev_missiles = 0;
        self.prev_health_total = 0;
        self.prev_item_flags = 0;
        self.died = false;
        self.first_step = true;
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
        let item_flags = ram[Self::RAM_ITEM_FLAGS];
        let health_total = Self::health_total(ram);

        if self.first_step {
            self.prev_missiles = missiles;
            self.prev_health_total = health_total;
            self.prev_item_flags = item_flags;
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

        let new_bits = (item_flags & !self.prev_item_flags).count_ones() as f64;
        if new_bits > 0.0 {
            acc.add("new_item", self.new_item_bonus * new_bits);
        }
        self.prev_item_flags = item_flags;

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
        self.prev_item_flags.count_ones() >= 8 && !self.died
    }
}

// ============================================================
// Factory — reads a game_profile-like weights dict.
// ============================================================

/// Convenience: weight lookup with default fallback.
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
        )));
    }
    if name.contains("contra") {
        return Some(Reward::Contra(ContraReward::new(
            w(weights, "forward_progress", 0.5),
            w(weights, "score_delta", 0.01),
            w(weights, "weapon_upgrade", 5.0),
            w(weights, "death_penalty", -25.0),
            w(weights, "time_penalty", -0.005),
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
            w(weights, "energy_tank", 20.0),
            w(weights, "new_item", 30.0),
            w(weights, "boss_damage", 2.0),
            w(weights, "boss_killed", 60.0),
            w(weights, "health_delta", -1.0),
            w(weights, "death_penalty", -25.0),
            w(weights, "time_penalty", -0.002),
        )));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    fn zram() -> Vec<u8> {
        vec![0u8; 2048]
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
