use crate::memory::Memory;

use serde_derive::{Deserialize, Serialize};

// Controller button bitmask layout — nes-py convention, the layout
// every Python caller already uses (`src/emulation/frame_utils.py`,
// the trainer action table, legacy `.state.bin` recordings). Bit
// order reads "right→A" high-to-low. Historical pre-alignment:
// nes_core used the reversed layout (A=bit0, right=bit7), which
// silently swapped every button on the trainer → pool path; fixed
// 2026-04-21 in the audit sweep.
//
// These live here (rather than privately in `python.rs` and
// `pool.rs`, which each kept an identical copy) so the controller-1
// and controller-2 decode paths physically cannot diverge.
pub const BUTTON_RIGHT: u8 = 1 << 7;
pub const BUTTON_LEFT: u8 = 1 << 6;
pub const BUTTON_DOWN: u8 = 1 << 5;
pub const BUTTON_UP: u8 = 1 << 4;
pub const BUTTON_START: u8 = 1 << 3;
pub const BUTTON_SELECT: u8 = 1 << 2;
pub const BUTTON_B: u8 = 1 << 1;
pub const BUTTON_A: u8 = 1 << 0;

#[derive(Copy, Clone, Default, Deserialize, Serialize)]
pub enum Button {
    #[default]
    A,
    B,
    Select,
    Start,
    Up,
    Down,
    Left,
    Right,
}

#[derive(Copy, Clone, Default, Deserialize, Serialize)]
pub struct GamePad {
    a_pressed: bool,
    b_pressed: bool,
    select_pressed: bool,
    start_pressed: bool,
    up_pressed: bool,
    down_pressed: bool,
    left_pressed: bool,
    right_pressed: bool,

    strobe_state: StrobeState,
}

impl GamePad {
    pub fn set_button_pressed(&mut self, button: Button, pressed: bool) {
        match button {
            Button::A => self.a_pressed = pressed,
            Button::B => self.b_pressed = pressed,
            Button::Select => self.select_pressed = pressed,
            Button::Start => self.start_pressed = pressed,
            Button::Up => self.up_pressed = pressed,
            Button::Down => self.down_pressed = pressed,
            Button::Left => self.left_pressed = pressed,
            Button::Right => self.right_pressed = pressed,
        }
    }

    pub fn button_pressed(&self, button: Button) -> bool {
        match button {
            Button::A => self.a_pressed,
            Button::B => self.b_pressed,
            Button::Select => self.select_pressed,
            Button::Start => self.start_pressed,
            Button::Up => self.up_pressed,
            Button::Down => self.down_pressed,
            Button::Left => self.left_pressed,
            Button::Right => self.right_pressed,
        }
    }

    /// Apply a whole nes-py-layout button bitmask in one call.
    ///
    /// Every one of the eight buttons is written (pressed AND
    /// released), so the pad's state afterwards is a pure function of
    /// `mask` — exactly what the eight open-coded
    /// `set_button_pressed` calls in `python.rs` / `pool.rs`
    /// `apply_buttons` did, in the same order. Strobe state is left
    /// alone: latching is the $4016 write's job, not the mask's.
    pub fn set_mask(&mut self, mask: u8) {
        self.set_button_pressed(Button::Right, mask & BUTTON_RIGHT != 0);
        self.set_button_pressed(Button::Left, mask & BUTTON_LEFT != 0);
        self.set_button_pressed(Button::Down, mask & BUTTON_DOWN != 0);
        self.set_button_pressed(Button::Up, mask & BUTTON_UP != 0);
        self.set_button_pressed(Button::Start, mask & BUTTON_START != 0);
        self.set_button_pressed(Button::Select, mask & BUTTON_SELECT != 0);
        self.set_button_pressed(Button::B, mask & BUTTON_B != 0);
        self.set_button_pressed(Button::A, mask & BUTTON_A != 0);
    }

    fn next_button_state(&mut self) -> bool {
        let state = self.button_pressed(self.strobe_state.button);
        self.strobe_state.next();
        state
    }

    /// Side-effect-free view of the bit `next_button_state` would return
    /// next: reads the button the shift register currently points at
    /// without advancing it. Used by the bus `peek_byte` path.
    fn peek_button_state(&self) -> bool {
        self.button_pressed(self.strobe_state.button)
    }
}

#[derive(Copy, Clone, Default, Deserialize, Serialize)]
struct StrobeState {
    button: Button,
}

impl StrobeState {
    fn next(&mut self) {
        self.button = match self.button {
            Button::A => Button::B,
            Button::B => Button::Select,
            Button::Select => Button::Start,
            Button::Start => Button::Up,
            Button::Up => Button::Down,
            Button::Down => Button::Left,
            Button::Left => Button::Right,
            Button::Right => Button::A,
        };
    }

    fn reset(&mut self) {
        self.button = Button::default();
    }
}

#[derive(Default)]
pub struct Input {
    pub game_pad_1: GamePad,
    pub game_pad_2: GamePad,
    /// Last value written to $4016 bit 0. Real hardware latches the
    /// shift register on the high→low strobe transition. Resetting on
    /// every write would re-latch button A on any 0-write that follows
    /// another 0-write. NOT serialized — strobe is transient (resets
    /// to 0 on apply_state), and adding it to the State struct would
    /// break bincode compatibility with already-saved state.bin files.
    last_strobe: u8,
}

#[derive(Copy, Clone, Deserialize, Serialize)]
pub struct State {
    pub game_pad_1: GamePad,
    pub game_pad_2: GamePad,
}

impl Input {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get_state(&self) -> State {
        State {
            game_pad_1: self.game_pad_1,
            game_pad_2: self.game_pad_2,
        }
    }

    pub fn apply_state(&mut self, state: &State) {
        self.game_pad_1 = state.game_pad_1;
        self.game_pad_2 = state.game_pad_2;
        // Reset to "no recent strobe" on state load so the next $4016
        // write triggers the latch path normally. If the loaded state
        // happened to be mid-strobe, this loses 1-2 cycles of state,
        // which is an acceptable rounding given the alternative is
        // breaking every saved state.bin from before this field
        // existed.
        self.last_strobe = 0;
    }

    fn reset_strobe_states(&mut self) {
        self.game_pad_1.strobe_state.reset();
        self.game_pad_2.strobe_state.reset();
    }

    /// Side-effect-free counterpart to the `Memory::read_byte` controller
    /// path: returns the bit the next $4016/$4017 read would yield without
    /// advancing the shift register. Any other address returns 0, matching
    /// `read_byte`. Used by the bus `peek_byte` path.
    pub fn peek_byte(&self, address: u16) -> u8 {
        if address == 0x4016 {
            self.game_pad_1.peek_button_state() as u8
        } else if address == 0x4017 {
            self.game_pad_2.peek_button_state() as u8
        } else {
            0
        }
    }
}

impl Memory for Input {
    fn read_byte(&mut self, address: u16) -> u8 {
        if address == 0x4016 {
            self.game_pad_1.next_button_state() as u8
        } else if address == 0x4017 {
            self.game_pad_2.next_button_state() as u8
        } else {
            0
        }
    }

    fn write_byte(&mut self, address: u16, value: u8) {
        if address == 0x4016 {
            let new_strobe = value & 1;
            // Latch the shift register at the strobe-falling edge
            // (1→0). While strobe is high, real hardware reloads
            // continuously — we model this by also resetting on
            // strobe=1 so the immediate next read returns A. The
            // bug fixed here is the spurious reset on a 0-strobe
            // write that would have re-latched A even though the
            // shift register was supposed to be advancing.
            if new_strobe == 1 || self.last_strobe == 1 {
                self.reset_strobe_states();
            }
            self.last_strobe = new_strobe;
        }
    }
}

/// Controller-2 lane + shared mask decode.
///
/// Controller 2 has always existed in `Input` (the $4017 read path
/// and the serialized `State` both carry it) but nothing above the
/// bus ever wrote to it — every driver applied its action byte to
/// pad 1 only. These tests pin the two properties the 2P lane rests
/// on: `set_mask` is a pure function of the mask (it clears as well
/// as sets), and the two pads are fully independent, so a pool that
/// never touches pad 2 keeps byte-identical $4016 behaviour.
#[cfg(test)]
mod two_player_tests {
    use super::*;

    /// Longhand reference decode — the exact eight-call sequence
    /// `apply_buttons` used before `set_mask` existed. Any divergence
    /// between this and `set_mask` is a behaviour change on the
    /// controller-1 hot path.
    fn reference_apply(pad: &mut GamePad, mask: u8) {
        pad.set_button_pressed(Button::Right, mask & BUTTON_RIGHT != 0);
        pad.set_button_pressed(Button::Left, mask & BUTTON_LEFT != 0);
        pad.set_button_pressed(Button::Down, mask & BUTTON_DOWN != 0);
        pad.set_button_pressed(Button::Up, mask & BUTTON_UP != 0);
        pad.set_button_pressed(Button::Start, mask & BUTTON_START != 0);
        pad.set_button_pressed(Button::Select, mask & BUTTON_SELECT != 0);
        pad.set_button_pressed(Button::B, mask & BUTTON_B != 0);
        pad.set_button_pressed(Button::A, mask & BUTTON_A != 0);
    }

    const ALL_BUTTONS: [Button; 8] = [
        Button::A,
        Button::B,
        Button::Select,
        Button::Start,
        Button::Up,
        Button::Down,
        Button::Left,
        Button::Right,
    ];

    fn pressed_vec(pad: &GamePad) -> Vec<bool> {
        ALL_BUTTONS.iter().map(|b| pad.button_pressed(*b)).collect()
    }

    /// Latch ($4016 strobe 1→0) then shift out `n` bits from `port`.
    fn poll(input: &mut Input, port: u16, n: usize) -> Vec<u8> {
        input.write_byte(0x4016, 1);
        input.write_byte(0x4016, 0);
        (0..n).map(|_| input.read_byte(port)).collect()
    }

    /// `set_mask` must agree with the longhand decode for all 256
    /// masks, from a dirty starting state (so the "clears too" half
    /// is actually exercised — a set-only implementation passes the
    /// from-default version of this test).
    #[test]
    fn set_mask_matches_longhand_decode_for_every_mask() {
        for prior in [0x00u8, 0xFF, 0xA5, 0x5A] {
            for mask in 0..=255u8 {
                let mut a = GamePad::default();
                let mut b = GamePad::default();
                reference_apply(&mut a, prior);
                reference_apply(&mut b, prior);

                a.set_mask(mask);
                reference_apply(&mut b, mask);

                assert_eq!(
                    pressed_vec(&a),
                    pressed_vec(&b),
                    "set_mask({mask:#04x}) from prior {prior:#04x} diverged from \
                     the longhand eight-call decode",
                );
            }
        }
    }

    /// Mutation guard for the above: a deliberately set-only decode
    /// (never releases) must FAIL the comparison, proving the test
    /// can see a broken `set_mask`.
    #[test]
    fn set_only_decode_is_detected_as_wrong() {
        let mut broken = GamePad::default();
        let mut correct = GamePad::default();
        reference_apply(&mut broken, 0xFF);
        reference_apply(&mut correct, 0xFF);

        // "set-only" decode of mask 0x00: nothing to press, and it
        // never releases — so the pad stays fully held.
        let mask = 0x00u8;
        for (button, bit) in ALL_BUTTONS.iter().zip([
            BUTTON_A, BUTTON_B, BUTTON_SELECT, BUTTON_START, BUTTON_UP, BUTTON_DOWN,
            BUTTON_LEFT, BUTTON_RIGHT,
        ]) {
            if mask & bit != 0 {
                broken.set_button_pressed(*button, true);
            }
        }
        correct.set_mask(mask);

        assert_ne!(
            pressed_vec(&broken),
            pressed_vec(&correct),
            "a set-only decode must be distinguishable from set_mask",
        );
    }

    /// Pad 2 defaults to fully released, and $4017 reads all-zero
    /// until someone writes to it. This is the "default-inert"
    /// property the whole 2P feature rests on: an existing caller
    /// that never touches pad 2 sees exactly what it saw before.
    #[test]
    fn pad2_defaults_to_released() {
        let mut input = Input::new();
        assert_eq!(poll(&mut input, 0x4017, 8), vec![0u8; 8]);
    }

    /// Driving pad 1 must leave pad 2 untouched and vice versa. A
    /// shared-pad regression (e.g. `game_pad_2()` accidentally
    /// returning pad 1) shows up here immediately.
    #[test]
    fn pads_are_independent() {
        let mut input = Input::new();

        input.game_pad_1.set_mask(0xFF);
        assert_eq!(poll(&mut input, 0x4016, 8), vec![1u8; 8], "pad 1 all pressed");
        assert_eq!(poll(&mut input, 0x4017, 8), vec![0u8; 8], "pad 2 must stay clear");

        input.game_pad_1.set_mask(0x00);
        input.game_pad_2.set_mask(0xFF);
        assert_eq!(poll(&mut input, 0x4016, 8), vec![0u8; 8], "pad 1 must be clear");
        assert_eq!(poll(&mut input, 0x4017, 8), vec![1u8; 8], "pad 2 all pressed");
    }

    /// $4017 shifts pad 2 out in the hardware order A, B, Select,
    /// Start, Up, Down, Left, Right — the same order $4016 uses for
    /// pad 1. Pins the mask→wire mapping so a P2 action byte means
    /// the same thing as a P1 action byte.
    #[test]
    fn pad2_shifts_out_in_hardware_button_order() {
        let cases: [(u8, [u8; 8]); 8] = [
            (BUTTON_A, [1, 0, 0, 0, 0, 0, 0, 0]),
            (BUTTON_B, [0, 1, 0, 0, 0, 0, 0, 0]),
            (BUTTON_SELECT, [0, 0, 1, 0, 0, 0, 0, 0]),
            (BUTTON_START, [0, 0, 0, 1, 0, 0, 0, 0]),
            (BUTTON_UP, [0, 0, 0, 0, 1, 0, 0, 0]),
            (BUTTON_DOWN, [0, 0, 0, 0, 0, 1, 0, 0]),
            (BUTTON_LEFT, [0, 0, 0, 0, 0, 0, 1, 0]),
            (BUTTON_RIGHT, [0, 0, 0, 0, 0, 0, 0, 1]),
        ];
        for (mask, expected) in cases {
            let mut input = Input::new();
            input.game_pad_2.set_mask(mask);
            assert_eq!(
                poll(&mut input, 0x4017, 8),
                expected.to_vec(),
                "pad 2 mask {mask:#04x} shifted out in the wrong bit order",
            );
        }
    }

    /// Pad 2 state survives a get/apply_state round trip — a solver
    /// that snapshots mid-2P-sequence must restore the same held
    /// buttons on both controllers.
    #[test]
    fn pad2_survives_state_round_trip() {
        let mut input = Input::new();
        input.game_pad_1.set_mask(BUTTON_RIGHT);
        input.game_pad_2.set_mask(BUTTON_A | BUTTON_START);
        let saved = input.get_state();

        let mut restored = Input::new();
        restored.apply_state(&saved);

        assert_eq!(pressed_vec(&restored.game_pad_1), pressed_vec(&input.game_pad_1));
        assert_eq!(pressed_vec(&restored.game_pad_2), pressed_vec(&input.game_pad_2));
    }
}
