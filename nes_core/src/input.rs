use crate::memory::Memory;

use serde_derive::{Deserialize, Serialize};

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

    fn next_button_state(&mut self) -> bool {
        let state = self.button_pressed(self.strobe_state.button);
        self.strobe_state.next();
        state
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
