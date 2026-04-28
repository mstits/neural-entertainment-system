use serde_derive::{Deserialize, Serialize};

pub struct OamDma {
    pub active: bool,
    pub page: u8,
    pub count: u16,
    pub data: u8,
    pub dummy_read: bool,
}

/// Persistable snapshot of OAM DMA state. Serialized as part of
/// `nes::State` so `save_state` / `load_state` round-trips are
/// byte-exact even when the emulator was paused mid-transfer (514
/// CPU-cycle window). Previously unserialized → save mid-DMA + load
/// would land the OAM array in an inconsistent state.
#[derive(Deserialize, Serialize, Default)]
pub struct State {
    pub active: bool,
    pub page: u8,
    pub count: u16,
    pub data: u8,
    pub dummy_read: bool,
}

impl OamDma {
    pub fn new() -> Self {
        Self {
            active: false,
            page: 0,
            count: 0,
            data: 0,
            dummy_read: false,
        }
    }

    pub fn activate(&mut self, page: u8) {
        self.page = page;
        self.active = true;
        self.count = 0;
        self.dummy_read = true;
    }

    pub fn get_state(&self) -> State {
        State {
            active: self.active,
            page: self.page,
            count: self.count,
            data: self.data,
            dummy_read: self.dummy_read,
        }
    }

    pub fn apply_state(&mut self, s: &State) {
        self.active = s.active;
        self.page = s.page;
        self.count = s.count;
        self.data = s.data;
        self.dummy_read = s.dummy_read;
    }
}
