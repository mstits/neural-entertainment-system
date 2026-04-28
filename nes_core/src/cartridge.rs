use byteorder::{BigEndian, ReadBytesExt};
use serde_derive::{Deserialize, Serialize};
use thiserror::Error;

use std::cmp::max;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::io;
use std::io::Read;

// ROM must begin with this constant ("NES" followed by MS-DOS end-of-file)
const MAGIC_CONSTANT: u32 = 0x4e45_531a;

pub const PRG_ROM_BANK_SIZE: u16 = 16 * 1024;
pub const CHR_ROM_BANK_SIZE: u16 = 8 * 1024;
pub const PRG_RAM_BANK_SIZE: u16 = 8 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum Mirroring {
    Horizontal,
    Vertical,
    OneScreenLower,
    OneScreenUpper,
    FourScreen,
}

impl Mirroring {
    pub fn mirror_address(self, address: u16) -> u16 {
        // Fold the $3000-$3EFF range into $2000-$2EFF.
        let address = (address - 0x2000) & 0x0FFF;

        // Determine which 1KB page of the nametable we are in.
        let page = address / 0x0400;
        let offset = address % 0x0400;

        match self {
            Mirroring::Horizontal => (page / 2) * 0x0400 + offset,
            Mirroring::Vertical => (page % 2) * 0x0400 + offset,
            Mirroring::OneScreenLower => offset,
            Mirroring::OneScreenUpper => 0x0400 + offset,
            Mirroring::FourScreen => address,
        }
    }
}

#[derive(Error, Debug)]
pub enum LoadError {
    #[error("{0}")]
    FormatError(String),
    #[error("{0}")]
    IoError(#[from] io::Error),
}

pub struct Cartridge {
    pub mapper: u16,
    pub sub_mapper: u8,
    pub mirroring: Mirroring,
    pub default_mirroring: Mirroring,
    pub prg_rom_num_banks: u8,
    pub prg_rom: Vec<u8>,
    pub chr_num_banks: u8,
    pub chr: Vec<u8>,
    pub prg_ram: Vec<u8>,
    pub is_battery_backed: bool,
    /// True when the header declared NES 2.0 (flags7 bits 2-3 == 0b10).
    /// Consumers that care about extended fields (sub_mapper, larger
    /// RAM, PlayChoice-10, etc.) check this; iNES 1.0 legacy dumps
    /// keep working via the `false` branch.
    pub is_nes20: bool,
    /// MD5 of the ROM payload (PRG + CHR, excluding the 16-byte header).
    /// Profiles can assert this to catch dirty dumps that would
    /// silently shift RAM addresses and break reward functions.
    /// Represented as the lowercase-hex string of the 16-byte digest.
    pub md5: String,
}

#[derive(Deserialize, Serialize)]
pub struct State {
    pub mirroring: Mirroring,
    #[serde(with = "serde_bytes")]
    pub chr: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub prg_ram: Vec<u8>,
}

impl Debug for Cartridge {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "mapper: {}", self.mapper)?;
        writeln!(f, "sub mapper: {}", self.sub_mapper)?;
        writeln!(f, "mirroring: {:?}", self.mirroring)?;
        writeln!(f, "PRG ROM size: {}", self.prg_rom.len())?;
        writeln!(f, "CHR ROM size: {}", self.chr.len())?;
        writeln!(f, "PRG RAM size: {}", self.prg_ram.len())?;
        writeln!(f, "battery backed: {}", self.is_battery_backed)
    }
}

impl Cartridge {
    pub fn load<R: Read>(r: &mut R) -> Result<Cartridge, LoadError> {
        let magic = r.read_u32::<BigEndian>()?;

        if magic != MAGIC_CONSTANT {
            return Err(LoadError::FormatError(
                "magic constant in header is incorrect".into(),
            ));
        }

        let prg_rom_num_banks_lo = r.read_u8()?;
        let chr_num_banks_lo = r.read_u8()?;

        let flags6 = r.read_u8()?;
        let flags7 = r.read_u8()?;

        // NES 2.0 detection: flags7 bits 2-3 == 0b10. When set, the
        // mapper number extends to 12 bits, sub-mapper appears, and
        // PRG/CHR sizes get high nibbles from flags9.
        let is_nes20 = (flags7 & 0x0C) == 0x08;

        let flags8 = r.read_u8()?;
        let flags9 = r.read_u8()?;
        let flags10 = r.read_u8()?;

        let (mapper, sub_mapper, prg_rom_num_banks, chr_num_banks) = if is_nes20 {
            // NES 2.0 mapper = flags8[3..0] << 8 | flags7[7..4] | flags6[7..4]
            let mapper = ((flags8 as u16 & 0x0F) << 8)
                | ((flags7 as u16 & 0xF0))
                | ((flags6 as u16 & 0xF0) >> 4);
            let sub_mapper = flags8 >> 4;
            // PRG high-nibble in flags9[3..0], CHR high-nibble in flags9[7..4].
            let prg_hi = flags9 & 0x0F;
            let chr_hi = flags9 >> 4;
            let prg_banks = prg_rom_num_banks_lo as u16 | ((prg_hi as u16) << 8);
            let chr_banks = chr_num_banks_lo as u16 | ((chr_hi as u16) << 8);
            // Clamp to u8 for our storage type; ROMs bigger than 4MB
            // PRG or 2MB CHR are exotic and unsupported here.
            let prg_u8 = prg_banks.min(0xFF) as u8;
            let chr_u8 = chr_banks.min(0xFF) as u8;
            (mapper, sub_mapper, prg_u8, chr_u8)
        } else {
            let mapper = ((flags7 & 0xf0) | (flags6 >> 4)) as u16;
            (mapper, 0u8, prg_rom_num_banks_lo, chr_num_banks_lo)
        };

        let prg_rom_size = prg_rom_num_banks as usize * PRG_ROM_BANK_SIZE as usize;
        let chr_rom_size = chr_num_banks as usize * CHR_ROM_BANK_SIZE as usize;

        // PRG-RAM sizing.
        //
        // iNES 1.0 (`flags8`): PRG-RAM size in 8 KB banks; the
        // legacy zero-means-8 KB fallback is preserved via `max(1, …)`.
        //
        // NES 2.0 (`flags10`) packs TWO nibbles, both interpreted as a
        // shift count S where the actual size is `64 << S` bytes:
        //   bits 0-3 (`A`): volatile PRG-RAM
        //   bits 4-7 (`B`): non-volatile (battery / EEPROM) PRG-RAM
        // S=0 means "no RAM of that kind". Total PRG-RAM is the sum;
        // both regions live in the same `$6000-$7FFF` window and the
        // mapper picks which one it sees. We previously read only the
        // lower nibble — which silently dropped the battery size on
        // certain Zelda dumps (`flags10 = 0x70`: A=0, B=7 → 8 KB
        // battery). With no PRG-RAM allocated, MMC1 reads at $6000
        // returned open-bus, the game's title-screen scratch
        // ($6000-$60FF) read garbage, and the eventual indirect-JMP
        // landed in non-code → BRK → IRQ trap at $FFF0.
        let prg_ram_size = if is_nes20 {
            let volatile_shift = flags10 & 0x0F;
            let battery_shift = (flags10 >> 4) & 0x0F;
            let shift_to_bytes = |s: u8| -> usize {
                if s == 0 { 0 } else { 64usize << s as usize }
            };
            shift_to_bytes(volatile_shift) + shift_to_bytes(battery_shift)
        } else {
            max(1, flags8) as usize * PRG_RAM_BANK_SIZE as usize
        };

        // Skip the rest of the 16-byte header (flags11..15).
        for _ in 0..5 {
            r.read_u8()?;
        }

        let is_battery_backed = (flags6 & 0x02) != 0;

        let has_trainer = (flags6 & 0x04) != 0;
        if has_trainer {
            for _ in 0..512 {
                r.read_u8()?;
            }
        }

        let mirroring = if (flags6 & 0x08) != 0 {
            Mirroring::FourScreen
        } else if (flags6 & 0x01) == 1 {
            Mirroring::Vertical
        } else {
            Mirroring::Horizontal
        };

        let mut prg_rom = vec![0u8; prg_rom_size];
        r.read_exact(&mut prg_rom[..])?;

        let mut chr = vec![0u8; chr_rom_size];
        r.read_exact(&mut chr[..])?;

        // Compute MD5 over the PRG+CHR payload — the bytes a dirty
        // dump would differ in. Header is excluded so re-headered
        // but byte-identical ROMs still match. Hex string for easy
        // comparison with profile YAML assertions.
        let md5 = {
            let mut ctx = md5::Context::new();
            ctx.consume(&prg_rom);
            ctx.consume(&chr);
            format!("{:x}", ctx.compute())
        };

        // Add CHR bank if not in file (CHR-RAM carts).
        let mut chr_num_banks = chr_num_banks;
        if chr_num_banks == 0 {
            chr_num_banks = 1;
            chr = vec![0u8; CHR_ROM_BANK_SIZE as usize];
        }

        let prg_ram = vec![0u8; prg_ram_size];

        Ok(Cartridge {
            mapper,
            sub_mapper,
            mirroring,
            default_mirroring: mirroring,
            prg_rom_num_banks,
            prg_rom,
            chr_num_banks,
            chr,
            prg_ram,
            is_battery_backed,
            is_nes20,
            md5,
        })
    }

    pub fn get_state(&self) -> State {
        State {
            mirroring: self.mirroring,
            chr: self.chr.clone(),
            prg_ram: self.prg_ram.clone(),
        }
    }

    pub fn apply_state(&mut self, state: &State) {
        self.mirroring = state.mirroring;
        self.chr = state.chr.clone();
        self.prg_ram = state.prg_ram.clone();
    }
}
