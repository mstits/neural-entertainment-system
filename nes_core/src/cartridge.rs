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

        // Read the remainder of the 16-byte header (bytes 11-15) so the
        // archaic-header heuristic below can inspect bytes 12-15.
        let mut header_tail = [0u8; 5];
        r.read_exact(&mut header_tail)?;

        // Archaic-iNES heuristic: pre-1.0 dumps (notably "DiskDude!"-
        // tagged ones) left ASCII garbage in header bytes 7-15. When
        // this is NOT a NES 2.0 header and header bytes 12-15 are
        // non-zero, the upper mapper nibble (flags7[7..4]) and flags8
        // cannot be trusted — flags7 might be 'D' (0x44), which would
        // otherwise yield mapper 0x40 and a multi-hundred-KB bogus
        // PRG-RAM. Treat as archaic: mapper from flags6 only, default
        // PRG-RAM. (header_tail[1..5] == header bytes 12-15.)
        let is_archaic = !is_nes20 && header_tail[1..5].iter().any(|&b| b != 0);

        let (mapper, sub_mapper, prg_rom_num_banks, chr_num_banks, prg_rom_size, chr_rom_size) =
            if is_nes20 {
                // NES 2.0 mapper = flags8[3..0] << 8 | flags7[7..4] | flags6[7..4]
                let mapper = ((flags8 as u16 & 0x0F) << 8)
                    | ((flags7 as u16 & 0xF0))
                    | ((flags6 as u16 & 0xF0) >> 4);
                let sub_mapper = flags8 >> 4;
                // PRG high-nibble in flags9[3..0], CHR high-nibble in flags9[7..4].
                let prg_hi = flags9 & 0x0F;
                let chr_hi = flags9 >> 4;

                // NES 2.0 ROM-size decode. Normal case: the full 12-bit
                // bank count (low byte | high nibble << 8) times the bank
                // size — no u8 clamp, so ROMs with > 255 banks parse to
                // their true size. Exponent-multiplier case (high nibble
                // == 0xF): the low byte is `EEEEEEMM`, and the size in
                // bytes is 2^E * (M*2 + 1) (not a bank multiple).
                // Hard sanity cap: the largest real NES ROM is a few MB. This
                // bounds the exponent-multiplier form BEFORE the payload vec is
                // allocated — otherwise a crafted/corrupt 16-byte header with a
                // large exponent (e.g. 2^60) computes a valid (non-overflowing)
                // usize, and vec![0u8; that] aborts the whole process with an
                // uncatchable OOM SIGABRT (one bad dump kills a library scan).
                const MAX_ROM_BYTES: usize = 64 * 1024 * 1024;
                let nes20_size =
                    |lo: u8, hi: u8, bank_size: usize| -> Result<usize, LoadError> {
                        let size = if hi == 0x0F {
                            let exponent = (lo >> 2) as u32;
                            let multiplier = (lo & 0x03) as usize;
                            (1usize)
                                .checked_shl(exponent)
                                .and_then(|base| base.checked_mul(multiplier * 2 + 1))
                                .ok_or_else(|| {
                                    LoadError::FormatError(
                                        "NES 2.0 exponent-multiplier ROM size overflows".into(),
                                    )
                                })?
                        } else {
                            let banks = lo as usize | ((hi as usize) << 8);
                            banks * bank_size
                        };
                        if size > MAX_ROM_BYTES {
                            return Err(LoadError::FormatError(
                                "NES 2.0 ROM size exceeds the 64 MB sanity cap".into(),
                            ));
                        }
                        Ok(size)
                    };
                let prg_rom_size =
                    nes20_size(prg_rom_num_banks_lo, prg_hi, PRG_ROM_BANK_SIZE as usize)?;
                let chr_rom_size =
                    nes20_size(chr_num_banks_lo, chr_hi, CHR_ROM_BANK_SIZE as usize)?;

                // The u8 `*_num_banks` fields feed mapper bank arithmetic.
                // Derive them from the true byte size and cap at u8::MAX;
                // mappers that bank-switch > 255 banks are unsupported,
                // but the ROM payload itself is loaded at full size below.
                let prg_u8 = (prg_rom_size / PRG_ROM_BANK_SIZE as usize).min(0xFF) as u8;
                let chr_u8 = (chr_rom_size / CHR_ROM_BANK_SIZE as usize).min(0xFF) as u8;
                (mapper, sub_mapper, prg_u8, chr_u8, prg_rom_size, chr_rom_size)
            } else {
                let mapper = if is_archaic {
                    (flags6 >> 4) as u16
                } else {
                    ((flags7 & 0xf0) | (flags6 >> 4)) as u16
                };
                let prg_rom_size =
                    prg_rom_num_banks_lo as usize * PRG_ROM_BANK_SIZE as usize;
                let chr_rom_size = chr_num_banks_lo as usize * CHR_ROM_BANK_SIZE as usize;
                (
                    mapper,
                    0u8,
                    prg_rom_num_banks_lo,
                    chr_num_banks_lo,
                    prg_rom_size,
                    chr_rom_size,
                )
            };

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
        } else if is_archaic {
            // flags8 is ASCII garbage on archaic dumps; use the default 8 KB.
            PRG_RAM_BANK_SIZE as usize
        } else {
            max(1, flags8) as usize * PRG_RAM_BANK_SIZE as usize
        };

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Assemble a 16-byte header (magic + `fields` for bytes 4..16) plus
    // `prg_len` PRG bytes and `chr_len` CHR bytes of zero payload.
    fn build_rom(fields: [u8; 12], prg_len: usize, chr_len: usize) -> Vec<u8> {
        let mut rom = vec![0x4E, 0x45, 0x53, 0x1A];
        rom.extend_from_slice(&fields);
        rom.resize(16 + prg_len + chr_len, 0);
        rom
    }

    // BUG 1: NES 2.0 PRG bank count > 255 must parse to the full size,
    // not be silently clamped to 255 banks.
    #[test]
    fn nes20_prg_over_255_banks_parses_full_size() {
        // flags7 = 0x08 marks NES 2.0. flags9 low nibble = 1 => PRG high
        // nibble, so bank count = 0x00 | (1 << 8) = 256 banks.
        let fields = [
            0x00, // byte 4: prg_lo
            0x00, // byte 5: chr_lo (CHR-RAM)
            0x00, // byte 6: flags6
            0x08, // byte 7: flags7 (NES 2.0)
            0x00, // byte 8: flags8
            0x01, // byte 9: flags9 (prg high nibble = 1)
            0x00, // byte 10: flags10
            0x00, 0x00, 0x00, 0x00, 0x00, // bytes 11-15
        ];
        let expected = 256 * PRG_ROM_BANK_SIZE as usize;
        let rom = build_rom(fields, expected, 0);
        let cart = Cartridge::load(&mut Cursor::new(rom)).unwrap();
        assert!(cart.is_nes20);
        assert_eq!(cart.prg_rom.len(), expected);
    }

    // BUG 1: NES 2.0 exponent-multiplier size notation (high nibble 0xF)
    // must use size = 2^E * (M*2 + 1), not the 12-bit-count formula.
    #[test]
    fn nes20_exponent_multiplier_prg_size() {
        // prg_lo = 0x39 => E = 0x39 >> 2 = 14, M = 0x39 & 3 = 1.
        // size = 2^14 * (1*2 + 1) = 16384 * 3 = 49152 bytes.
        let fields = [
            0x39, // byte 4: prg_lo (EEEEEEMM)
            0x00, // byte 5: chr_lo
            0x00, // byte 6: flags6
            0x08, // byte 7: flags7 (NES 2.0)
            0x00, // byte 8: flags8
            0x0F, // byte 9: flags9 (prg high nibble = 0xF => exponent form)
            0x00, // byte 10: flags10
            0x00, 0x00, 0x00, 0x00, 0x00, // bytes 11-15
        ];
        let expected = (1usize << 14) * (1 * 2 + 1);
        assert_eq!(expected, 49152);
        let rom = build_rom(fields, expected, 0);
        let cart = Cartridge::load(&mut Cursor::new(rom)).unwrap();
        assert_eq!(cart.prg_rom.len(), expected);
    }

    // BUG 2: an archaic "DiskDude!"-tagged iNES 1.0 header must ignore
    // the garbage upper-mapper nibble (flags7) and flags8, yielding the
    // low-nibble mapper and the default 8 KB PRG-RAM.
    #[test]
    fn archaic_diskdude_header_ignores_garbage_fields() {
        // Bytes 7-15 spell "DiskDude!"; byte 7 ('D' = 0x44) would push
        // mapper to 0x40 and byte 8 ('i' = 0x69) to a 840 KB PRG-RAM if
        // trusted. The real mapper here is 0 (flags6 upper nibble = 0).
        let fields = [
            0x01, // byte 4: prg_lo (1 PRG bank)
            0x01, // byte 5: chr_lo (1 CHR bank)
            0x00, // byte 6: flags6 (mapper low nibble = 0)
            0x44, // byte 7: 'D'
            0x69, // byte 8: 'i'
            0x73, // byte 9: 's'
            0x6B, // byte 10: 'k'
            0x44, // byte 11: 'D'
            0x75, // byte 12: 'u'
            0x64, // byte 13: 'd'
            0x65, // byte 14: 'e'
            0x21, // byte 15: '!'
        ];
        let rom = build_rom(
            fields,
            PRG_ROM_BANK_SIZE as usize,
            CHR_ROM_BANK_SIZE as usize,
        );
        let cart = Cartridge::load(&mut Cursor::new(rom)).unwrap();
        assert!(!cart.is_nes20);
        assert_eq!(cart.mapper, 0);
        assert_eq!(cart.prg_ram.len(), PRG_RAM_BANK_SIZE as usize);
    }
}
