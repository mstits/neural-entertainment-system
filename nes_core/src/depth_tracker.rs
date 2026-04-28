//! Per-run depth tracker — pure logic, shared by Python via PyO3 wrapper.
//!
//! Extracts a 3-tuple "depth key" from a RAM snapshot using a
//! per-game reader, and records the furthest key any worker has
//! reached in this run. Returns a structured record when a new depth
//! is set, `None` otherwise.

/// 3-tuple depth key. Lexicographic comparison determines "deeper":
/// higher first component wins outright, ties broken by second, then
/// third. Matches the Python `(dungeon, mx, my)` / `(world, level, x)`
/// contract.
pub type DepthKey = (u32, u32, u32);

/// Classification of the game's RAM layout. Unrecognised games fall
/// back to `Generic` which reads the first two bytes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GameKind {
    Zelda,
    Mario,
    Generic,
}

impl GameKind {
    /// Case-insensitive substring match, same contract as the Python
    /// `_depth_reader_for` lookup.
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        if lower.contains("zelda") {
            Self::Zelda
        } else if lower.contains("mario") {
            Self::Mario
        } else {
            Self::Generic
        }
    }

    pub fn read(&self, ram: &[u8]) -> Option<DepthKey> {
        match self {
            Self::Zelda => {
                // Zelda reads at offsets $10, $EB, $EC. RAM is 2048B
                // ($0800). Bound is highest-offset + 1 = $00ED.
                // The old `< 0x10EC` typo (4332B > 2048B RAM) made
                // this branch unconditionally return None, silently
                // disabling Zelda auto-curriculum for the entire
                // session.
                if ram.len() < 0x00ED {
                    return None;
                }
                Some((u32::from(ram[0x10]), u32::from(ram[0xEB]), u32::from(ram[0xEC])))
            }
            Self::Mario => {
                if ram.len() < 0x0761 {
                    return None;
                }
                let world = u32::from(ram[0x075F]);
                let level = u32::from(ram[0x0760]);
                let x_hi = u32::from(ram[0x006D]);
                let x_lo = u32::from(ram[0x0086]);
                Some((world, level, x_hi * 256 + x_lo))
            }
            Self::Generic => {
                if ram.len() < 2 {
                    return None;
                }
                Some((0, u32::from(ram[0]), u32::from(ram[1])))
            }
        }
    }

    pub fn caption(&self, key: DepthKey) -> String {
        match self {
            Self::Zelda => {
                let (dungeon, mx, my) = key;
                if dungeon == 0 {
                    format!("furthest overworld tile yet ({:02x},{:02x})", mx, my)
                } else {
                    format!(
                        "deepest dungeon point yet — L{}, room ({:02x},{:02x})",
                        dungeon, mx, my
                    )
                }
            }
            Self::Mario => {
                let (world, level, x) = key;
                format!("deepest point yet — W{}-{}, x={}", world + 1, level + 1, x)
            }
            Self::Generic => format!("new depth key {:?}", key),
        }
    }
}

/// Stateful max-depth tracker. A fresh run starts with `best = None`;
/// every `observe()` call that reads a strictly greater key updates
/// the record and yields a `Record`.
pub struct DepthTracker {
    kind: GameKind,
    best: Option<DepthKey>,
}

#[derive(Clone, Debug)]
pub struct Record {
    pub generation: u32,
    pub worker_id: u32,
    pub genome_name: String,
    pub key: DepthKey,
    pub caption: String,
}

impl DepthTracker {
    pub fn new(game: &str) -> Self {
        Self {
            kind: GameKind::from_name(game),
            best: None,
        }
    }

    pub fn best(&self) -> Option<DepthKey> {
        self.best
    }

    pub fn observe(
        &mut self,
        ram: &[u8],
        worker_id: u32,
        genome_name: &str,
        generation: u32,
    ) -> Option<Record> {
        let key = self.kind.read(ram)?;
        let improved = match self.best {
            None => true,
            Some(prev) => key > prev,
        };
        if !improved {
            return None;
        }
        self.best = Some(key);
        Some(Record {
            generation,
            worker_id,
            genome_name: genome_name.to_string(),
            key,
            caption: self.kind.caption(key),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ram_of(size: usize, kv: &[(usize, u8)]) -> Vec<u8> {
        let mut r = vec![0u8; size];
        for (idx, v) in kv.iter() {
            r[*idx] = *v;
        }
        r
    }

    #[test]
    fn zelda_key_is_dungeon_mx_my() {
        let mut t = DepthTracker::new("The Legend of Zelda");
        let ram = ram_of(0x00ED, &[(0x10, 2), (0xEB, 4), (0xEC, 7)]);
        let rec = t.observe(&ram, 0, "alice", 0).unwrap();
        assert_eq!(rec.key, (2, 4, 7));
        assert!(rec.caption.contains("L2"));
    }

    #[test]
    fn mario_key_is_world_level_x16() {
        let mut t = DepthTracker::new("Super Mario Bros");
        let ram = ram_of(
            0x0800,
            &[(0x075F, 1), (0x0760, 2), (0x006D, 1), (0x0086, 100)],
        );
        let rec = t.observe(&ram, 0, "bob", 0).unwrap();
        assert_eq!(rec.key, (1, 2, 256 + 100));
    }

    #[test]
    fn strict_improvement_only() {
        let mut t = DepthTracker::new("zelda");
        let ram_a = ram_of(0x00ED, &[(0x10, 1), (0xEB, 2), (0xEC, 3)]);
        let ram_b = ram_of(0x00ED, &[(0x10, 1), (0xEB, 2), (0xEC, 3)]);
        let ram_c = ram_of(0x00ED, &[(0x10, 1), (0xEB, 2), (0xEC, 4)]);
        assert!(t.observe(&ram_a, 0, "x", 0).is_some());
        assert!(t.observe(&ram_b, 0, "x", 0).is_none());
        assert!(t.observe(&ram_c, 0, "x", 0).is_some());
    }

    #[test]
    fn short_ram_returns_none() {
        let mut t = DepthTracker::new("zelda");
        let ram = vec![0u8; 16];
        assert!(t.observe(&ram, 0, "x", 0).is_none());
    }
}
