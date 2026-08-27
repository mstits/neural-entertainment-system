"""Cross-reference every quarantined RAM address against the EXECUTING layer.

The config quarantine (`quarantined_external_knowledge:` blocks) retracts a
DOCUMENTATION claim. It does not touch the Rust constant or the Python
literal that actually runs. This module closes that gap: it derives the
quarantined address set from the config blocks themselves, derives which
code owns each ROM from the source, and reports every place a quarantined
address is still live in code.

Nothing here knows a single fact about any game. It reads the repository's
own records and matches them against the repository's own source.

Two scans, kept separate because they answer different questions:

  `scan_quarantined_uses()` — where is a quarantined address still LIVE?
  `witness_ledger()`        — which ROMs has this repo actually witnessed
                              a clear on, derived from solver tapes?
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"
RUST_DIR = REPO / "nes_core" / "src"

QUARANTINE_KEY = "quarantined_external_knowledge"

#: Directories that hold copies of the tree, build output, or third-party
#: code. Scanning them would double-count the very sites we are auditing.
EXCLUDED_DIR_PARTS = frozenset({
    ".venv", ".git", "target", "build", "node_modules", "__pycache__",
    ".claude", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist",
})


def _excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_PARTS for part in path.parts)


def _load_yaml(path: Path) -> dict:
    try:
        doc = yaml.safe_load(path.read_text())
    except Exception:
        return {}
    return doc if isinstance(doc, dict) else {}


# =====================================================================
# 1. The quarantine record — derived, never hand-listed
# =====================================================================

@dataclass(frozen=True)
class Quarantine:
    """One quarantined address, with everything needed to scope it."""
    config: str          # "configs/megaman.yaml"
    reward_id: str       # "mega_man" — the arm that owns this ROM's code
    rom: str             # "Mega Man 2 (USA).nes"
    key: str             # "q_boss_health"
    addr: int            # 0x06C1


def _addr_leaves(node) -> dict[str, int]:
    """Quarantined addresses are STRING values starting `0x` — that is the
    shape the quarantine convention uses so `int(v)` raises and the value
    cannot be folded into an instrument's exclusion set.

    Deliberately NOT a regex over prose: `configs/metroid.yaml`'s block
    explains in prose that `0x0107` is legitimately VERIFIED, and a
    "any 0x token in the block" scan would quarantine the opposite of
    what the block says.
    """
    out: dict[str, int] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and re.fullmatch(r"0x[0-9A-Fa-f]+", v.strip()):
                out[str(k)] = int(v.strip(), 16)
    return out


def quarantines(config_dir: Path | None = None,
                repo: Path | None = None) -> list[Quarantine]:
    """Every quarantined address in the tree, scoped to the reward arm that
    owns its ROM. Sourced from the blocks themselves so this cannot drift
    from the record it guards.

    `config_dir` / `repo` are injectable so the mutation tests can point the
    whole checker at a synthetic tree and prove it still catches things.
    """
    config_dir = config_dir if config_dir is not None else CONFIG_DIR
    repo = repo if repo is not None else REPO
    found: list[Quarantine] = []
    for path in sorted(config_dir.rglob("*.yaml")):
        doc = _load_yaml(path)
        block = doc.get(QUARANTINE_KEY)
        if not isinstance(block, dict):
            continue
        reward_id = str(doc.get("reward_id") or "").strip()
        rom = Path(str(block.get("applies_to_rom") or "")).name
        rel = str(path.relative_to(repo))
        for key, addr in _addr_leaves(block).items():
            found.append(Quarantine(rel, reward_id, rom, key, addr))
    return found


# =====================================================================
# 2. Rust ownership — derived from the dispatch table in the source
# =====================================================================

#: `Reward::MegaMan(_) => "mega_man",` in the `reward_id()` match. This is
#: the source's own variant -> id mapping; reading it means a renamed arm
#: cannot silently escape the scan.
_VARIANT_TO_ID = re.compile(r"Reward::(\w+)\s*\(\s*_\s*\)\s*=>\s*\"([a-z0-9_]+)\"")

#: A RAM address constant. `usize` is the address type throughout
#: `rewards.rs`; `u8` constants are VALUES (`SONG_ENDING: u8 = 0x10`,
#: `DIRECTIONAL_MASK: u8 = 0x80 | ...`) and are not addresses. Keying on
#: the type is what keeps Zelda's `SONG_ENDING = 0x10` from colliding with
#: its quarantined `q_dungeon_level = 0x10`.
_RUST_ADDR_CONST = re.compile(
    r"^\s*(?:pub\s+)?const\s+(\w+)\s*:\s*usize\s*=\s*(0[xX][0-9A-Fa-f]+|\d+)\s*;")
_RUST_ADDR_ARRAY = re.compile(
    r"^\s*(?:pub\s+)?const\s+(\w+)\s*:\s*\[\s*usize\s*;\s*\d+\s*\]\s*=\s*\[([^\]]*)\]")
#: A direct literal RAM subscript, e.g. `ram[0x0030]`.
_RUST_RAM_INDEX = re.compile(r"\bram\d?\s*\[\s*(0[xX][0-9A-Fa-f]+)\s*\]")


def _rust_variant_ids(text: str) -> dict[str, str]:
    """`{"MegaManReward": "mega_man", ...}` — the arm's struct name is the
    variant name plus `Reward`, which is the convention the file uses."""
    return {f"{m.group(1)}Reward": m.group(2)
            for m in _VARIANT_TO_ID.finditer(text)}


def _impl_regions(lines: list[str]) -> list[tuple[str, int, int]]:
    """`[(struct_name, start_line, end_line), ...]` for every `impl X {`.

    Brace-counted, so a nested block cannot end the region early. String
    and char literals containing braces do not occur in these impls; the
    counter would still be correct across balanced ones.
    """
    regions: list[tuple[str, int, int]] = []
    i = 0
    while i < len(lines):
        m = re.match(r"^impl(?:<[^>]*>)?\s+(\w+)\s*\{", lines[i])
        if not m:
            i += 1
            continue
        depth, j = 0, i
        while j < len(lines):
            depth += lines[j].count("{") - lines[j].count("}")
            if depth <= 0 and j > i:
                break
            j += 1
        regions.append((m.group(1), i, j))
        i = j + 1
    return regions


def _test_line_spans(lines: list[str]) -> list[tuple[int, int]]:
    """Line ranges of `#[cfg(test)] mod ... { }` blocks.

    Test fixtures write synthetic RAM (`ram[0x0001] = 1`) precisely to
    PROVE a predicate's behaviour, including proving it cannot fire. Those
    are guards, not claims, so they are reported under a separate kind and
    never counted as a live production use.
    """
    spans: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        if "#[cfg(test)]" not in line:
            continue
        j = i
        while j < len(lines) and not re.search(r"\bmod\s+\w+\s*\{", lines[j]):
            j += 1
        if j >= len(lines):
            continue
        depth, k = 0, j
        while k < len(lines):
            depth += lines[k].count("{") - lines[k].count("}")
            if depth <= 0 and k > j:
                break
            k += 1
        spans.append((i, k))
    return spans


# =====================================================================
# 3. The scan
# =====================================================================

#: Kinds whose sites are part of the EXECUTING layer — the code that runs
#: when a policy trains or a solver searches. A quarantined address here is
#: an unretracted claim in the thing that actually decides reward and
#: episode boundaries, which is exactly the gap a config quarantine leaves
#: open. These are ENFORCED: every one must be declared.
ENFORCED_KINDS = frozenset({"rust-const", "rust-index", "python",
                           "rust-unowned"})

#: Kinds that are REPORTED but not enforced. A test that writes
#: `ram[0x0672] = 1` and asserts the reward does NOT report a win is a
#: GUARD, not a claim — it is the mechanism holding the quarantine shut.
#: Failing on those would make the check fight its own guards, and a check
#: that fights its own guards gets deleted.
REPORTED_KINDS = frozenset({"rust-test", "python-test", "python-literal"})


@dataclass(frozen=True)
class Use:
    """One live use of a quarantined address in the executing layer."""
    file: str
    line: int
    symbol: str          # the constant name, or the raw subscript text
    addr: int
    reward_id: str
    quarantine: str      # "configs/megaman.yaml:q_boss_health"
    kind: str

    @property
    def site_id(self) -> str:
        """Stable identity of a site across edits that move it: file +
        symbol + address. Deliberately excludes the line number so an
        unrelated edit above does not invalidate the record."""
        return f"{self.file}::{self.symbol}::0x{self.addr:04X}"

    @property
    def enforced(self) -> bool:
        return self.kind in ENFORCED_KINDS


@dataclass
class Site:
    """A deduplicated site. Sibling configs quarantine the same addresses
    (`zelda.yaml` and `zelda_gui_tuned.yaml` are byte-identical on all 12),
    so one line of source is reached by several quarantine entries; the
    site is the unit a disclosure covers."""
    site_id: str
    file: str
    lines: list[int] = field(default_factory=list)
    symbol: str = ""
    addr: int = 0
    reward_id: str = ""
    kind: str = ""
    quarantines: list[str] = field(default_factory=list)

    @property
    def enforced(self) -> bool:
        return self.kind in ENFORCED_KINDS


def dedupe(uses: list[Use]) -> list[Site]:
    by_id: dict[str, Site] = {}
    for u in uses:
        s = by_id.get(u.site_id)
        if s is None:
            s = by_id[u.site_id] = Site(u.site_id, u.file, [], u.symbol,
                                        u.addr, u.reward_id, u.kind, [])
        if u.line not in s.lines:
            s.lines.append(u.line)
        if u.quarantine not in s.quarantines:
            s.quarantines.append(u.quarantine)
    for s in by_id.values():
        s.lines.sort()
        s.quarantines.sort()
    return sorted(by_id.values(), key=lambda s: (s.file, s.lines[0]))


#: Emulator-hardware modules. CPU/PPU/APU/mapper/cartridge code is full of
#: constants that collide numerically with RAM addresses (`PRG_BANK:
#: usize = 0x2000`), and hardware fidelity sits outside the purity line by
#: standing project rule. Restricting the unowned-Rust branch by MODULE
#: rather than by identifier prefix is what lets the prefix requirement go:
#: `pub const GANON_DEFEATED: usize = 0x0672;` in a helper module used to
#: pass simply because it was not called `RAM_*`.
_FIDELITY_MODULES = (
    "mapper", "ppu", "apu", "cpu", "cartridge", "bus", "controller",
    "nes.rs", "video_sink", "audio",
)


def _is_fidelity_module(rel: str) -> bool:
    name = Path(rel).name
    return any(tok in name for tok in _FIDELITY_MODULES)


def _scan_unowned_rust(lines: list[str], rel: str,
                       qs: list[Quarantine]) -> list[Use]:
    """`RAM_*: usize = 0x..` in a file with no reward dispatch table."""
    tests = _test_line_spans(lines)
    out: list[Use] = []
    for n, line in enumerate(lines):
        if line.lstrip().startswith("//"):
            continue
        m = _RUST_ADDR_CONST.match(line)
        if not m:
            continue
        sym = m.group(1)
        if (_is_fidelity_module(rel) or _BITMASK_NAME.search(sym)
                or _QUANTITY_NAME.search(sym)):
            continue
        value = int(m.group(2), 0)
        for q in qs:
            if q.addr != value:
                continue
            out.append(Use(
                rel, n + 1, m.group(1), value, q.reward_id,
                f"{q.config}:{q.key}",
                "rust-test" if any(a <= n <= b for a, b in tests)
                else "rust-unowned"))
    return out


def scan_rust(qs: list[Quarantine], rust_dir: Path | None = None,
              repo: Path | None = None) -> list[Use]:
    rust_dir = rust_dir if rust_dir is not None else RUST_DIR
    repo = repo if repo is not None else REPO
    by_id: dict[str, list[Quarantine]] = {}
    for q in qs:
        if q.reward_id:
            by_id.setdefault(q.reward_id, []).append(q)

    every = [q for q in qs]
    uses: list[Use] = []
    for path in sorted(rust_dir.rglob("*.rs")):
        if _excluded(path.relative_to(repo)):
            continue
        text = path.read_text()
        lines = text.splitlines()
        rel = str(path.relative_to(repo))
        struct_to_id = _rust_variant_ids(text)
        if not struct_to_id:
            # No reward dispatch here, so there is no ownership signal. The
            # obvious way to evade an owner-scoped check is to move the
            # constant OUT of the owned impl into a helper module, so this
            # branch closes that door: any `RAM_*: usize` — this codebase's
            # own naming convention for a RAM address — carrying a
            # quarantined value is flagged, whichever game it belongs to.
            #
            # Cross-game collision is a real possibility here (0x0010 and
            # 0x0070 are ordinary zero-page addresses) and it is handled by
            # the disclosure record rather than by an exemption: an
            # unrelated address that collides costs ONE row saying so,
            # which is a useful fact to have written down. That is the
            # difference between a check with an escape hatch and a check
            # people disable.
            uses.extend(_scan_unowned_rust(lines, rel, every))
            continue
        tests = _test_line_spans(lines)

        def in_tests(n: int) -> bool:
            return any(a <= n <= b for a, b in tests)

        for struct, start, end in _impl_regions(lines):
            reward_id = struct_to_id.get(struct)
            if reward_id is None:
                continue
            targets = by_id.get(reward_id) or []
            if not targets:
                continue
            for n in range(start, min(end + 1, len(lines))):
                line = lines[n]
                if line.lstrip().startswith("//"):
                    continue
                hits: list[tuple[str, int]] = []
                m = _RUST_ADDR_CONST.match(line)
                if m:
                    hits.append((m.group(1), int(m.group(2), 0)))
                ma = _RUST_ADDR_ARRAY.match(line)
                if ma:
                    for tok in re.findall(r"0[xX][0-9A-Fa-f]+|\d+", ma.group(2)):
                        hits.append((ma.group(1), int(tok, 0)))
                for mi in _RUST_RAM_INDEX.finditer(line):
                    hits.append((mi.group(0), int(mi.group(1), 16)))
                for symbol, value in hits:
                    for q in targets:
                        if q.addr != value:
                            continue
                        uses.append(Use(
                            rel, n + 1, symbol, value, reward_id,
                            f"{q.config}:{q.key}",
                            "rust-test" if in_tests(n) else (
                                "rust-index" if symbol.startswith("ram")
                                else "rust-const"),
                        ))
    return uses


#: `ANYTHING = 0x0672` / `anything[0x0672]` / `[0x0672, ...]` in Python.
#:
#: These deliberately do NOT require the identifier to be named `RAM_*` or
#: `*_addr`. An earlier version did, which meant the check detected this
#: repo's NAMING CONVENTION rather than the address: renaming
#: `RAM_GANON_DEFEATED` to `GANON_DEFEATED`, or reading through a variable
#: called `r` instead of `ram`, walked straight through `make purity-check`
#: while still reading the quarantined byte live. Three real in-tree sites
#: were invisible for exactly that reason, including this sweep's own
#: measurement probe.
#:
#: The shape discipline that keeps false positives down is elsewhere: a
#: bare hex literal in arithmetic is still NOT a hit (so `& 0x0001` in
#: `apu.rs` and a controller-bit comment do not flag), and the file must
#: OWN the ROM (see `_py_owner_tokens`). Measured over the whole tree:
#: flagging every literal would report 354 sites across 50 files; these
#: three shapes report 84, of which the enforced subset is small enough to
#: disclose one row at a time.
_PY_ADDR_ASSIGN = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*(?::\s*[^=]+)?=\s*(0[xX][0-9A-Fa-f]+)\s*$")
_PY_RAM_INDEX = re.compile(r"\b[A-Za-z_]\w*\s*\[\s*(0[xX][0-9A-Fa-f]+)\s*\]")
#: Identifiers that name a BIT, not an address. Broadening the assignment
#: pattern to any identifier reintroduced exactly one false positive:
#: `src/diagnostics/worker_debug.py`'s `_BTN_UP = 0x10`, a controller
#: bitmask, in a file that mentions "zelda" only in a comment explaining a
#: past defect. A checker that reports a button bit as a purity breach
#: trains readers to ignore it, so the bitmask vocabulary is excluded by
#: name. This is a NEGATIVE allowlist and it is pinned by its own test:
#: `test_bitmask_names_are_the_only_suppression` fails if it ever
#: suppresses a site that is not a bit constant.
_BITMASK_NAME = re.compile(
    r"(?i)(^|_)(btn|button|mask|bit|bits|flag|flags)(_|$)")

#: Names that describe a SIZE, a BANK or an OFFSET rather than a location.
#: `nes_core/src/memory.rs` has `RAM_SIZE: usize = 0x0800` and mapper code
#: is full of `PRG_BANK: usize = 0x2000`; those collide numerically with
#: RAM addresses without ever reading one.
_QUANTITY_NAME = re.compile(
    r"(?i)(^|_)(len|size|count|bank|offset|base|stride|cap|capacity|"
    r"width|height|stride|step)(_|$)")

#: An address sitting in a list/tuple/set literal or as a dict VALUE —
#: `WATCH = [0x11, 0x70, 0x84]`, `{"ganon": 0x0672}`. Both forms read the
#: byte just as live as a named constant does.
#: The `[` case requires that no identifier precedes it, or every
#: `ram[0x0077]` subscript would be counted twice — once as a binding read
#: and once as a literal.
_PY_ADDR_ELEMENT = re.compile(
    r"(?:(?<![\w\)\]])\[|[\(\{,:])\s*(0[xX][0-9A-Fa-f]+)\s*"
    r"(?=[,\]\)\}]|$)")


def _py_owner_tokens(q: Quarantine) -> set[str]:
    """What must appear in a Python file for it to be considered code that
    OWNS this ROM. Without this, `0x0001` in `tests/test_button_bit_
    alignment.py` (a controller-bit comment) would be flagged as a
    Punch-Out purity breach — a false positive that trains readers to
    ignore the check."""
    toks = {q.reward_id, Path(q.config).stem}
    stem = Path(q.rom).stem
    toks.add(stem)
    toks.add(re.sub(r"\s*\([^)]*\)", "", stem).strip())
    return {t.lower() for t in toks if t}


def scan_python(qs: list[Quarantine], repo: Path | None = None) -> list[Use]:
    repo = repo if repo is not None else REPO
    uses: list[Use] = []
    for path in sorted(repo.rglob("*.py")):
        rel_path = path.relative_to(repo)
        if _excluded(rel_path):
            continue
        rel = str(rel_path)
        try:
            text = path.read_text()
        except Exception:
            continue
        low = text.lower()
        lines = text.splitlines()
        for q in qs:
            if not any(t in low for t in _py_owner_tokens(q)):
                continue
            for n, raw in enumerate(lines):
                if raw.lstrip().startswith("#"):
                    continue
                # A trailing comment is prose, not a read.
                line = raw.split("#")[0]
                hits: list[tuple[str, int, bool]] = []
                m = _PY_ADDR_ASSIGN.match(line)
                if m and not _BITMASK_NAME.search(m.group(1)):
                    hits.append((m.group(1), int(m.group(2), 16), True))
                for mi in _PY_RAM_INDEX.finditer(line):
                    hits.append((mi.group(0).strip(), int(mi.group(1), 16),
                                 True))
                for mi in _PY_ADDR_ELEMENT.finditer(line):
                    hits.append((f"literal {mi.group(1)}",
                                 int(mi.group(1), 16), False))
                is_test = (rel.startswith("tests/")
                           or Path(rel).name.startswith("test_"))
                for symbol, value, binds in hits:
                    if value != q.addr:
                        continue
                    if is_test:
                        kind = "python-test"
                    elif binds:
                        kind = "python"
                    else:
                        kind = "python-literal"
                    uses.append(Use(rel, n + 1, symbol, value, q.reward_id,
                                    f"{q.config}:{q.key}", kind))
    return uses


def scan_quarantined_uses(repo: Path | None = None) -> list[Site]:
    """Every deduplicated site in the tree where a quarantined address is
    still reachable, ENFORCED and REPORTED kinds alike.

    Pass `repo` to run the whole checker against a synthetic tree — that is
    how the mutation tests prove the predicate still bites.
    """
    if repo is None:
        # The real tree does not change inside one process, and the scan
        # is the expensive part of `make test`'s purity gate. Injected
        # roots are NEVER cached: the mutation tests write to a temp tree
        # and re-scan it, and a stale answer there would make a revert
        # experiment silently pass.
        return list(_scan_default())
    qs = quarantines(repo / "configs", repo)
    return dedupe(scan_rust(qs, repo / "nes_core" / "src", repo)
                  + scan_python(qs, repo))


@functools.lru_cache(maxsize=1)
def _scan_default() -> tuple[Site, ...]:
    qs = quarantines()
    return tuple(dedupe(scan_rust(qs) + scan_python(qs)))


# =====================================================================
# 3b. The disclosure record
# =====================================================================

DISCLOSURES = REPO / "docs" / "purity" / "engine_quarantine_disclosures.yaml"

#: Fields every disclosure row must carry, non-empty, whatever kind of
#: admission it is making. `earns_it` is NOT here: a MEASUREMENT_INSTRUMENT
#: row has nothing to earn back, because it never made a claim. It is
#: required of LIVE_UNRETRACTED rows by its own dedicated check, which is
#: where the rediscovery path actually matters.
REQUIRED_FIELDS = ("site", "disposition", "asserts", "no_witness",
                   "behaviour")

VALID_DISPOSITIONS = frozenset({
    "LIVE_UNRETRACTED", "MEASUREMENT_INSTRUMENT", "FIDELITY_TRACE",
})

#: Text that looks like a field but says nothing. A disclosure whose
#: justification is "TODO" is the vacuous gate this file exists to avoid
#: becoming.
VACUOUS = frozenset({"", "-", "tbd", "todo", "n/a", "na", "none", "?",
                     "unknown", "xxx", "fixme"})


def load_disclosures(path: Path | None = None) -> dict:
    path = path if path is not None else DISCLOSURES
    return _load_yaml(path)


def check_disclosures(repo: Path | None = None,
                      disclosures: dict | None = None) -> list[str]:
    """Every reason the tree fails the engine-quarantine check, as text.

    ONE implementation, shared by `make purity-check` and by
    `tests/test_purity_engine_sweep.py`. A Makefile gate and a pytest gate
    that each reimplement the rule drift apart, and the one that drifts
    quiet is the one nobody notices — that is how a vacuous gate is born.
    """
    doc = disclosures if disclosures is not None else load_disclosures(
        (repo / "docs" / "purity" / "engine_quarantine_disclosures.yaml")
        if repo is not None else None)
    declared = {str(r["site"]): r for r in (doc.get("sites") or [])
                if isinstance(r, dict) and r.get("site")}
    live = [s for s in scan_quarantined_uses(repo) if s.enforced]

    problems: list[str] = []
    for s in live:
        if s.site_id not in declared:
            problems.append(
                f"UNDECLARED  {s.file}:{','.join(map(str, s.lines))}  "
                f"{s.symbol} = 0x{s.addr:04X}  quarantined by "
                f"{'; '.join(s.quarantines)}")
    for site in sorted(set(declared) - {s.site_id for s in live}):
        problems.append(f"STALE       {site} — declared, but no such live site")

    cap = doc.get("max_enforced_sites")
    if not isinstance(cap, int):
        problems.append("RATCHET     `max_enforced_sites:` is missing or not an int")
    elif len(live) > cap:
        problems.append(
            f"RATCHET     {len(live)} enforced sites against a cap of {cap}. "
            f"Remove the use; do not raise the cap.")

    for site, row in sorted(declared.items()):
        for f in REQUIRED_FIELDS:
            v = str(row.get(f, "")).strip()
            if not v or v.lower() in VACUOUS:
                problems.append(f"VACUOUS     {site}: `{f}` is empty or a placeholder")
        d = str(row.get("disposition", "")).strip()
        if d and d not in VALID_DISPOSITIONS:
            problems.append(f"VACUOUS     {site}: unknown disposition {d!r}")
        if (d == "LIVE_UNRETRACTED"
                and len(str(row.get("earns_it", "")).strip()) < 25):
            problems.append(
                f"VACUOUS     {site}: LIVE_UNRETRACTED with no usable `earns_it`")
    return problems


# =====================================================================
# 4. The witness ledger — generated from solver tapes, not hand-listed
# =====================================================================

#: A solver clear tape names its ROM only indirectly, through the run
#: tree's `roots.json`. A tree rooted at a MINTED state (a chained level)
#: inherits its ROM from whichever tree minted that state.
_ROMS_STATE = re.compile(r"^roms/(.+?)(?:_start)?\.state(?:\.bin)?$")


def _root_paths(tree: Path) -> list[str]:
    rj = tree / "roots.json"
    if not rj.exists():
        return []
    try:
        doc = json.loads(rj.read_text())
    except Exception:
        return []
    return [str(r["path"]) for r in doc.values()
            if isinstance(r, dict) and r.get("path")]


def witness_ledger(runs_dir: Path | None = None) -> dict:
    """Which ROMs this repository has a WITNESSED solver clear on.

    A solution tape counts as a witness only when it carries a non-empty
    `start_wd` AND `clear_wd` that DIFFER — an independent observable
    actually moved across the tape. That is the exact discriminant that
    separates the real clears from the two withdrawn false positives:
    Kirby's and Double Dragon's banked `sol_000` both carry `start_wd []`
    and `clear_wd []`, fired by the confluence detector with nothing
    behind them.

    ABSENCE OF A WITNESS HERE IS NOT A CLAIM THAT NOTHING WAS CLEARED.
    This reads ONE evidence source — the solver's own tapes. A game whose
    clear was receipted another way (a rendered finale frame, a
    pre-registered byte_change) has no tape here and is reported as
    `witnessed: false` BY THIS SOURCE. Read the row, not the boolean.
    """
    runs = runs_dir if runs_dir is not None else REPO / "runs"
    trees = sorted({p.parent.parent for p in runs.rglob("solutions/*.json")})

    tree_rom: dict[Path, str] = {}
    for t in trees:
        for p in _root_paths(t):
            m = _ROMS_STATE.match(p)
            if m:
                tree_rom[t] = m.group(1)
                break

    # Minted-state inheritance: a tree rooted under a directory whose
    # already-attributed trees all agree on one ROM inherits that ROM.
    for _ in range(32):
        changed = False
        for t in trees:
            if t in tree_rom:
                continue
            for p in _root_paths(t):
                cand: set[str] = set()
                probe = (runs.parent / p).parent
                while probe != runs.parent and probe != probe.parent:
                    cand = {r for tt, r in tree_rom.items()
                            if str(tt).startswith(str(probe))}
                    if cand:
                        break
                    probe = probe.parent
                if len(cand) == 1:
                    tree_rom[t] = cand.pop()
                    changed = True
                    break
        if not changed:
            break

    rows: dict[str, dict] = {}
    unattributed = 0
    unattributed_wit = 0
    for t in trees:
        tapes = sorted(t.glob("solutions/*.json"))
        wit = []
        for s in tapes:
            try:
                d = json.loads(s.read_text())
            except Exception:
                continue
            a, b = d.get("start_wd"), d.get("clear_wd")
            if a and b and a != b:
                try:
                    wit.append(str(s.relative_to(REPO)))
                except ValueError:
                    wit.append(str(s.relative_to(runs.parent)))
        rom = tree_rom.get(t)
        if rom is None:
            unattributed += 1
            unattributed_wit += len(wit)
            continue
        r = rows.setdefault(rom, {"rom": rom, "witnessed": False,
                                  "tapes": 0, "witness_tapes": 0,
                                  "evidence": []})
        r["tapes"] += len(tapes)
        r["witness_tapes"] += len(wit)
        if wit:
            r["witnessed"] = True
            r["evidence"] = sorted(set(r["evidence"] + wit))[:3]

    return {
        "source": "solver clear tapes (runs/**/solutions/*.json + roots.json)",
        "discriminant": "start_wd and clear_wd both non-empty AND different",
        "caveat": ("witnessed:false means NO WITNESS FROM THIS SOURCE. Games "
                   "receipted by a rendered finale frame or a pre-registered "
                   "byte_change leave no solver tape and appear false here."),
        "unattributed_trees": unattributed,
        "unattributed_witness_tapes": unattributed_wit,
        "blind_spot": ("a run tree whose roots.json points at a minted state "
                       "with no attributable ancestor cannot be traced back "
                       "to a ROM. The SMB chain roots at runs/chain_handoffs/ "
                       "and lands here — SMB's 32-level clear is real and is "
                       "receipted elsewhere; it is invisible TO THIS SOURCE. "
                       "The count is published so the gap is sized, not hidden."),
        "roms": [rows[k] for k in sorted(rows)],
    }


if __name__ == "__main__":  # pragma: no cover - operator entry point
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", action="store_true",
                    help="print the evidence-derived witness ledger as JSON")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if a quarantined address is live in "
                         "the executing layer with no disclosure")
    args = ap.parse_args()

    if args.ledger:
        print(json.dumps(witness_ledger(), indent=2))
    elif args.check:
        problems = check_disclosures()
        for line in problems:
            print(line)
        if problems:
            print(f"\n{len(problems)} problem(s). Quarantining the YAML "
                  f"retracts the documentation claim, not the constant — "
                  f"see {DISCLOSURES.relative_to(REPO)}.")
            raise SystemExit(1)
        n = sum(1 for s in scan_quarantined_uses() if s.enforced)
        print(f"purity: {n} live quarantined-address site(s), all disclosed.")
    else:
        sites = scan_quarantined_uses()
        for s in sites:
            mark = "ENFORCED" if s.enforced else "reported"
            print(f"{mark}  {s.file}:{','.join(map(str, s.lines))}  "
                  f"{s.kind:12s} 0x{s.addr:04X}  {s.symbol:22s} "
                  f"<- {'; '.join(s.quarantines)}")
        n_enf = sum(1 for s in sites if s.enforced)
        print(f"\n{len(sites)} site(s): {n_enf} ENFORCED, "
              f"{len(sites) - n_enf} reported-only.")
