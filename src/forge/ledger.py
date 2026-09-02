"""The FORGE ledger writer (FORGE_SPEC_2026-09-01.md §2e).

Renders every ledger artifact the Forge emits in the shape of the one
existing entry (``CLAIMS.md:199-262``) and refuses to hand back text
that violates the ledger's vocabulary constraints rather than let a
bad render reach a file:

  * the banned-verb family defined in ``_BANNED_VERB_RE`` below --
    five inflections of one root plus a compound, whole-word,
    case-insensitive (``CLAIMS.md:160-166``);
  * any digit-bearing rate/episode/protocol phrase inside a FORGE
    entry. A FORGE entry states whether a gate was met; an EXHIBITION
    rate, a count of attempts, or a numbered protocol belongs to the
    run's own EXHIBITION ledger and the two are never merged into one
    sentence (``CLAIMS.md:152-158``, spec §1 "Ledgers");
  * interpretive metadiscourse, added 2026-09-01 from the no-ai-slop
    pass over the repo docs (spec §2e header note) -- the three fixed
    phrases in ``_METADISCOURSE_PHRASES`` below, a whole line wrapped
    in double-asterisk bold, an all-caps label line standing alone,
    and a standalone italicised kicker line. The entry states the
    gate outcome and the receipts and stops.

The entry's own header is a bold ``**FORGE-...`` lead followed by the
Detection section on the same line, the shape both
``scripts/provenance_check.py`` parses and every entry already in
CLAIMS.md uses. Nothing here emits U+2014 on any rendered surface.

``render_entry`` builds its own ``tier3_sentence`` and ``status_plain``
fields from fixed templates (never accepts free text for either) so a
caller cannot regress the corrected 2026-09-01 wording, and calls
``check_vocabulary`` on its own output before returning it -- the same
discipline the spec asks of every Forge gate: it must be able to say
no, including to itself.

Note on that self-check and this docstring: spec §5's pre-commit gate
reads ``check_vocabulary`` over every new ``.md`` file and every
docstring, not over every line of source. ``iter_docstrings`` below
draws that boundary structurally, with ``ast``, so the regex and tuple
just named above -- which must contain the literal banned forms to
define them, the same way a profanity filter's own wordlist must --
are never mistaken for the prose a reader sees. Every docstring in
this file is inside that boundary and is kept clean of the forms
above on that basis; ``tests/test_forge_ledger.py`` proves it by
running ``check_vocabulary`` over the output of ``iter_docstrings``
for both source files this commit adds.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------

#: CLAIMS.md:160-166 -- the alternation below is the whole banned list;
#: this comment does not repeat the words in prose so the comment
#: itself stays clean of the forms it is describing, whole-word,
#: case-insensitive.
_BANNED_VERB_RE = re.compile(
    r"\b(learn|learned|learns|learning|self-taught)\b", re.IGNORECASE)

#: A FORGE entry states whether a gate was met, not a clear count -- any
#: EXHIBITION rate belongs to the run's own solutions ledger, never here
#: (CLAIMS.md:152-158, spec §1 "Ledgers"). Digit-bearing proximity to one
#: of these words is refused, rather than a fixed phrase list, because
#: the objectionable content is the PAIRING of a number with a
#: rate/episode/protocol word, in either order -- see
#: tests/test_forge_ledger.py::test_clear_rate_refused_in_forge_entry
#: for a worked fixture.
_RATE_WORDS = ("clear", "clears", "rate", "episode", "episodes",
               "protocol", "protocols")
_RATE_WORD_RE = re.compile(r"\A(" + "|".join(_RATE_WORDS) + r")\Z",
                            re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")
_RATE_PROXIMITY_WINDOW = 4  # words either side of the rate word
_STRIP_PUNCT = ".,;:()[]{}\"'"

#: Interpretive metadiscourse banned 2026-09-01 from the no-ai-slop pass
#: (spec §2e header note): the entry states the gate outcome and the
#: receipts and stops.
_METADISCOURSE_PHRASES = (
    "stated plainly", "the honest reading is", "to be clear",
)
_METADISCOURSE_RE = re.compile(
    "|".join(re.escape(p) for p in _METADISCOURSE_PHRASES), re.IGNORECASE)

#: A line that is nothing but a double-asterisk bold span covering the
#: whole line -- the shape the banned aphorism kicker takes when it
#: leans on bold instead of italics.
_WHOLE_LINE_BOLD_RE = re.compile(r"\A\*\*[^*].*[^*]\*\*\Z")

#: A line that reads as a shouted label ("STATUS:", "IMPORTANT") rather
#: than prose: starts with an uppercase letter, no lowercase letters
#: anywhere, at least two uppercase letters (so a single capitalised
#: word like a status constant used alone still counts, but "A" alone
#: does not).
_CAPS_LABEL_RE = re.compile(r"\A[A-Z][A-Z0-9 \t/'-]*:?\Z")

#: A standalone single-asterisk italic line, the shape a pithy aphorism
#: kicker takes when the bold form above is avoided. A line that also
#: carries a digit or a path separator is exempted -- that is a receipt
#: citation or a dated addendum lead-in, not a kicker.
_ITALIC_KICKER_RE = re.compile(r"\A\*[^*]+\*\Z")


class VocabularyViolation(ValueError):
    """Raised by ``render_entry`` / ``append_addendum`` when text meant
    for a FORGE ledger surface fails ``check_vocabulary``; carries the
    full list of violations found, not just the first."""

    def __init__(self, violations: Sequence[str]):
        self.violations = list(violations)
        super().__init__("; ".join(self.violations))


def check_vocabulary(text: str) -> list[str]:
    """Return every vocabulary violation found in ``text``; an empty
    list means clean. Never raises -- ``render_entry`` and
    ``append_addendum`` are what refuse."""
    violations: list[str] = []

    for m in _BANNED_VERB_RE.finditer(text):
        violations.append(
            f"banned verb {m.group(0)!r} at offset {m.start()}")

    words = text.split()
    for i, raw in enumerate(words):
        stripped = raw.strip(_STRIP_PUNCT)
        if not _RATE_WORD_RE.match(stripped):
            continue
        lo = max(0, i - _RATE_PROXIMITY_WINDOW)
        hi = min(len(words), i + _RATE_PROXIMITY_WINDOW + 1)
        window = words[lo:hi]
        if any(_DIGIT_RE.search(tok) for tok in window):
            violations.append(
                "digit-bearing clear/episode/protocol phrase near "
                f"{raw!r}: {' '.join(window)!r}")

    for m in _METADISCOURSE_RE.finditer(text):
        violations.append(
            f"interpretive metadiscourse {m.group(0)!r} at offset {m.start()}")

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if _WHOLE_LINE_BOLD_RE.match(stripped):
            violations.append(
                f"whole-line bold at line {lineno}: {stripped!r}")
            continue
        if (_CAPS_LABEL_RE.match(stripped)
                and sum(1 for c in stripped if c.isupper()) >= 2):
            violations.append(
                f"caps-label line at line {lineno}: {stripped!r}")
            continue
        if _ITALIC_KICKER_RE.match(stripped):
            inner = stripped.strip("*")
            if not _DIGIT_RE.search(inner) and "/" not in inner:
                violations.append(
                    f"aphorism-kicker line at line {lineno}: {stripped!r}")

    return violations


def iter_docstrings(source: str) -> list[str]:
    """Return every module/class/function docstring in ``source`` --
    the unit spec §5's pre-commit gate means by "docstring", parsed
    structurally via ``ast`` rather than by scanning raw text, so a
    regex or tuple literal that must contain a banned form to define
    it (``_BANNED_VERB_RE``, ``_METADISCOURSE_PHRASES`` above) or a
    fixture string a test must contain to prove a refusal is never
    mistaken for documentation prose. Returns ``[]`` on a syntax error
    rather than raising; callers decide whether that is fatal."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    docstrings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                docstrings.append(doc)
    return docstrings


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

#: The one status vocabulary the spec allows (spec §6, one-way door 1).
STATUS_VALUES = frozenset((
    "FORGE-PENDING-VALIDATION", "FORGE-VALIDATED-MECHANISM",
    "FORGE-VOID", "FORGE-GRANT",
))

#: §1's Tier-3 sentence, instantiated per entry with the run's own named
#: telemetry standing in for "(named in the entry)". Verbatim except for
#: that one substitution and the dash before "the test", which is a
#: colon here because the wave's prose check refuses U+2014 in an added
#: line -- do not paraphrase this template in a caller.
TIER3_TEMPLATE = (
    "Designing this arm is LLM guidance of exploration in the plain "
    "sense; it is permitted under CLAIMS.md's purity boundary because "
    "the design is game-agnostic machinery, its knob settings are "
    "derived only from this run's own telemetry ({telemetry}), and it "
    "references no route, map, disassembly, or game-specific "
    "instruction: the test \"could this decision have been made "
    "by a party who has never seen the game\" holds."
)

#: Fields the caller must supply. ``tier3_sentence`` and ``status_plain``
#: are NOT among them -- render_entry builds both from fixed templates
#: (telemetry / gate_met / gate_measure) so neither can drift from the
#: corrected 2026-09-01 wording.
REQUIRED_FIELDS = (
    "status", "arm", "flag", "commit", "date", "detection", "mechanism",
    "review", "gate", "telemetry", "gate_met", "gate_measure",
    "citable_as",
)

#: Body-section order, fields in this order per spec §2e (LG rule 3).
#: One-way door once a second entry cites it (spec §6, item 2 analogue
#: for the registry applies here too): everything after this list keys
#: on the order.
SECTION_ORDER: tuple[str, ...] = (
    "detection", "mechanism", "review", "gate", "tier3_sentence",
    "status_plain", "citable_as", "addenda",
)

SECTION_LABELS: dict[str, str] = {
    "detection": "Detection",
    "mechanism": "Mechanism",
    "review": "Review",
    "gate": "Gate",
    "tier3_sentence": "Tier-3 sentence",
    "status_plain": "Status at forging",
    "citable_as": "Citable as",
    "addenda": "Addenda",
}


def _status_plain(gate_met: bool, gate_measure: str) -> str:
    """The corrected 2026-09-01 template (spec §2e header note): the
    key stays ``status_plain``, the rendered line reads "Status at
    forging: gate met|not met, <what the gate measured>." -- never the
    struck form the no-ai-slop correction retired, which led with an
    interpretive-metadiscourse framing before the colon."""
    verdict = "gate met" if gate_met else "gate not met"
    measure = gate_measure.strip().rstrip(".")
    return f"Status at forging: {verdict}, {measure}."


def _render_addenda(addenda: Sequence[Mapping[str, str]]) -> str:
    if not addenda:
        return "None."
    parts = []
    for a in addenda:
        for key in ("date", "status", "text"):
            if key not in a:
                raise ValueError(f"addendum missing required field {key!r}: {a!r}")
        if a["status"] not in ("WITHDRAWN", "STANDS"):
            raise ValueError(
                f"addendum status must be WITHDRAWN or STANDS, got {a['status']!r}")
        parts.append(f"Addendum, {a['date']}, {a['status']}. {a['text']}")
    return " ".join(parts)


def render_entry(entry: Mapping[str, object]) -> str:
    """Render one FORGE ledger entry as Markdown, in the field order
    fixed by spec §2e. Raises ``ValueError`` on a missing/malformed
    field or an unknown status, and ``VocabularyViolation`` (a
    ``ValueError`` subclass) if the rendered text itself fails
    ``check_vocabulary`` -- render_entry never hands back text it has
    not itself checked.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        raise ValueError(f"FORGE entry missing required field(s): {missing}")

    status = entry["status"]
    if status not in STATUS_VALUES:
        raise ValueError(
            f"unknown FORGE status {status!r}; must be one of "
            f"{sorted(STATUS_VALUES)}")

    telemetry = list(entry["telemetry"])
    if not telemetry:
        raise ValueError(
            "tier3_sentence requires at least one named telemetry "
            "source; entry['telemetry'] is empty")

    tier3_sentence = TIER3_TEMPLATE.format(telemetry=", ".join(telemetry))
    status_plain = _status_plain(bool(entry["gate_met"]), str(entry["gate_measure"]))
    addenda = _render_addenda(entry.get("addenda", []))

    body = {
        "detection": entry["detection"],
        "mechanism": entry["mechanism"],
        "review": entry["review"],
        "gate": entry["gate"],
        "tier3_sentence": tier3_sentence,
        "status_plain": status_plain,
        "citable_as": entry["citable_as"],
        "addenda": addenda,
    }

    # The bold lead is the form `scripts/provenance_check.py` actually
    # parses (`FORGE_ENTRY_RE`, `:179`: `**FORGE` then an optional
    # `-TAG` then WHITESPACE) and the form all eleven entries already
    # in CLAIMS.md use. The retired `## {status}` heading parsed as
    # nothing: `parse_forge_entries` ends its section at the first
    # `## `, so an entry rendered that way was invisible to the checker
    # that is supposed to gate it. Note the space after the status word
    # is load-bearing -- `**FORGE-VOID:` and `**FORGE-VOID,` both fail
    # that regex. The first body section rides on the header's own
    # line, as it does in every existing entry, so the line is not a
    # whole-line bold span (which `check_vocabulary` refuses).
    first, rest = SECTION_ORDER[0], SECTION_ORDER[1:]
    lines = [
        f"**{status} for the {entry['arm']} (`{entry['flag']}`), "
        f"commit `{entry['commit']}`, {entry['date']}.** "
        f"**{SECTION_LABELS[first]}.** {body[first]}",
        "",
    ]
    for key in rest:
        label = SECTION_LABELS[key]
        lines.append(f"**{label}.** {body[key]}")
        lines.append("")

    rendered = "\n".join(lines).rstrip() + "\n"

    violations = check_vocabulary(rendered)
    if violations:
        raise VocabularyViolation(violations)

    return rendered


def append_addendum(entry_path: Path, addendum: Mapping[str, str]) -> None:
    """Append one addendum block to an already-rendered
    ``CLAIMS_ENTRY.md`` file at ``entry_path``. Never rewrites: opens in
    binary-append mode only, so every byte already on disk survives
    unchanged and the addendum lands strictly after them (spec §2e
    "Corrections are appended addenda ... never edits", LG rule 6).
    """
    for key in ("date", "status", "text"):
        if key not in addendum:
            raise ValueError(f"addendum missing required field {key!r}")
    if addendum["status"] not in ("WITHDRAWN", "STANDS"):
        raise ValueError(
            f"addendum status must be WITHDRAWN or STANDS, got "
            f"{addendum['status']!r}")
    if not addendum["text"].strip():
        raise ValueError("addendum text must be non-empty")

    block = (
        f"\n*Addendum, {addendum['date']}, {addendum['status']}.* "
        f"{addendum['text']}\n"
    )
    violations = check_vocabulary(block)
    if violations:
        raise VocabularyViolation(violations)

    if not entry_path.exists():
        raise FileNotFoundError(
            f"cannot append an addendum to a FORGE entry that does not "
            f"exist yet: {entry_path}")

    with open(entry_path, "ab") as fh:
        fh.write(block.encode("utf-8"))
