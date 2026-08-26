"""Static census of the vacuous-gate idiom: a verdict computed as the
mere ABSENCE of a findings-shaped collection (`passed = not findings`,
`ok = len(errors) == 0`, ...) rather than the PRESENCE of a positive,
measured threshold.

This idiom is not inherently wrong -- `progress_signal_gate.assess`'s
`"passed": not instrument` is exactly this shape and is legitimate,
because every branch that can append to `instrument` is a real,
independently-measured check. What made it a bug three separate times
this week (2026-08-26: is_clear's `() > ()`, area()'s literal-0 default,
the D6 camera-static override that force-passed on a flat trace) is
never the syntax -- it is a collection that can be empty for a reason
that has nothing to do with the thing under test. Syntax is the only
part a scanner can see, so this module finds every site with the SHAPE
and tests/test_anti_vacuity_gates.py requires a human-reviewed proof,
run every time, that each one can still report both a pass and a fail.

Deliberately narrow: exact AST shapes only (`not <atom>`,
`len(<atom>) == 0`), an exact five-word verdict-name allowlist, and only
`scripts/` + `src/` (the modules that gate real decisions). It will not
catch every vacuous check in the world -- it exists to make a NEW
instance of THIS idiom impossible to add silently, not to replace
review. See tests/test_anti_vacuity_gates.py for the enforcement half.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("scripts", "src")
EXCLUDE_DIR_NAMES = {"__pycache__", ".venv", "node_modules", "vendor",
                     "_preserved", ".git"}

#: Exact verdict-name allowlist. Kept short and literal on purpose --
#: widening this to substrings ("passed" matching "elapsed", "ok"
#: matching anything containing "ok") is exactly the kind of scope creep
#: that turns a narrow, always-right check into a noisy one.
VERDICT_NAMES = {"passed", "success", "ok", "valid", "clear"}


class VacuityHit(NamedTuple):
    file: str          # repo-relative path, forward slashes
    line: int
    func: str          # dotted qualname of the enclosing def, or "<module>"
    target: str
    source: str
    shape: str


def _is_verdict_target(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower() in VERDICT_NAMES
    if isinstance(node, ast.Name):
        return node.id.lower() in VERDICT_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr.lower() in VERDICT_NAMES
    if isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return sl.value.lower() in VERDICT_NAMES
    return False


def _atom_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant):
            return str(sl.value)
        return "<subscript>"
    try:
        return ast.unparse(node)
    except Exception:
        return "<expr>"


def _target_label(node: ast.expr) -> str:
    if isinstance(node, ast.Constant):
        return str(node.value)
    return _atom_name(node)


class _Visitor(ast.NodeVisitor):
    def __init__(self, relpath: str, hits: list):
        self.relpath = relpath
        self.hits = hits
        self.func_stack: list[str] = []

    def _qualname(self) -> str:
        return ".".join(self.func_stack) if self.func_stack else "<module>"

    def visit_FunctionDef(self, node):
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _shape_of(self, value: ast.expr):
        """Returns (source_atom_name, shape_label) if `value` is
        `not <atom>` or `len(<atom>) == 0` (either operand order), else
        None. Anything more complex (boolean combinations, comparisons
        against non-zero, membership tests) is out of scope by design --
        those are not the literal idiom named in the postmortem."""
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
            return _atom_name(value.operand), "not <x>"
        if (isinstance(value, ast.Compare) and len(value.ops) == 1
                and isinstance(value.ops[0], ast.Eq)):
            left, right = value.left, value.comparators[0]

            def as_len_call(n):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id == "len" and n.args):
                    return _atom_name(n.args[0])
                return None

            if as_len_call(left) is not None and isinstance(right, ast.Constant) and right.value == 0:
                return as_len_call(left), "len(<x>) == 0"
            if as_len_call(right) is not None and isinstance(left, ast.Constant) and left.value == 0:
                return as_len_call(right), "0 == len(<x>)"
        return None

    def _record(self, target: ast.expr, value: ast.expr, lineno: int):
        shaped = self._shape_of(value)
        if shaped is None or not _is_verdict_target(target):
            return
        source, shape = shaped
        self.hits.append(VacuityHit(
            file=self.relpath, line=lineno, func=self._qualname(),
            target=_target_label(target), source=source, shape=shape))

    def visit_Assign(self, node):
        for t in node.targets:
            self._record(t, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._record(node.target, node.value, node.lineno)
        self.generic_visit(node)

    def visit_Dict(self, node):
        for k, v in zip(node.keys, node.values):
            if k is not None:
                self._record(k, v, node.lineno)
        self.generic_visit(node)


def scan_repo(repo: Path | None = None,
             scan_dirs: tuple = SCAN_DIRS) -> list:
    """Walks `scan_dirs` under `repo` and returns every VacuityHit found,
    sorted by (file, line). Deterministic, side-effect-free, safe to call
    from a test on every run."""
    repo = repo or REPO
    hits: list = []
    for d in scan_dirs:
        base = repo / d
        if not base.is_dir():
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in EXCLUDE_DIR_NAMES
                      and not x.startswith(".")]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                p = Path(root) / fname
                rel = p.relative_to(repo).as_posix()
                try:
                    tree = ast.parse(p.read_text(encoding="utf-8"), filename=rel)
                except SyntaxError:
                    # Silent, deliberately: a file that fails to parse
                    # already fails to import, which is a loud failure
                    # everywhere else in the suite. This scanner staying
                    # quiet about it too means a file mid-write in a
                    # shared working tree can't turn into a SPURIOUS
                    # anti-vacuity failure on top of the real one.
                    continue
                _Visitor(rel, hits).visit(tree)
    hits.sort(key=lambda h: (h.file, h.line))
    return hits


def main() -> int:
    hits = scan_repo()
    for h in hits:
        print(f"{h.file}:{h.line}  func={h.func}  target={h.target!r}  "
              f'shape="{h.shape}"  source={h.source!r}')
    print(f"\n{len(hits)} site(s) found", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
