# Verifying this tape against its receipt

`receipts.json`'s `tape_sha256` is computed over the RAW uint8 action
bytes — `np.load("full_tape.npy").tobytes()` — not over the `.npy`
file (numpy prepends a ~128-byte header, so `shasum full_tape.npy`
gives a DIFFERENT hash on an intact artifact; that misread froze a
127G archival decision on 2026-08-29).

Recipe (also enforced by `tests/test_full_run_receipt_integrity.py`):

```python
import numpy as np, hashlib, json
arr = np.load("docs/receipts/full_run/full_tape.npy")
assert hashlib.sha256(arr.tobytes()).hexdigest() == \
    json.load(open("docs/receipts/full_run/receipts.json"))["tape_sha256"]
```

Verified 2026-08-29: exact match (31,202 steps, one uint8 action per
step, dtype/shape as attested). The banked `receipts.json` predates the
`tape_sha256_domain` field the assembler now writes; it is history and
stays byte-identical.

Re-verified 2026-09-01: `make verify-full-run` -> replay_2026-09-01.json
(nes_core rebuilt from HEAD at 05f18243ed74, cold boot, 32/32 level
boundaries OK, opermode=2, 119s).
