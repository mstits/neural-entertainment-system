# Banked-tape corpus: replay-integrity verdict (final)

Sweep of every `runs/**/solutions/*.json` (310 tapes), each replayed from
its own recorded root under its own recorded machine lineage, scored by
its own profile's clear predicate. Receipt:
`runs/engine/replay_sweep_final.json` (stamp 2026-08-22-final).

| verdict | n | meaning |
|---|---|---|
| PASS | **107** | replays from root to its banked clear |
| FAIL | **0** | — |
| UNSCORABLE | 112 | root already satisfies the predicate (provenance mislabel), or banked under a predicate the verifier cannot express (SMB finales via opermode) |
| ERROR | 91 | unverifiable — 83 never recorded a profile, 6 point at deleted /tmp profiles, 2 other |

**Zero broken tapes.** Four candidate failures each dissolved under
scrutiny: the 8-4 finale (predicate inexpressible) and three Castlevania
tapes (root hw-lineage mismatch; all clear when replayed under the root
sidecar's recorded flags). The earlier standing finding "2 of 8 banked
tapes do not replay" is REFUTED — both replay fine and were merely
unverifiable for want of recorded provenance.

The gate still reports not-passed, deliberately: 91 tapes cannot be
checked, and "cannot be checked" is not "passes." Those tapes' claims
rest on their original run logs only. Every future solver run records
profile + hw lineage in the sidecar, which makes this class extinct going
forward.

Verifier defects fixed to get here (each was reporting fiction):
frame-0 false passes (111 of the first 171), Exception-vs-SystemExit
killing whole sweeps, missing rom fallback, profile recovery via
consumer manifests and sibling tapes, root-sidecar lineage pinning,
finale-predicate UNSCORABLE class.
