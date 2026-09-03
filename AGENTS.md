# Agent operating contract (SSOI)

This repository owns TCDR training, bundle I/O, and `VirtualSensor` inference.

BibMon is the host system. Do not import `bibmon` from `ssoi`. Do not copy
BibMon source. Canonical integration spec lives in the BibMon repo:

- `doc/SSOI_BIBMON.md`
- `specs/001-ssoi-outage-integration/`
- `doc/adr/`

Roadmap in this repo: `doc/PLANO_INTEGRACAO_BIBMON.md`.

## Git

Read-only git inspection is allowed. Create a local branch when implementing.

Do not run `git commit`, `git push`, merge, rebase, PR, tag, or release without
explicit user authorization for that operation and target.

Do not run `git reset --hard`, `git clean -fd`, or force-push unless the user
explicitly requests it.

At the end of a task, propose commit commands; do not execute them.
