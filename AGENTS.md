# Agent operating contract (SSOI)

This repository owns TCDR training, bundle I/O, and `VirtualSensor` inference.

BibMon is the host system. Do not import `bibmon` from `ssoi`. Do not copy
BibMon source. Canonical integration spec lives in the BibMon repo:

- `doc/SSOI_BIBMON.md`
- `specs/001-ssoi-outage-integration/`
- `doc/adr/`

Roadmap in this repo: `doc/PLANO_INTEGRACAO_BIBMON.md`.
Living status (flowcharts, sketches): BibMon `doc/SSOI_PLANO_VIVO.md`.
After each implementation slice, update those two files.

## Cursor chat

Answer the user's questions in clear Portuguese, without jargon. Lead with
the direct answer (yes/no, what happened). When a timeline or a decision is
involved, add a short everyday example. Distinguish "the program worked"
from "the plant data was incomplete". Do not assume the user memorized
internal task IDs, file names, or spec labels.

See `.cursor/rules/chat-respostas-claras.mdc`.

## Git

Read-only git inspection is allowed. Create a local branch when implementing.

Do not run `git commit`, `git push`, merge, rebase, PR, tag, or release without
explicit user authorization for that operation and target.

Do not run `git reset --hard`, `git clean -fd`, or force-push unless the user
explicitly requests it.

The Cursor agent is not a co-author. Never add
`Co-authored-by: Cursor <cursoragent@cursor.com>` to a commit message.

At the end of a task, propose commit commands; do not execute them.
