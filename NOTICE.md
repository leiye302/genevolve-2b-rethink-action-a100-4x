# Source attribution

This independent protocol experiment derives from the RETHINK adaptation at
`leiye302/genevolve-2b-rethink-a100-4x`, commit
`b18f659e563d36fc161357da5ddfd65f4266af8f`.
It changes action serialization and its aligned data/runtime interfaces, not
the original Route-SFT or GenEvolve learning objectives.

`genevolve_repro/action_agent.py` adapts the execution loop from
MeiGen-AI/GenEvolve, commit `23c847c559ccc0f95bbf4b3d8925898463822f4c`.
Copyright 2026 The GenEvolve Authors. Licensed under Apache License 2.0;
the upstream notice is retained in `LICENSES/GenEvolve-Apache-2.0-notice.txt`.
Changes are documented in `docs/PROTOCOL.md`; the remainder of upstream code
is downloaded unmodified at its pinned revision and retains its license.

The inherited stage-1 MIT notice is retained in `LICENSES/stage1-MIT.txt`.
Other upstream repositories, model weights, datasets and retrieved web content
retain their own applicable licenses. This notice does not relicense external
models or source material. See `sources.json` and `DATASET.md` for provenance.
