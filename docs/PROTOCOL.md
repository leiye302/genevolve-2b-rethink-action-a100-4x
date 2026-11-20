# Unified action: precise scope

This is a protocol ablation relative to native RETHINK commit `b18f659e563d36fc161357da5ddfd65f4266af8f`. It is not the subsequent type-BCE experiment. The original repository is not modified.

The later requested schedule update changes both public variants to **one epoch / 297 updates**. No other hyperparameter or objective is changed by that update. Use a new WORK from the same SFT-550 parent: an old 297-step checkpoint from a two-epoch cosine schedule is not the new one-epoch endpoint, and old 594-step exports are not accepted for this stage-two handoff.

## Mapping

| Native serialization | Command-first serialization | Same effect |
| --- | --- | --- |
| tool_call, name=search, arguments=A | `<action>search A</action>` | text search |
| tool_call, name=image_search, arguments=A | `<action>image_search A</action>` | reference retrieval |
| tool_call, name=query_knowledge, arguments=A | `<action>query_knowledge A</action>` | native skill lookup |
| answer, payload=P | `<action>generate P</action>` | finalize original image program |

Unlike a cosmetic `<action><tool_call>…` wrapper, the first semantic content is the command verb. Unlike a new finish/stop action, generate keeps the complete gen_prompt and reference_images. This aligns the boundary structure with ALFWorld, not its task semantics or naturally shorter arguments.

## One consistent chain

1. The same SFT-550 weights initialize Route-SFT; there is no hidden format-training stage.
2. The sealed pack uses transformed system prompts and historical action syntax, and original current reasoning. There are zero old action-tag mentions in the current reasoning of this pack.
3. Good continues with action. Bad continues with the single marker then action. `route_position = target_start - 1`; all history and reasoning, including `</think>\n`, are masked. No `><`-merged boundary reconstruction is used.
4. RL and evaluation use the same CommandAgent via `make_agent`; generated token IDs are scored unchanged. No post-generation rewriting is used to make a wrong format look successful.
5. Native actions in new rollouts are not accepted as a hidden compatibility fallback. Wrong syntax follows the existing retry/final-round behavior. There is no new requirement enforced by the parser to emit reasoning, and no action-type classifier.
6. Teacher action-span selection follows the new envelope and still excludes reasoning and marker from SDL selection. Every generated token remains in GRPO. The native top-fraction and SDL formula are unchanged.
7. Rendering, reward, reference image handling, search tools, and evaluation split/scoring are reused unchanged. Internal `termination="answer"` is retained for existing render/reward code; it now means a valid `generate` command was parsed.

## The unchanged last-round exception

The pinned GenEvolve loop adds a final-round message requiring immediate final output without `<think>`. We retain that exception, substituting `<action>generate …</action>`. This repository does not claim every forced last-round response has explicit reasoning. Changing that behavior would be a separate experiment.

## Edge cases checked

The pinned native parser can execute a complete literal tool call quoted inside reasoning. Historical serialization therefore includes such examples, preserving which search produced the recorded observation. Current supervised reasoning is not rewritten. One invalid historical envelope remains invalid; no corrected observation is fabricated. Tests compare real native and command-loop outputs with scripted responses for all tools, invalid output, final-step override, exceptions, reference ordering/capping and termination. JSON strings containing action-tag text are parsed without truncating the payload.

## What this experiment cannot establish by itself

Unified boundaries do not guarantee better stopping, stronger marker discrimination or better image quality. The task's final program can still be much longer than a search command. There is no extra type loss or data reweighting to conceal that. Compare the unchanged native method and this action variant with the same parent weights, data semantics, seeds, schedule and evaluation. Report protocol differences explicitly; do not present unadapted native SFT-550 under this new prompt as its original benchmark performance.

## Modules

- `action_protocol.py`: deterministic conversion / command parsing.
- `action_agent.py`: pinned execution loop with format substitutions; tools/finalizer inherited.
- `policy.py`: same actor factory for training/eval; updated SDL spans only.
- `teacher.py`: marker-position telemetry updated; routing/hint/loss unchanged.
- `route_data.py`: converted pack integrity and boundary checks.
- `route_sft.py`: unchanged training/loss implementation; only the export-validation error message now refers to the configured epoch count instead of hard-coding two epochs.
- `objectives.py`, `feedback.py`, `distributed.py`: unchanged training/loss/runtime implementations.
- `scripts/build_action_pack.py`: full-library conversion and equivalence audit.

Hashes and core-method comparisons are provided in `data/`. The test suite and clean public download are CPU evidence; no GPU training or live scoring result is supplied with this protocol release.
