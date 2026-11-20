# Command-first Route-SFT training pack

[Public download](https://github.com/leiye302/genevolve-2b-rethink-action-a100-4x/releases/tag/route-sft-command-first-v1), or:

```bash
python -m pip install filelock==3.20.0
python run.py --work "$WORK" prepare-route
```

The archive has exactly three files: `manifest.json`, `train.jsonl`, `batches.jsonl`. Sizes and SHA-256 are pinned in [configs/route_release.json](configs/route_release.json). The complete training input, masks, targets, route positions and batch membership are public; no private annotations or external local files are required. GPU training additionally requires the declared 2B SFT-550 parent and dependencies.

## Composition and unchanged selection

2,376 decisions: 1,782 Good + 594 Bad (394 natural, 200 synthetic same-state contrasts), 297 fixed batches of 6:2, two epochs / 594 updates. The paired synthetic Bad and its source Good stay in the same batch. There is no additional validation split.

| Command | Good | Bad | Total |
| --- | ---: | ---: | ---: |
| search | 210 | 219 | 429 |
| image_search | 650 | 219 | 869 |
| query_knowledge | 430 | 27 | 457 |
| generate | 492 | 129 | 621 |

Good is the original naturally sampled action. Bad is the already-reviewed corrected action, with `<RETHINK>` before the common `<action>` boundary. No additional correction, selection, balancing or synthetic examples were introduced for this protocol experiment. Labels mean a definite uncorrected reasoning error visible in the original actor context; poor outcomes alone are not a Bad label.

## What was converted

The source is the existing [reviewed-v3 public archive](https://github.com/leiye302/genevolve-2b-rethink-a100-4x/releases/tag/route-sft-reviewed-v3-public), pinned by [native_route_release.json](configs/native_route_release.json). It was already sanitized for public release. The collection model was 8B; this repository trains a matched 2B SFT-550, verifying its actual tokenizer before training.

- `<tool_call>{"name":"search","arguments":ARGS}</tool_call>` becomes `<action>search ARGS</action>`; analogous mapping for the other two tools.
- `<answer>PROGRAM</answer>` becomes `<action>generate PROGRAM</action>`.
- Argument and final-program JSON source strings are retained, not rewritten or shortened. Current reasoning text and token IDs are identical.
- System instructions and protocol-bearing historical assistant responses use the same mapping. Historical action examples inside reasoning are also converted: the pinned native executor searches the whole response, so omitting them could change which action the recorded observation belongs to.
- Other user requests and tool observations are unchanged. The one exact format-error message uses the new format. One malformed historical JSON action stays malformed and retains its real error feedback; it is not a supervised malformed target.
- Only changed history and action suffixes are encoded again; current reasoning IDs, row IDs, labels, task/source provenance, pair links and batch file are unchanged. All 2,376 target action meanings and 6,091 historical assistant action meanings were checked against the native parser.
- All online chat-template prefixes and masked separators were verified. Longest input is 11,078 tokens; the unchanged 12,288 cap rejects overflow, never truncates.

Source metadata `source_action_kind` / `target_action_kind` retains the original `tool` / `answer` provenance; executable targets use command-first syntax (`answer` maps to `generate`). `source_visible_sha256` is a source-state provenance link, not a digest of the reformatted history.

The manifest records both native and converted hashes. The old native pack and its checkpoints are rejected here; use a new WORK. This archive contains no model weights, images, raw episodes, API keys or private annotation logs. Official RL/evaluation splits remain separately pinned and fetched via `prepare`.

## Rebuild / audit

The released pack is ready to train; rebuilding is optional. To reproduce conversion, obtain the public source archive and matching tokenizer files (the three exact hashes are in [protocol_audit.json](data/protocol_audit.json)). A matching 2B parent tokenizer may be used when the hashes match. Conversion fails rather than guessing.

```bash
bash scripts/setup.sh --sources-only
python scripts/build_action_pack.py --source /path/to/native-public-pack \
  --tokenizer /path/to/matching-tokenizer --output /path/to/new-output/pack
```

This generates a new deterministic ZIP, lock and audit report. It does not update labels, download weights, use an API or overwrite a sealed pack. A training run still validates against the actual 2B parent's tokenizer.

Derived decisions are not an author-provided GenEvolve annotation dataset. Original benchmark/data provenance and licenses remain those of the pinned [GenEvolve](https://github.com/MeiGen-AI/GenEvolve/tree/23c847c559ccc0f95bbf4b3d8925898463822f4c) and [GenEvolve-Data-Bench](https://huggingface.co/datasets/MeiGen-AI/GenEvolve-Data-Bench/tree/9f291879d5dfe1195f47883e23069d3f6e829afe) releases. External retrieved text may have its own rights; this release does not grant ownership of third-party source material.
