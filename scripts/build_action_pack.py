"""Rebuild the published pack by serialization only; no API or relabeling.

Usage: python scripts/build_action_pack.py --source NATIVE_PACK
       --tokenizer TOKENIZER_DIRECTORY --output NEW_DIRECTORY
The original public pack hashes and the tokenizer are checked before writing.
Generated lock/report files are outside the three-member training archive.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genevolve_repro.action_protocol import (
    ACTION_PROTOCOL, NATIVE_BLOCK, commands, convert_messages, native_to_command,
    parse_generate, parse_tool,
)
from genevolve_repro.actor_text import ACTOR_TEXT_PROTOCOL
from genevolve_repro.common import ROOT, jsonl, read_json
from genevolve_repro.route_data import EOS_ID, MARKER_ID, register_marker, verify_tokenizer
from genevolve_repro.sft_export import file_sha256


def write_json(path, value):
    # Stable archives across Windows/Linux as well as repeated invocations.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def native_semantics(text):
    match = NATIVE_BLOCK.search(text)
    if match is None:
        raise ValueError("Expected one complete native action")
    obj = json.loads(match["body"])
    return ("generate", obj) if match["tag"] == "answer" else (obj["name"], obj["arguments"])


def convert_row(row, tokenizer):
    messages = convert_messages(row["messages"])
    from genevolve.agent import _parse_answer, _parse_tool_call
    for before, after in zip(row["messages"], messages):
        if before["role"] == "assistant":
            if (_parse_answer(before["content"]) != parse_generate(after["content"])
                    or _parse_tool_call(before["content"]) != parse_tool(after["content"])):
                raise ValueError("Historical action semantics changed: " + row["id"])
    target = native_to_command(row["target"])
    parsed = list(commands(target))
    if (len(parsed) != 1 or parsed[0].start != len("<RETHINK>") * row["route_label"]
            or parsed[0].end != len(target)
            or (parsed[0].name, parsed[0].arguments) != native_semantics(row["target"])):
        raise ValueError("Target semantics/format changed: " + row["id"])
    history_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # Keep naturally generated reasoning IDs verbatim, including noncanonical
    # BPE sequences. Only H and the genuinely reformatted action are encoded.
    reason_ids = row["input_ids"][row["history_token_count"]:row["target_start"]]
    prefix = history_ids + reason_ids
    suffix = tokenizer.encode(target, add_special_tokens=False) + [EOS_ID]
    if suffix[0] != (MARKER_ID if row["route_label"] else tokenizer.encode("<action>", add_special_tokens=False)[0]):
        raise ValueError("Incorrect causal route boundary")
    ids = prefix + suffix
    if len(ids) > 12288:
        raise ValueError("Unchanged 12288-token budget exceeded: " + row["id"])
    out = {**row, "messages": messages, "target": target, "input_ids": ids,
           "labels": [-100] * len(prefix) + suffix, "target_start": len(prefix),
           "route_position": len(prefix) - 1, "history_token_count": len(history_ids)}
    if out["assistant_prefix"] != row["assistant_prefix"]:
        raise AssertionError("Reasoning changed")
    # All metadata, selection, labels, pair links and source provenance stay put.
    changed = {k for k in row if row[k] != out[k]}
    if not changed <= {"messages", "target", "input_ids", "labels", "target_start",
                       "route_position", "history_token_count"}:
        raise AssertionError("Unexpected sample change")
    return out


def build(source, tokenizer_folder, output):
    from transformers import AutoTokenizer

    spec = read_json(ROOT / "configs" / "native_route_release.json")
    for name, lock in spec["files"].items():
        p = source / name
        if p.stat().st_size != lock["bytes"] or file_sha256(p) != lock["sha256"]:
            raise ValueError("Native source does not match public reviewed v3: " + name)
    # The released 8B collection and matched 2B parent use the same text
    # tokenizer. The runtime still verifies the actual training parent.
    expected = read_json(ROOT / "configs" / "route_data_source_8b_sha256.json")
    for name in ("tokenizer.json", "chat_template.jinja", "tokenizer_config.json"):
        if file_sha256(tokenizer_folder / name) != expected[name]:
            raise ValueError("Unexpected tokenizer asset: " + name)
    tokenizer = register_marker(AutoTokenizer.from_pretrained(tokenizer_folder, local_files_only=True))
    if output.exists():
        raise ValueError("Use a new output directory; never overwrite a sealed pack")
    output.mkdir(parents=True)
    rows, native = [], []
    for index, row in enumerate(jsonl(source / "train.jsonl")):
        native.append(row)
        try:
            rows.append(convert_row(row, tokenizer))
        except Exception as exc:
            raise ValueError(f"Conversion failed at row {index}: {row['id']}") from exc
    verify_tokenizer(rows, tokenizer)
    with (output / "train.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
    shutil.copyfile(source / "batches.jsonl", output / "batches.jsonl")
    counts = Counter(("bad" if r["route_label"] else "good") + ":" + next(commands(r["target"])).name for r in rows)
    manifest = read_json(source / "manifest.json")
    manifest.update(
        schema="genevolve-route-sft-6g2b-action-v1", action_protocol=ACTION_PROTOCOL,
        actor_text_protocol=ACTOR_TEXT_PROTOCOL, release_variant="reviewed-v3-command-first-public",
        native_manifest_sha256=spec["files"]["manifest.json"]["sha256"],
        native_artifact_sha256={n: spec["files"][n]["sha256"] for n in ("train.jsonl", "batches.jsonl")},
        retokenized_targets=True, supervision_identical_to_reviewed_v3=False,
        supervision_action_semantics_unchanged=True, reasoning_text_and_tokens_unchanged=True,
        labels_ids_and_batch_order_unchanged=True,
        artifact_sha256={n: file_sha256(output / n) for n in ("train.jsonl", "batches.jsonl")},
    )
    write_json(output / "manifest.json", manifest)
    report = {
        "schema": "command-first-conversion-audit-v1", "rows": len(rows), "batches": 297,
        "action_counts": dict(sorted(counts.items())),
        "reasoning_text_and_tokens_unchanged": True, "target_semantics_checked": len(rows),
        "history_actions_semantics_checked": sum(m["role"] == "assistant" for r in rows for m in r["messages"]),
        "historical_format_errors_preserved": sum(
            m["role"] == "assistant" and "<action>" in m["content"] and not list(commands(m["content"]))
            for r in rows for m in r["messages"]),
        "all_online_prefixes_verified": True, "sample_ids_labels_order_and_metadata_unchanged": True,
        "batch_file_byte_identical": (output / "batches.jsonl").read_bytes() == (source / "batches.jsonl").read_bytes(),
        "max_tokens_before": max(len(r["input_ids"]) for r in native),
        "max_tokens_after": max(len(r["input_ids"]) for r in rows),
        "current_reasoning_native_tag_mentions": sum(any(t in r["assistant_prefix"] for t in ("<tool_call>", "<answer>")) for r in rows),
        "native_manifest_sha256": spec["files"]["manifest.json"]["sha256"],
        "converted_manifest_sha256": file_sha256(output / "manifest.json"),
        "tokenizer_sha256": {n: expected[n] for n in ("tokenizer.json", "chat_template.jinja", "tokenizer_config.json")},
    }
    archive = output.parent / "genevolve-route-sft-command-first-v1.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in ("manifest.json", "train.jsonl", "batches.jsonl"):
            info = ZipInfo(name, date_time=(2026, 11, 20, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, (output / name).read_bytes())
    lock = {"tag": "route-sft-command-first-v1", "name": archive.name,
            "bytes": archive.stat().st_size, "sha256": file_sha256(archive),
            "files": {n: {"bytes": (output / n).stat().st_size, "sha256": file_sha256(output / n)}
                      for n in ("manifest.json", "train.jsonl", "batches.jsonl")}}
    write_json(output.parent / "route_release.json", lock)
    write_json(output.parent / "protocol_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build(args.source, args.tokenizer, args.output)
