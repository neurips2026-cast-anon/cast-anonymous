#!/usr/bin/env python3
"""Build task-bundle interaction data: gold pairs as 1, negative pairs as 0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def coarse_negative_type(labels: list[str]) -> str:
    label_set = set(labels)
    if "missing_skill_negative_bundle" in label_set:
        return "missing"
    if "zero_interaction_non_gold_bundle" in label_set:
        return "zero_interaction"
    if "relative_top20_higher_risk_bundle" in label_set:
        return "higher_risk"
    if "relative_top20_higher_cost_bundle" in label_set:
        return "higher_cost"
    if "relative_top20_conflict_bundle" in label_set:
        return "conflict"
    if "relative_top20_redundancy_bundle" in label_set:
        return "redundancy"
    if "relative_top20_multi_replace_bundle" in label_set:
        return "multi_replace"
    if "relative_top20_all_replace_bundle" in label_set or "relative_top20_replace_bundle" in label_set:
        return "replace"
    return "other_negative"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task-bundle interaction table.")
    parser.add_argument("--paper-data", default="../../data")
    parser.add_argument("--output-dir", default="../data")
    parser.add_argument("--task-json", default="tasks_all_metadata_for_embedding_v2.json")
    parser.add_argument("--gold-json", default="benchmark_gold_manifest.json")
    parser.add_argument("--negative-json", default="benchmark_negative_bundle_index.json")
    parser.add_argument("--soft-negative-json", default="../neative bundle/soft_negative_bundles_225.json")
    parser.add_argument("--soft-target", type=float, default=0.7)
    parser.add_argument("--output-jsonl", default="task_bundle_interactions.jsonl")
    parser.add_argument("--output-bundles", default="task_bundle_catalog.json")
    parser.add_argument("--output-summary", default="task_bundle_interactions_summary.json")
    args = parser.parse_args()

    data_dir = Path(args.paper_data).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_json(data_dir / args.task_json)
    gold = load_json(data_dir / args.gold_json)
    negatives = load_json(data_dir / args.negative_json)
    soft_path = Path(args.soft_negative_json)
    if not soft_path.is_absolute():
        soft_path = (Path.cwd() / soft_path).resolve()
    soft_negatives = load_json(soft_path) if soft_path.exists() else []
    soft_keys = {
        (int(x["task_index"]), tuple(int(v) for v in x["negative_bundle"]["skill_indices"]))
        for x in soft_negatives
    }

    if len(tasks) != len(gold):
        raise RuntimeError(f"task/gold mismatch: {len(tasks)} vs {len(gold)}")

    bundle_catalog: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []

    def add_bundle(
        *,
        task_index: int,
        task_id: str,
        role: str,
        skill_indices: list[int],
        skill_names: list[str],
        labels: list[str],
        source_negative_record_id: int | None = None,
        changed_skills: list[dict[str, Any]] | None = None,
    ) -> int:
        bundle_id = len(bundle_catalog)
        rec: dict[str, Any] = {
            "bundle_id": bundle_id,
            "task_index": int(task_index),
            "task_id": task_id,
            "role": role,
            "skill_indices": [int(x) for x in skill_indices],
            "skill_names": [str(x) for x in skill_names],
            "bundle_size": len(skill_indices),
            "labels": labels,
        }
        if source_negative_record_id is not None:
            rec["source_negative_record_id"] = int(source_negative_record_id)
        if changed_skills is not None:
            rec["changed_skills"] = changed_skills
        bundle_catalog.append(rec)
        return bundle_id

    # Positive interactions: task-gold bundle = 1.
    for i, (task, row) in enumerate(zip(tasks, gold)):
        task_id = str(task["task_id"])
        if int(row["task_index"]) != i or str(row["task_id"]) != task_id:
            raise RuntimeError(f"gold alignment mismatch at {i}")
        bid = add_bundle(
            task_index=i,
            task_id=task_id,
            role="positive",
            skill_indices=[int(x) for x in row["gold_merged_skill_indices"]],
            skill_names=[str(x) for x in row["gold_local_skill_names"]],
            labels=["gold_bundle"],
        )
        interactions.append(
            {
                "task_index": i,
                "task_id": task_id,
                "bundle_id": bid,
                "interaction": 1,
                "interaction_kind": "positive",
                "use_for_message_passing": True,
                "label": "gold_bundle",
            }
        )

    # Negative interactions: task-negative bundle = 0.
    for rid, row in enumerate(negatives):
        ti = int(row["task_index"])
        if not (0 <= ti < len(tasks)):
            raise RuntimeError(f"negative task_index out of range at {rid}: {ti}")
        task_id = str(row["task_id"])
        if task_id != str(tasks[ti]["task_id"]):
            raise RuntimeError(f"negative task_id mismatch at {rid}: {task_id} vs {tasks[ti]['task_id']}")
        labels = [str(x) for x in row.get("negative_labels", [])]
        if not labels:
            labels = [str(row.get("negative_label_primary", "negative_bundle"))]
        nidx = [int(x) for x in row["negative_bundle"]["skill_indices"]]
        neg_type = coarse_negative_type(labels)
        bid = add_bundle(
            task_index=ti,
            task_id=task_id,
            role="negative",
            skill_indices=nidx,
            skill_names=[str(x) for x in row["negative_bundle"]["skill_names"]],
            labels=labels,
            source_negative_record_id=rid,
            changed_skills=list(row["negative_bundle"].get("changed_skills", [])),
        )
        nkey = (ti, tuple(int(x) for x in nidx))
        interaction_value = float(args.soft_target) if nkey in soft_keys else 0.0
        interaction_kind = "soft_negative" if nkey in soft_keys else "hard_negative"
        interactions.append(
            {
                "task_index": ti,
                "task_id": task_id,
                "bundle_id": bid,
                "interaction": interaction_value,
                "interaction_kind": interaction_kind,
                "use_for_message_passing": False,
                "negative_type": neg_type,
                "label": row.get("negative_label_primary", labels[0]),
                "labels": labels,
                "source_negative_record_id": rid,
            }
        )

    out_jsonl = out_dir / args.output_jsonl
    out_bundles = out_dir / args.output_bundles
    out_summary = out_dir / args.output_summary

    out_jsonl.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in interactions) + "\n", encoding="utf-8")
    out_bundles.write_text(json.dumps(bundle_catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    pos = sum(1 for x in interactions if int(x["interaction"]) == 1)
    soft = sum(1 for x in interactions if float(x["interaction"]) == float(args.soft_target))
    neg = sum(1 for x in interactions if float(x["interaction"]) == 0.0)
    negative_type_counts: dict[str, int] = {}
    for x in interactions:
        if int(x.get("interaction", -1)) == 1:
            continue
        nt = str(x.get("negative_type", ""))
        negative_type_counts[nt] = negative_type_counts.get(nt, 0) + 1
    summary = {
        "task_count": len(tasks),
        "bundle_count": len(bundle_catalog),
        "interaction_count": len(interactions),
        "positive_interactions": pos,
        "soft_negative_interactions": soft,
        "hard_negative_interactions": neg,
        "negative_type_counts": negative_type_counts,
        "soft_target": float(args.soft_target),
        "matrix_semantics": "A_task_bundle[t,b]=1 for gold bundle, 0.7 for close soft negative bundle, 0 for hard/easy negative bundle; unobserved pairs are omitted.",
        "output_jsonl": str(out_jsonl),
        "output_bundle_catalog": str(out_bundles),
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
