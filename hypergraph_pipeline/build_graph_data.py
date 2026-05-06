#!/usr/bin/env python3
"""Build PET-style graph data from task/gold/negative bundle records.

Views:
- task-bundle edges: task -> gold/negative bundle
- task-skill edges: task -> positive/negative skill
- bundle-skill edges: bundle -> member skill
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_bundle_key(skill_indices: list[int]) -> str:
    return ",".join(str(int(x)) for x in skill_indices)


def add_bundle(
    catalog: dict[tuple[int, str, str], int],
    bundles: list[dict[str, Any]],
    *,
    task_index: int,
    task_id: str,
    bundle_role: str,
    skill_indices: list[int],
    skill_names: list[str],
    labels: list[str],
    source_negative_record_id: int | None = None,
) -> int:
    key = (int(task_index), bundle_role, make_bundle_key(skill_indices))
    if key in catalog:
        bid = catalog[key]
        existing = bundles[bid]
        for label in labels:
            if label not in existing["labels"]:
                existing["labels"].append(label)
        if source_negative_record_id is not None:
            existing.setdefault("source_negative_record_ids", []).append(int(source_negative_record_id))
        return bid

    bid = len(bundles)
    catalog[key] = bid
    rec = {
        "bundle_id": bid,
        "task_index": int(task_index),
        "task_id": task_id,
        "bundle_role": bundle_role,
        "skill_indices": [int(x) for x in skill_indices],
        "skill_names": [str(x) for x in skill_names],
        "bundle_size": len(skill_indices),
        "labels": list(labels),
    }
    if source_negative_record_id is not None:
        rec["source_negative_record_ids"] = [int(source_negative_record_id)]
    bundles.append(rec)
    return bid


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PET-style graph data.")
    parser.add_argument("--paper-data", default="../../data")
    parser.add_argument("--output-dir", default="../data")
    parser.add_argument("--task-json", default="tasks_all_metadata_for_embedding_v2.json")
    parser.add_argument("--gold-json", default="benchmark_gold_manifest.json")
    parser.add_argument("--negative-json", default="benchmark_negative_bundle_index.json")
    parser.add_argument("--skill-json", default="benchmark_merged_skills.json")
    args = parser.parse_args()

    data_dir = Path(args.paper_data).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_json(data_dir / args.task_json)
    gold = load_json(data_dir / args.gold_json)
    negatives = load_json(data_dir / args.negative_json)
    skills = load_json(data_dir / args.skill_json)

    if len(tasks) != len(gold):
        raise RuntimeError(f"task/gold mismatch: {len(tasks)} vs {len(gold)}")

    bundles: list[dict[str, Any]] = []
    bundle_catalog: dict[tuple[int, str, str], int] = {}
    task_bundle_edges: list[dict[str, Any]] = []
    task_skill_edges: list[dict[str, Any]] = []
    bundle_skill_edges: list[dict[str, Any]] = []

    # Gold bundles and positive task-skill edges.
    for i, (task, grow) in enumerate(zip(tasks, gold)):
        task_id = str(task.get("task_id"))
        if int(grow["task_index"]) != i or str(grow["task_id"]) != task_id:
            raise RuntimeError(f"gold alignment mismatch at {i}")
        gold_indices = [int(x) for x in grow["gold_merged_skill_indices"]]
        gold_names = [str(x) for x in grow["gold_local_skill_names"]]
        if len(gold_indices) != len(gold_names):
            raise RuntimeError(f"gold skill name/index mismatch at {i}")

        bid = add_bundle(
            bundle_catalog,
            bundles,
            task_index=i,
            task_id=task_id,
            bundle_role="positive",
            skill_indices=gold_indices,
            skill_names=gold_names,
            labels=["gold_bundle"],
        )
        task_bundle_edges.append(
            {
                "task_index": i,
                "task_id": task_id,
                "bundle_id": bid,
                "edge_label": "positive",
                "target": 1,
                "labels": ["gold_bundle"],
            }
        )
        for pos, (sidx, sname) in enumerate(zip(gold_indices, gold_names)):
            task_skill_edges.append(
                {
                    "task_index": i,
                    "task_id": task_id,
                    "skill_index": int(sidx),
                    "skill_name": str(sname),
                    "edge_label": "positive",
                    "target": 1,
                    "source": "gold_bundle",
                    "position": pos,
                }
            )
        for pos, (sidx, sname) in enumerate(zip(gold_indices, gold_names)):
            bundle_skill_edges.append(
                {
                    "bundle_id": bid,
                    "task_index": i,
                    "task_id": task_id,
                    "skill_index": int(sidx),
                    "skill_name": str(sname),
                    "edge_label": "member",
                    "bundle_role": "positive",
                    "position": pos,
                }
            )

    # Negative bundles and negative task/bundle/skill edges.
    for neg_id, nrow in enumerate(negatives):
        ti = int(nrow["task_index"])
        if not (0 <= ti < len(tasks)):
            raise RuntimeError(f"negative task index out of range: row={neg_id} ti={ti}")
        task_id = str(nrow["task_id"])
        if task_id != str(tasks[ti].get("task_id")):
            raise RuntimeError(f"negative task id mismatch: row={neg_id}")

        nidx = [int(x) for x in nrow["negative_bundle"]["skill_indices"]]
        nnames = [str(x) for x in nrow["negative_bundle"]["skill_names"]]
        labels = [str(x) for x in nrow.get("negative_labels", [nrow.get("negative_label_primary", "negative")])]
        primary = str(nrow.get("negative_label_primary", labels[0] if labels else "negative"))

        bid = add_bundle(
            bundle_catalog,
            bundles,
            task_index=ti,
            task_id=task_id,
            bundle_role="negative",
            skill_indices=nidx,
            skill_names=nnames,
            labels=labels,
            source_negative_record_id=neg_id,
        )
        task_bundle_edges.append(
            {
                "task_index": ti,
                "task_id": task_id,
                "bundle_id": bid,
                "edge_label": "negative",
                "target": 0,
                "labels": labels,
                "primary_label": primary,
                "source_negative_record_id": neg_id,
            }
        )
        changed_indices = {
            int(c.get("new_skill_index", c.get("original_skill_index", -1)))
            for c in nrow["negative_bundle"].get("changed_skills", [])
            if c.get("new_skill_index", c.get("original_skill_index", None)) is not None
        }
        for pos, (sidx, sname) in enumerate(zip(nidx, nnames)):
            bundle_skill_edges.append(
                {
                    "bundle_id": bid,
                    "task_index": ti,
                    "task_id": task_id,
                    "skill_index": int(sidx),
                    "skill_name": str(sname),
                    "edge_label": "member",
                    "bundle_role": "negative",
                    "position": pos,
                    "negative_labels": labels,
                    "is_changed_skill": int(sidx) in changed_indices,
                }
            )
            if int(sidx) in changed_indices:
                task_skill_edges.append(
                    {
                        "task_index": ti,
                        "task_id": task_id,
                        "skill_index": int(sidx),
                        "skill_name": str(sname),
                        "edge_label": "negative",
                        "target": 0,
                        "source": "negative_bundle_changed_skill",
                        "negative_labels": labels,
                    }
                )

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")

    (output_dir / "pet_bundle_catalog.json").write_text(json.dumps(bundles, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "pet_task_bundle_edges.jsonl", task_bundle_edges)
    write_jsonl(output_dir / "pet_task_skill_edges.jsonl", task_skill_edges)
    write_jsonl(output_dir / "pet_bundle_skill_edges.jsonl", bundle_skill_edges)

    summary = {
        "task_count": len(tasks),
        "skill_count": len(skills),
        "gold_bundle_count": len(gold),
        "negative_bundle_count": len(negatives),
        "bundle_catalog_count": len(bundles),
        "task_bundle_edge_count": len(task_bundle_edges),
        "task_skill_edge_count": len(task_skill_edges),
        "bundle_skill_edge_count": len(bundle_skill_edges),
        "positive_task_bundle_edges": sum(1 for x in task_bundle_edges if x["edge_label"] == "positive"),
        "negative_task_bundle_edges": sum(1 for x in task_bundle_edges if x["edge_label"] == "negative"),
        "positive_task_skill_edges": sum(1 for x in task_skill_edges if x["edge_label"] == "positive"),
        "negative_task_skill_edges": sum(1 for x in task_skill_edges if x["edge_label"] == "negative"),
    }
    (output_dir / "pet_graph_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
