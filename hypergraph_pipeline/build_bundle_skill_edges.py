#!/usr/bin/env python3
"""Build bundle-skill interaction edges with positive/negative bundle edge types."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bundle-skill interaction edges.")
    parser.add_argument("--pet-data", default="../data")
    parser.add_argument("--bundle-catalog", default="task_bundle_catalog.json")
    parser.add_argument("--task-bundle-interactions", default="task_bundle_interactions.jsonl")
    parser.add_argument("--output-jsonl", default="bundle_skill_interactions.jsonl")
    parser.add_argument("--output-summary", default="bundle_skill_interactions_summary.json")
    args = parser.parse_args()

    data_dir = Path(args.pet_data).resolve()
    bundles = load_json(data_dir / args.bundle_catalog)
    interaction_rows = [
        json.loads(line)
        for line in (data_dir / args.task_bundle_interactions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    interaction_by_bundle = {int(x["bundle_id"]): x for x in interaction_rows}

    edges: list[dict[str, Any]] = []
    for bundle in bundles:
        bid = int(bundle["bundle_id"])
        inter = interaction_by_bundle.get(bid)
        if inter is None:
            raise RuntimeError(f"Missing task-bundle interaction for bundle_id={bid}")
        role = str(bundle["role"])
        if role == "positive":
            edge_type = "positive_bundle_member"
        elif role == "negative":
            edge_type = "negative_bundle_member"
        else:
            edge_type = f"{role}_bundle_member"

        skill_indices = [int(x) for x in bundle["skill_indices"]]
        skill_names = [str(x) for x in bundle["skill_names"]]
        if len(skill_indices) != len(skill_names):
            raise RuntimeError(f"Skill index/name mismatch for bundle_id={bid}")

        changed_by_new: dict[int, list[dict[str, Any]]] = {}
        changed_by_old: dict[int, list[dict[str, Any]]] = {}
        is_missing_bundle = "missing_skill_negative_bundle" in set(str(x) for x in bundle.get("labels", []))
        for c in bundle.get("changed_skills", []) or []:
            # Missing-skill negatives do not connect the missing skill in bundle-skill graph.
            # Their type is represented in the task-bundle graph.
            if is_missing_bundle:
                continue
            if c.get("new_skill_index") is not None:
                changed_by_new.setdefault(int(c["new_skill_index"]), []).append(c)
            if c.get("original_skill_index") is not None:
                changed_by_old.setdefault(int(c["original_skill_index"]), []).append(c)

        for pos, (sidx, sname) in enumerate(zip(skill_indices, skill_names)):
            changed_records = changed_by_new.get(int(sidx), [])
            is_changed = bool(changed_records)
            if is_changed:
                change_types = sorted({str(c.get("change_type", "")) for c in changed_records if c.get("change_type")})
                changed_role = "new_or_added_skill"
            elif int(sidx) in changed_by_old:
                change_types = sorted({str(c.get("change_type", "")) for c in changed_by_old[int(sidx)] if c.get("change_type")})
                changed_role = "original_skill_reference"
            else:
                change_types = []
                changed_role = "unchanged_member"

            edges.append(
                {
                    "bundle_id": bid,
                    "task_index": int(bundle["task_index"]),
                    "task_id": str(bundle["task_id"]),
                    "skill_index": int(sidx),
                    "skill_name": str(sname),
                    "position": pos,
                    "edge_type": edge_type,
                    "bundle_role": role,
                    "use_for_message_passing": True,
                    "negative_labels": list(bundle.get("labels", [])) if role == "negative" else [],
                    "is_changed_skill": is_changed,
                    "changed_role": changed_role,
                    "change_types": change_types,
                    "changed_records": changed_records,
                    "bundle_labels": list(bundle.get("labels", [])),
                    "task_bundle_label": str(inter.get("label", "")),
                }
            )

    out_jsonl = data_dir / args.output_jsonl
    out_summary = data_dir / args.output_summary
    out_jsonl.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in edges) + "\n", encoding="utf-8")

    pos_edges = sum(1 for x in edges if x["edge_type"] == "positive_bundle_member")
    neg_edges = sum(1 for x in edges if x["edge_type"] == "negative_bundle_member")
    unique_skills = len({int(x["skill_index"]) for x in edges})
    changed_edges = sum(1 for x in edges if x["is_changed_skill"])
    neg_label_counts: dict[str, int] = {}
    for x in edges:
        if x["bundle_role"] != "negative":
            continue
        for lab in x.get("negative_labels", []):
            neg_label_counts[lab] = neg_label_counts.get(lab, 0) + 1
    summary = {
        "bundle_count": len(bundles),
        "bundle_skill_edge_count": len(edges),
        "positive_bundle_skill_edges": pos_edges,
        "negative_bundle_skill_edges": neg_edges,
        "changed_skill_edges": changed_edges,
        "unique_skill_count": unique_skills,
        "negative_label_edge_counts": neg_label_counts,
        "edge_type_semantics": {
            "positive_bundle_member": "skill belongs to a gold/positive bundle",
            "negative_bundle_member": "skill belongs to a constructed negative bundle",
        },
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
