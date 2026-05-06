#!/usr/bin/env python3
"""Build PET node tables for tasks, bundles, and the full skill universe."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PET node tables.")
    parser.add_argument("--paper-data", default="../../data")
    parser.add_argument("--pet-data", default="../data")
    parser.add_argument("--output-dir", default="../data")
    args = parser.parse_args()

    paper_data = Path(args.paper_data).resolve()
    pet_data = Path(args.pet_data).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_json(paper_data / "tasks_all_metadata_for_embedding_v2.json")
    skills = load_json(paper_data / "benchmark_merged_skills.json")
    bundles = load_json(pet_data / "task_bundle_catalog.json")
    task_skill_edges = read_jsonl(pet_data / "task_skill_interactions.jsonl")
    bundle_skill_edges = read_jsonl(pet_data / "bundle_skill_interactions.jsonl")

    ts_degree = Counter(int(e["skill_index"]) for e in task_skill_edges)
    bs_degree = Counter(int(e["skill_index"]) for e in bundle_skill_edges)
    gold_skill = {
        int(e["skill_index"])
        for e in task_skill_edges
        if e.get("edge_type") == "task_gold_skill"
    }
    top20_skill = {
        int(e["skill_index"])
        for e in task_skill_edges
        if e.get("edge_type") == "task_top20_skill"
    }
    changed_skill = {
        int(e["skill_index"])
        for e in task_skill_edges
        if e.get("edge_type") == "task_negative_changed_skill"
    }

    task_nodes = []
    for i, t in enumerate(tasks):
        task_nodes.append(
            {
                "task_index": i,
                "task_id": str(t.get("task_id")),
                "domain": t.get("domain"),
                "id": t.get("id"),
            }
        )

    bundle_nodes = []
    for b in bundles:
        bundle_nodes.append(
            {
                "bundle_id": int(b["bundle_id"]),
                "task_index": int(b["task_index"]),
                "task_id": str(b["task_id"]),
                "role": str(b["role"]),
                "bundle_size": int(b["bundle_size"]),
                "labels": list(b.get("labels", [])),
                "skill_indices": [int(x) for x in b["skill_indices"]],
            }
        )

    skill_nodes = []
    for i, s in enumerate(skills):
        deg_ts = int(ts_degree.get(i, 0))
        deg_bs = int(bs_degree.get(i, 0))
        skill_nodes.append(
            {
                "skill_index": i,
                "skill_name": str(s.get("name") or f"skill_{i}"),
                "description": str(s.get("description") or ""),
                "permission_risk_level": s.get("permission_risk_level"),
                "permission_risk_value": s.get("permission_risk_value"),
                "token_cost_level": s.get("token_cost_level"),
                "token_cost_value": s.get("token_cost_value"),
                "has_task_skill_edge": deg_ts > 0,
                "has_bundle_skill_edge": deg_bs > 0,
                "has_any_graph_edge": (deg_ts + deg_bs) > 0,
                "task_skill_degree": deg_ts,
                "bundle_skill_degree": deg_bs,
                "is_gold_skill": i in gold_skill,
                "is_top20_candidate_skill": i in top20_skill,
                "is_negative_changed_skill": i in changed_skill,
                "cold_start_skill": (deg_ts + deg_bs) == 0,
            }
        )

    out_task = out_dir / "task_nodes.json"
    out_bundle = out_dir / "bundle_nodes.json"
    out_skill = out_dir / "skill_nodes.json"
    out_task.write_text(json.dumps(task_nodes, ensure_ascii=False, indent=2), encoding="utf-8")
    out_bundle.write_text(json.dumps(bundle_nodes, ensure_ascii=False, indent=2), encoding="utf-8")
    out_skill.write_text(json.dumps(skill_nodes, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "task_node_count": len(task_nodes),
        "bundle_node_count": len(bundle_nodes),
        "skill_node_count_full_universe": len(skill_nodes),
        "skill_with_any_graph_edge": sum(1 for x in skill_nodes if x["has_any_graph_edge"]),
        "cold_start_skill_count": sum(1 for x in skill_nodes if x["cold_start_skill"]),
        "gold_skill_count": len(gold_skill),
        "top20_candidate_skill_count": len(top20_skill),
        "negative_changed_skill_count": len(changed_skill),
    }
    (out_dir / "pet_node_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
