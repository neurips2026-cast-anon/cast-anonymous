#!/usr/bin/env python3
"""Split PET graph data into train/val/test task-induced subgraphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")


def split_one(
    *,
    split_name: str,
    task_indices: set[int],
    base_dir: Path,
    output_root: Path,
    task_nodes: list[dict[str, Any]],
    skill_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    out_dir = output_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles_all = load_json(base_dir / "bundle_nodes.json")
    tb_all = read_jsonl(base_dir / "task_bundle_interactions.jsonl")
    ts_all = read_jsonl(base_dir / "task_skill_interactions.jsonl")
    bs_all = read_jsonl(base_dir / "bundle_skill_interactions.jsonl")

    tb = [x for x in tb_all if int(x["task_index"]) in task_indices]
    bundle_ids = {int(x["bundle_id"]) for x in tb}
    bundles = [x for x in bundles_all if int(x["bundle_id"]) in bundle_ids]
    ts = [x for x in ts_all if int(x["task_index"]) in task_indices]
    bs = [x for x in bs_all if int(x["bundle_id"]) in bundle_ids]
    tasks = [x for x in task_nodes if int(x["task_index"]) in task_indices]

    used_skill_ids = {int(x["skill_index"]) for x in ts}
    used_skill_ids.update(int(x["skill_index"]) for x in bs)
    skills = [x for x in skill_nodes if int(x["skill_index"]) in used_skill_ids or split_name == "train_full_skill_universe"]

    (out_dir / "task_nodes.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "bundle_nodes.json").write_text(json.dumps(bundles, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "skill_nodes.json").write_text(json.dumps(skills, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "task_bundle_interactions.jsonl", tb)
    write_jsonl(out_dir / "task_skill_interactions.jsonl", ts)
    write_jsonl(out_dir / "bundle_skill_interactions.jsonl", bs)

    summary = {
        "split": split_name,
        "task_count": len(tasks),
        "bundle_count": len(bundles),
        "task_bundle_edge_count": len(tb),
        "task_skill_edge_count": len(ts),
        "bundle_skill_edge_count": len(bs),
        "skill_node_count": len(skills),
        "positive_task_bundle_edges": sum(1 for x in tb if float(x["interaction"]) == 1.0),
        "soft_negative_task_bundle_edges": sum(1 for x in tb if abs(float(x["interaction"]) - 0.7) < 1e-9),
        "hard_negative_task_bundle_edges": sum(1 for x in tb if float(x["interaction"]) == 0.0),
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Split PET graph data by task split.")
    parser.add_argument("--pet-data", default="../data")
    parser.add_argument("--paper-data", default="../../data")
    parser.add_argument("--split-json", default="benchmark_train_test_split.json")
    parser.add_argument("--output-root", default="../data/splits")
    args = parser.parse_args()

    pet_data = Path(args.pet_data).resolve()
    paper_data = Path(args.paper_data).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    split = load_json(paper_data / args.split_json)
    train_set = set(int(x) for x in split.get("train_task_indices", []))
    val_set = set(int(x) for x in split.get("val_task_indices", []))
    test_set = set(int(x) for x in split.get("test_task_indices", []))
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Split sets overlap")

    task_nodes = load_json(pet_data / "task_nodes.json")
    skill_nodes = load_json(pet_data / "skill_nodes.json")

    summaries = {
        "train": split_one(
            split_name="train",
            task_indices=train_set,
            base_dir=pet_data,
            output_root=output_root,
            task_nodes=task_nodes,
            skill_nodes=skill_nodes,
        ),
        "val": split_one(
            split_name="val",
            task_indices=val_set,
            base_dir=pet_data,
            output_root=output_root,
            task_nodes=task_nodes,
            skill_nodes=skill_nodes,
        ),
        "test": split_one(
            split_name="test",
            task_indices=test_set,
            base_dir=pet_data,
            output_root=output_root,
            task_nodes=task_nodes,
            skill_nodes=skill_nodes,
        ),
    }
    (output_root / "split_graph_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
