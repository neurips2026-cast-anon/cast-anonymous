#!/usr/bin/env python3
"""Build task-skill interaction edges: gold, top-k retrieved, and negative changed skills."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def l2norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def skill_name(skills: list[dict[str, Any]], idx: int) -> str:
    if 0 <= idx < len(skills):
        return str(skills[idx].get("name") or f"skill_{idx}")
    return f"skill_{idx}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task-skill interaction graph.")
    parser.add_argument("--paper-data", default="../../data")
    parser.add_argument("--output-dir", default="../data")
    parser.add_argument("--task-json", default="tasks_all_metadata_for_embedding_v2.json")
    parser.add_argument("--gold-json", default="benchmark_gold_manifest.json")
    parser.add_argument("--negative-json", default="benchmark_negative_bundle_index.json")
    parser.add_argument("--task-emb", default="task_embeddings_bigmodel_512_f32_v2.npy")
    parser.add_argument("--skill-json", default="benchmark_merged_skills.json")
    parser.add_argument("--skill-emb", default="benchmark_merged_skill_embeddings.npy")
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--output-jsonl", default="task_skill_interactions.jsonl")
    parser.add_argument("--output-summary", default="task_skill_interactions_summary.json")
    args = parser.parse_args()

    data_dir = Path(args.paper_data).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = json.loads((data_dir / args.task_json).read_text(encoding="utf-8"))
    gold = json.loads((data_dir / args.gold_json).read_text(encoding="utf-8"))
    negatives = json.loads((data_dir / args.negative_json).read_text(encoding="utf-8"))
    skills = json.loads((data_dir / args.skill_json).read_text(encoding="utf-8"))
    task_emb = l2norm_rows(np.load(data_dir / args.task_emb).astype(np.float32, copy=False))
    skill_emb = l2norm_rows(np.load(data_dir / args.skill_emb).astype(np.float32, copy=False))

    if len(tasks) != len(gold) or task_emb.shape[0] != len(tasks):
        raise RuntimeError(f"task/gold/embedding mismatch: {len(tasks)} {len(gold)} {task_emb.shape}")
    if skill_emb.shape[0] != len(skills):
        raise RuntimeError(f"skill embedding mismatch: {len(skills)} {skill_emb.shape}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    gold_by_task: dict[int, set[int]] = {}

    # Strong positive edges from gold bundle.
    for i, (task, grow) in enumerate(zip(tasks, gold)):
        task_id = str(task["task_id"])
        if int(grow["task_index"]) != i or str(grow["task_id"]) != task_id:
            raise RuntimeError(f"gold alignment mismatch at {i}")
        gold_set = set(int(x) for x in grow["gold_merged_skill_indices"])
        gold_by_task[i] = gold_set
        for pos, sidx in enumerate(grow["gold_merged_skill_indices"]):
            sidx = int(sidx)
            key = (i, sidx, "task_gold_skill")
            seen.add(key)
            rows.append(
                {
                    "task_index": i,
                    "task_id": task_id,
                    "skill_index": sidx,
                    "skill_name": skill_name(skills, sidx),
                    "edge_type": "task_gold_skill",
                    "target": 1.0,
                    "use_for_message_passing": True,
                    "is_gold_skill": True,
                    "gold_position": pos,
                }
            )

    # Retrieved candidate edges from top-N task-skill similarity.
    for i, task in enumerate(tasks):
        task_id = str(task["task_id"])
        scores = skill_emb @ task_emb[i]
        topn = min(args.topn, scores.shape[0])
        idx = np.argpartition(-scores, topn - 1)[:topn]
        idx = idx[np.argsort(-scores[idx])]
        for rank, sidx in enumerate(idx.tolist(), start=1):
            sidx = int(sidx)
            # Do not duplicate gold edges; gold edge is stronger and already present.
            if sidx in gold_by_task[i]:
                continue
            key = (i, sidx, "task_top20_skill")
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "task_index": i,
                    "task_id": task_id,
                    "skill_index": sidx,
                    "skill_name": skill_name(skills, sidx),
                    "edge_type": "task_top20_skill",
                    "target": float(scores[sidx]),
                    "use_for_message_passing": True,
                    "is_gold_skill": False,
                    "retrieval_rank": rank,
                    "task_skill_similarity": float(scores[sidx]),
                }
            )

    # Negative changed-skill edges from negative bundle construction.
    for rid, nrow in enumerate(negatives):
        ti = int(nrow["task_index"])
        task_id = str(nrow["task_id"])
        if task_id != str(tasks[ti]["task_id"]):
            raise RuntimeError(f"negative task mismatch at {rid}")
        labels = list(nrow.get("negative_labels", []))
        for c in nrow["negative_bundle"].get("changed_skills", []):
            if c.get("new_skill_index") is None:
                # Removed/missing skills are not present as changed negative skills.
                continue
            sidx = int(c["new_skill_index"])
            if sidx in gold_by_task[ti]:
                continue
            key = (ti, sidx, "task_negative_changed_skill")
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "task_index": ti,
                    "task_id": task_id,
                    "skill_index": sidx,
                    "skill_name": skill_name(skills, sidx),
                    "edge_type": "task_negative_changed_skill",
                    "target": 0.0,
                    "use_for_message_passing": False,
                    "is_gold_skill": False,
                    "source_negative_record_id": rid,
                    "negative_labels": labels,
                    "change_type": str(c.get("change_type", "")),
                }
            )

    out_jsonl = out_dir / args.output_jsonl
    out_summary = out_dir / args.output_summary
    out_jsonl.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")

    type_counts: dict[str, int] = {}
    for r in rows:
        type_counts[r["edge_type"]] = type_counts.get(r["edge_type"], 0) + 1
    summary = {
        "task_count": len(tasks),
        "skill_count": len(skills),
        "topn": args.topn,
        "task_skill_edge_count": len(rows),
        "edge_type_counts": type_counts,
        "unique_skill_count": len({int(r["skill_index"]) for r in rows}),
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
