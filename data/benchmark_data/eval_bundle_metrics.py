#!/usr/bin/env python3
"""Evaluate bundle recommendation metrics from prediction JSON and gold bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def avg(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def set_metrics(pred: list[int], gold: list[int]) -> dict[str, float]:
    p = set(int(x) for x in pred)
    g = set(int(x) for x in gold)
    inter = len(p & g)
    precision = inter / max(len(p), 1)
    recall = inter / max(len(g), 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    jaccard = inter / max(len(p | g), 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
    }


def dcg(rels: list[float]) -> float:
    total = 0.0
    for i, r in enumerate(rels, start=1):
        total += float(r) / np.log2(i + 1)
    return float(total)


def skill_values(skill_items: list[dict[str, Any]], indices: list[int], key: str) -> list[float]:
    vals = []
    for idx in indices:
        if 0 <= int(idx) < len(skill_items):
            try:
                vals.append(float(skill_items[int(idx)].get(key, 0.0)))
            except Exception:
                vals.append(0.0)
    return vals


def redundancy_stats(skill_emb: np.ndarray, indices: list[int], threshold: float) -> dict[str, float]:
    if len(indices) < 2:
        return {"pair_sim_mean": 0.0, "pair_sim_max": 0.0, "redundant_pair_rate": 0.0}
    vec = skill_emb[np.asarray(indices, dtype=np.int64)]
    sim = vec @ vec.T
    tri = sim[np.triu_indices(sim.shape[0], k=1)]
    return {
        "pair_sim_mean": float(np.mean(tri)),
        "pair_sim_max": float(np.max(tri)),
        "redundant_pair_rate": float(np.mean(tri >= float(threshold))),
    }


def extract_states(task_row: dict[str, Any]) -> list[dict[str, Any]]:
    beam = task_row.get("beam", {})
    states = beam.get("final_top_states") or beam.get("final_top4_states") or []
    if states:
        return states
    idx = beam.get("selected_indices_47153space") or []
    if idx:
        return [{"selected_indices_47153space": idx, "pet_score": beam.get("final_pet_score", beam.get("final_bundle_score", 0.0))}]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bundle recommendation metrics.")
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--gold-json", default="./benchmark_gold_manifest.json")
    parser.add_argument("--split-json", default="./benchmark_train_test_split.json")
    parser.add_argument("--skill-json", default="./benchmark_merged_skills.json")
    parser.add_argument("--skill-emb", default="./benchmark_merged_skill_embeddings.npy")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--redundancy-threshold", type=float, default=0.85)
    parser.add_argument("--output-json", default="./eval_bundle_metrics_output.json")
    args = parser.parse_args()

    pred = load_json(Path(args.prediction_json))
    gold = load_json(Path(args.gold_json))
    split = load_json(Path(args.split_json))
    skills = load_json(Path(args.skill_json))
    skill_emb = np.load(Path(args.skill_emb)).astype(np.float32, copy=False)
    skill_emb = skill_emb / np.clip(np.linalg.norm(skill_emb, axis=1, keepdims=True), 1e-12, None)

    gold_by_task = {int(x["task_index"]): [int(v) for v in x["gold_merged_skill_indices"]] for x in gold}
    if args.split == "all":
        eval_set = set(gold_by_task.keys())
    else:
        key = f"{args.split}_task_indices"
        eval_set = set(int(x) for x in split.get(key, []))

    rows = []
    for task_row in pred.get("tasks", []):
        ti = int(task_row.get("task_index", -1))
        if ti not in eval_set or ti not in gold_by_task:
            continue
        states = extract_states(task_row)
        if not states:
            continue
        gold_idx = gold_by_task[ti]
        top_state = states[0]
        pred_idx = [int(x) for x in top_state.get("selected_indices_47153space", [])]
        sm = set_metrics(pred_idx, gold_idx)
        size_gap = len(pred_idx) - len(gold_idx)
        risk_vals = skill_values(skills, pred_idx, "permission_risk_value")
        cost_vals = skill_values(skills, pred_idx, "token_cost_value")
        red = redundancy_stats(skill_emb, pred_idx, float(args.redundancy_threshold))

        rels = []
        for s in states[: int(args.k)]:
            idx = [int(x) for x in s.get("selected_indices_47153space", [])]
            rels.append(set_metrics(idx, gold_idx)["f1"])
        ideal = sorted(rels, reverse=True)
        ndcg = dcg(rels) / max(dcg(ideal), 1e-12)
        hit = float(any(r > 0.0 for r in rels))
        best_recall_at_k = max((set_metrics([int(x) for x in s.get("selected_indices_47153space", [])], gold_idx)["recall"] for s in states[: int(args.k)]), default=0.0)
        ranks = [i + 1 for i, r in enumerate(rels) if r > 0.0]
        mrr = 1.0 / ranks[0] if ranks else 0.0

        rows.append(
            {
                "task_index": ti,
                "task_id": task_row.get("task_label", task_row.get("task_id")),
                "pred_size": len(pred_idx),
                "gold_size": len(gold_idx),
                "size_gap": float(size_gap),
                "size_mae": float(abs(size_gap)),
                **sm,
                "hit_at_k": hit,
                "best_recall_at_k": float(best_recall_at_k),
                "ndcg_at_k": float(ndcg),
                "mrr_at_k": float(mrr),
                "risk_mean": avg(risk_vals),
                "risk_max": max(risk_vals) if risk_vals else 0.0,
                "cost_mean": avg(cost_vals),
                "cost_max": max(cost_vals) if cost_vals else 0.0,
                **red,
            }
        )

    metric_keys = [
        "precision",
        "recall",
        "f1",
        "jaccard",
        "hit_at_k",
        "best_recall_at_k",
        "ndcg_at_k",
        "mrr_at_k",
        "pred_size",
        "gold_size",
        "size_gap",
        "size_mae",
        "risk_mean",
        "risk_max",
        "cost_mean",
        "cost_max",
        "pair_sim_mean",
        "pair_sim_max",
        "redundant_pair_rate",
    ]
    summary = {
        "prediction_json": str(args.prediction_json),
        "split": args.split,
        "task_count": len(rows),
        "k": int(args.k),
        "metrics": {m: avg([float(r[m]) for r in rows]) for m in metric_keys},
    }
    out = {"summary": summary, "tasks": rows}
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
