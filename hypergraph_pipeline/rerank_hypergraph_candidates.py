#!/usr/bin/env python3
"""Local hypergraph rerank for PET beam candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES = [
    "pet_score",
    "node_support",
    "pair_support",
    "redundancy_max",
    "pair_similarity_mean",
    "bundle_size_norm",
    "risk_max",
    "cost_mean",
    "coverage_max",
    "coverage_mean",
]


def pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank PET beam candidates with local hypergraph features.")
    parser.add_argument("--beam-json", default="../output/pet_beam_search.json")
    parser.add_argument("--skill-json", default="../../data/benchmark_merged_skills.json")
    parser.add_argument("--skill-emb", default="../../data/benchmark_merged_skill_embeddings.npy")
    parser.add_argument("--task-emb", default="../../data/task_embeddings_bigmodel_512_f32_v2.npy")
    parser.add_argument("--config-json", default="")
    parser.add_argument("--lambda-hyper", type=float, default=0.15)
    parser.add_argument("--lambda-redundancy", type=float, default=0.05)
    parser.add_argument("--lambda-size", type=float, default=0.01)
    parser.add_argument("--output-json", default="../output/pet_beam_hypergraph_reranked.json")
    args = parser.parse_args()

    beam = json.loads(Path(args.beam_json).read_text(encoding="utf-8"))
    cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8")) if args.config_json else None
    feature_names = list(cfg.get("feature_names", FEATURE_NAMES)) if cfg else FEATURE_NAMES[:]
    drop_features: list[str] = []
    unknown_drop_features: list[str] = []
    drop_feature_ids: set[int] = set()
    if cfg:
        for name in cfg.get("drop_features", []):
            if name in feature_names:
                drop_features.append(str(name))
                drop_feature_ids.add(feature_names.index(name))
            else:
                unknown_drop_features.append(str(name))
        w1 = np.asarray(cfg["mlp_w1"], dtype=np.float32)
        b1 = np.asarray(cfg["mlp_b1"], dtype=np.float32)
        w2 = np.asarray(cfg["mlp_w2"], dtype=np.float32)
        b2 = float(cfg["mlp_b2"])
        if w1.ndim != 2 or w1.shape[1] != len(feature_names):
            raise ValueError(f"MLP w1 shape {w1.shape} does not match feature count {len(feature_names)}")
        if b1.ndim != 1 or b1.shape[0] != w1.shape[0]:
            raise ValueError(f"MLP b1 shape {b1.shape} does not match hidden count {w1.shape[0]}")
        if w2.ndim != 1 or w2.shape[0] != w1.shape[0]:
            raise ValueError(f"MLP w2 shape {w2.shape} does not match hidden count {w1.shape[0]}")
        mlp = (w1, b1, w2, b2)
    else:
        mlp = None
    skill_meta = json.loads(Path(args.skill_json).read_text(encoding="utf-8"))
    task_emb = np.load(Path(args.task_emb)).astype(np.float32, copy=False)
    task_emb = task_emb / np.clip(np.linalg.norm(task_emb, axis=1, keepdims=True), 1e-12, None)
    skill_emb = np.load(Path(args.skill_emb)).astype(np.float32, copy=False)
    skill_emb = skill_emb / np.clip(np.linalg.norm(skill_emb, axis=1, keepdims=True), 1e-12, None)

    out_tasks = []
    for task in beam.get("tasks", []):
        states = task.get("beam", {}).get("final_top_states", []) or []
        if not states:
            out_tasks.append(task)
            continue

        max_score = max(float(s.get("pet_score", 0.0)) for s in states)
        min_score = min(float(s.get("pet_score", 0.0)) for s in states)
        denom = max(max_score - min_score, 1e-6)

        node_support: dict[int, float] = defaultdict(float)
        pair_support: dict[tuple[int, int], float] = defaultdict(float)
        for s in states:
            idx = [int(x) for x in s.get("selected_indices_47153space", [])]
            w = (float(s.get("pet_score", 0.0)) - min_score) / denom
            for x in idx:
                node_support[x] += w
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    pair_support[pair_key(idx[i], idx[j])] += w

        max_node = max(node_support.values()) if node_support else 1.0
        max_pair = max(pair_support.values()) if pair_support else 1.0

        reranked = []
        task_vec = task_emb[int(task.get("task_index", 0))]
        for s in states:
            idx = [int(x) for x in s.get("selected_indices_47153space", [])]
            if not idx:
                continue
            node_score = sum(node_support.get(x, 0.0) / max_node for x in idx) / max(len(idx), 1)
            pairs = [pair_key(idx[i], idx[j]) for i in range(len(idx)) for j in range(i + 1, len(idx))]
            pair_score = sum(pair_support.get(p, 0.0) / max_pair for p in pairs) / max(len(pairs), 1) if pairs else 0.0
            if len(idx) > 1:
                vec = skill_emb[np.asarray(idx, dtype=np.int64)]
                sim = vec @ vec.T
                tri = sim[np.triu_indices(sim.shape[0], k=1)]
                redundancy = float(np.max(tri))
            else:
                redundancy = 0.0
                tri = np.asarray([0.0], dtype=np.float32)
            risks = []
            costs = []
            for sidx in idx:
                row = skill_meta[int(sidx)] if 0 <= int(sidx) < len(skill_meta) else {}
                try:
                    risks.append(float(row.get("permission_risk_value", 0.0)))
                except Exception:
                    risks.append(0.0)
                try:
                    costs.append(float(row.get("token_cost_value", 0.0)))
                except Exception:
                    costs.append(0.0)
            risk_max = float(max(risks) if risks else 0.0)
            cost_mean = float(sum(costs) / max(len(costs), 1))
            task_sims = skill_emb[np.asarray(idx, dtype=np.int64)] @ task_vec if idx else np.asarray([], dtype=np.float32)
            coverage_max = float(np.max(task_sims)) if len(task_sims) else 0.0
            coverage_mean = float(np.mean(task_sims)) if len(task_sims) else 0.0
            hyper_score = 0.60 * node_score + 0.40 * pair_score
            feature_values = {
                "pet_score": float(s.get("pet_score", 0.0)),
                "node_support": float(node_score),
                "pair_support": float(pair_score),
                "redundancy_max": float(redundancy),
                "pair_similarity_mean": float(np.mean(tri) if len(idx) > 1 else 0.0),
                "bundle_size_norm": float(len(idx)) / 10.0,
                "risk_max": risk_max,
                "cost_mean": cost_mean,
                "coverage_max": coverage_max,
                "coverage_mean": coverage_mean,
            }
            feat = np.asarray([feature_values.get(name, 0.0) for name in feature_names], dtype=np.float32)
            for drop_idx in drop_feature_ids:
                feat[int(drop_idx)] = 0.0
            if mlp:
                w1, b1, w2, b2 = mlp
                final = float(np.tanh(w1 @ feat + b1) @ w2 + b2)
            else:
                final = (
                    float(s.get("pet_score", 0.0))
                    + float(args.lambda_hyper) * hyper_score
                    - float(args.lambda_redundancy) * redundancy
                    - float(args.lambda_size) * max(len(idx) - 2, 0)
                )
            item = dict(s)
            item["hypergraph"] = {
                "node_support_score": float(node_score),
                "pair_support_score": float(pair_score),
                "hyper_score": float(hyper_score),
                "redundancy_penalty": float(redundancy),
                "score_source": "learned_mlp" if mlp else "fixed_rule",
                "feature_values": {name: float(feat[pos]) for pos, name in enumerate(feature_names)},
                "final_score": float(final),
            }
            reranked.append(item)

        reranked.sort(key=lambda x: x["hypergraph"]["final_score"], reverse=True)
        new_task = dict(task)
        new_task["beam"] = dict(task.get("beam", {}))
        new_task["beam"]["final_top_states_before_hypergraph"] = states
        new_task["beam"]["final_top_states"] = reranked
        new_task["beam"]["selected_indices_47153space"] = reranked[0]["selected_indices_47153space"] if reranked else []
        new_task["beam"]["final_hypergraph_score"] = reranked[0]["hypergraph"]["final_score"] if reranked else 0.0
        out_tasks.append(new_task)

    out = {
        "config": {
            "beam_json": args.beam_json,
            "lambda_hyper": float(args.lambda_hyper),
            "lambda_redundancy": float(args.lambda_redundancy),
            "lambda_size": float(args.lambda_size),
            "config_json": args.config_json,
            "mode": "learned_mlp" if mlp else "fixed_rule",
            "score_source": "learned_mlp" if mlp else "fixed_rule",
            "feature_names": feature_names,
            "drop_features": drop_features,
            "drop_feature_ids": sorted(drop_feature_ids),
            "unknown_drop_features": unknown_drop_features,
            "utility_mode": cfg.get("utility_mode") if cfg else None,
            "selected_epoch": cfg.get("selected_epoch") if cfg else None,
            "selection_metric": cfg.get("selection_metric") if cfg else None,
            "val_metrics": cfg.get("val_metrics") if cfg else None,
            "fixed_rule_formula": (
                "pet_score + lambda_hyper * hyper_score - lambda_redundancy * redundancy - "
                "lambda_size * max(bundle_size - 2, 0)"
            )
            if not mlp
            else None,
        },
        "tasks": out_tasks,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path} tasks={len(out_tasks)}")


if __name__ == "__main__":
    main()
