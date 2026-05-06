#!/usr/bin/env python3
"""
Train a compact learned reranker over hypergraph-derived top-k candidate features.

Outputs a config JSON compatible with apply_bundle_reranker.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

FEATURE_PRESETS: dict[str, list[str]] = {
    "baseline6": ["node_stability", "edge_mean", "edge_worst", "edge_consistency", "hyper_support", "edge_progress"],
    "core3": ["edge_mean", "hyper_support", "edge_progress"],
    "recovery4": ["edge_mean", "hyper_support", "edge_progress", "edge_recovery_prior"],
    "hypergap4": ["edge_mean", "hyper_support", "edge_progress", "hyper_gap"],
    "bottleneck4": ["edge_mean", "hyper_support", "edge_progress", "bottleneck_gap"],
}


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _softmax_np(x: np.ndarray, temp: float) -> np.ndarray:
    t = max(float(temp), 1e-6)
    z = np.asarray(x, dtype=np.float64) / t
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.clip(np.sum(e), 1e-12, None)


def _f1(indices: list[int], gold: list[int]) -> float:
    p = set(int(x) for x in indices)
    g = set(int(x) for x in gold)
    inter = len(p & g)
    if not p or not g or inter == 0:
        return 0.0
    prec = inter / len(p)
    rec = inter / len(g)
    return 0.0 if prec + rec == 0 else 2.0 * prec * rec / (prec + rec)


def _preference_tuple(indices: list[int], gold: list[int], state: dict[str, Any]) -> tuple[float, float, float, float]:
    f1 = _f1(indices, gold)
    bundle_size = float(len(indices))
    gold_size = float(len(gold))
    risk_penalty = float(state.get("risk_penalty", 0.0))
    token_penalty = float(state.get("token_penalty", 0.0))
    size_gap = abs(bundle_size - gold_size)
    # Lexicographic preference: F1 first, then smaller size-gap, then lower risk/cost.
    return (float(f1), -size_gap, -risk_penalty, -token_penalty)


def _build_feature_rows(
    beam_data: dict[str, Any],
    hgraph: dict[str, Any],
    gold_by_task: dict[int, list[int]],
    split_train: set[int],
    rerank_candidate_count: int,
    feature_keys: list[str],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], float]]:
    node_lookup = {}
    for s in hgraph.get("skill_nodes", []):
        idx = int(s.get("skill_index", -1))
        if idx >= 0:
            node_lookup[idx] = s

    edge_lookup = {}
    max_cooccur = 1.0
    for e in hgraph.get("edges", []):
        key = _pair_key(int(e["src_skill_index"]), int(e["dst_skill_index"]))
        edge_lookup[key] = e
        max_cooccur = max(max_cooccur, float(e.get("cooccur_count", 0.0)))

    hyper_lookup = {}
    max_hyper = 1.0
    for h in hgraph.get("hyperedges", []):
        key = tuple(int(x) for x in h.get("skill_indices", []))
        hyper_lookup[key] = h
        max_hyper = max(max_hyper, float(h.get("cooccur_count", 0.0)))

    task_cache: list[dict[str, Any]] = []
    pair_total: dict[tuple[int, int], int] = {}
    pair_win: dict[tuple[int, int], int] = {}
    total_candidates = 0
    total_winners = 0
    for task in beam_data.get("tasks", []):
        ti = int(task.get("task_index", -1))
        if ti not in split_train or ti not in gold_by_task:
            continue

        beam = task.get("beam", {})
        cands = (
            beam.get("final_top_states")
            or beam.get("final_top4_states")
            or beam.get("global_top_beam_states", [])[:rerank_candidate_count]
        )
        cands = cands[:rerank_candidate_count]
        if len(cands) < 2:
            continue

        raw_feature_rows: list[dict[str, float]] = []
        pref = []
        cand_pairs: list[list[tuple[int, int]]] = []
        gold = gold_by_task[ti]
        for state in cands:
            indices = [int(x) for x in state.get("selected_indices_47153space", [])]
            if not indices:
                continue

            skill_appear = []
            skill_drop = []
            skill_displaced = []
            for idx in indices:
                s = node_lookup.get(idx, {})
                total_rounds = float(s.get("total_rounds", 0.0))
                skill_appear.append(float(s.get("appear_rate", 0.0)))
                skill_drop.append(float(s.get("drop_count", 0.0)) / total_rounds if total_rounds else 0.0)
                skill_displaced.append(float(s.get("displaced_count", 0.0)) / total_rounds if total_rounds else 0.0)

            co_rates = []
            pair_drop_rates = []
            pair_continue_gains = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    edge = edge_lookup.get(_pair_key(indices[i], indices[j]))
                    if edge:
                        co_rates.append(float(edge.get("cooccur_count", 0.0)) / max_cooccur)
                        pair_drop_rates.append(float(edge.get("drop_rate", 0.0)))
                        pair_continue_gains.append(float(edge.get("avg_continuation_gain", 0.0)))
            pairs = [_pair_key(indices[i], indices[j]) for i in range(len(indices)) for j in range(i + 1, len(indices))]

            hkey = tuple(sorted(indices))
            hinfo = hyper_lookup.get(hkey, {})
            hyper_support = float(hinfo.get("cooccur_count", 0.0)) / max_hyper if hinfo else 0.0

            avg_appear = float(sum(skill_appear) / len(skill_appear)) if skill_appear else 0.0
            avg_drop = float(sum(skill_drop) / len(skill_drop)) if skill_drop else 0.0
            avg_displaced = float(sum(skill_displaced) / len(skill_displaced)) if skill_displaced else 0.0
            edge_mean = float(sum(co_rates) / len(co_rates)) if co_rates else 0.0
            edge_worst = float(min(co_rates)) if co_rates else 0.0
            avg_pair_drop = float(sum(pair_drop_rates) / len(pair_drop_rates)) if pair_drop_rates else 0.0
            edge_consistency = float(edge_mean - 0.5 * avg_pair_drop)
            edge_progress = float(sum(pair_continue_gains) / len(pair_continue_gains)) if pair_continue_gains else 0.0
            node_stability = float(avg_appear - 0.5 * avg_drop - 0.5 * avg_displaced)

            raw_feature_rows.append(
                {
                    "node_stability": float(node_stability),
                    "edge_mean": float(edge_mean),
                    "edge_worst": float(edge_worst),
                    "edge_consistency": float(edge_consistency),
                    "hyper_support": float(hyper_support),
                    "edge_progress": float(edge_progress),
                }
            )
            pref.append(_preference_tuple(indices, gold, state))
            cand_pairs.append(pairs)

        if len(raw_feature_rows) >= 2:
            best_pref = max(pref)
            winner_idx = {i for i, p in enumerate(pref) if p == best_pref}
            total_candidates += len(pref)
            total_winners += len(winner_idx)
            for i, pairs in enumerate(cand_pairs):
                for pair in pairs:
                    pair_total[pair] = pair_total.get(pair, 0) + 1
                    if i in winner_idx:
                        pair_win[pair] = pair_win.get(pair, 0) + 1
            task_cache.append(
                {
                    "task_index": ti,
                    "raw_features": raw_feature_rows,
                    "preference": pref,
                    "cand_pairs": cand_pairs,
                }
            )

    global_win_rate = float(total_winners / max(total_candidates, 1))
    pair_recovery_prior: dict[tuple[int, int], float] = {}
    for pair, tot in pair_total.items():
        win = pair_win.get(pair, 0)
        # Smoothed advantage over global winner prior.
        post = float((win + 1.0) / (tot + 2.0))
        pair_recovery_prior[pair] = float(post - global_win_rate)

    rows: list[dict[str, Any]] = []
    for cache in task_cache:
        feat_rows = []
        for base_feat, pairs in zip(cache["raw_features"], cache["cand_pairs"]):
            if pairs:
                recovery = float(sum(pair_recovery_prior.get(p, 0.0) for p in pairs) / len(pairs))
            else:
                recovery = 0.0
            feature_bank = dict(base_feat)
            feature_bank["edge_recovery_prior"] = float(recovery)
            feature_bank["hyper_gap"] = float(feature_bank["hyper_support"] - feature_bank["edge_mean"])
            feature_bank["bottleneck_gap"] = float(feature_bank["edge_mean"] - feature_bank["edge_worst"])
            feat_rows.append([float(feature_bank.get(k, 0.0)) for k in feature_keys])
        rows.append(
            {
                "task_index": cache["task_index"],
                "features": feat_rows,
                "preference": cache["preference"],
            }
        )
    return rows, pair_recovery_prior


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned hypergraph reranker.")
    parser.add_argument("--beam-json", required=True)
    parser.add_argument("--hypergraph-json", required=True)
    parser.add_argument("--gold-manifest", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--rerank-candidate-count", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pair-margin", type=float, default=0.05)
    parser.add_argument(
        "--feature-set",
        choices=tuple(FEATURE_PRESETS.keys()),
        default="recovery4",
        help="Structural feature preset for reranker training.",
    )
    args = parser.parse_args()

    beam_data = json.loads(Path(args.beam_json).read_text(encoding="utf-8"))
    hgraph = json.loads(Path(args.hypergraph_json).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.gold_manifest).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))

    gold_by_task = {int(x["task_index"]): [int(v) for v in x["gold_merged_skill_indices"]] for x in manifest}
    split_train = set(int(x) for x in split.get("train_task_indices", []))
    feature_keys = FEATURE_PRESETS[str(args.feature_set)]
    train_rows, pair_recovery_prior = _build_feature_rows(
        beam_data=beam_data,
        hgraph=hgraph,
        gold_by_task=gold_by_task,
        split_train=split_train,
        rerank_candidate_count=int(args.rerank_candidate_count),
        feature_keys=feature_keys,
    )
    if not train_rows:
        raise ValueError("No train rows available for reranker training.")

    mlp = torch.nn.Sequential(
        torch.nn.Linear(len(feature_keys), 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    for _ in range(int(args.epochs)):
        total = 0.0
        for row in train_rows:
            x = torch.tensor(row["features"], dtype=torch.float32)
            logits = mlp(x).squeeze(-1)
            pref = row["preference"]
            loss_terms = []
            for i in range(len(pref)):
                for j in range(i + 1, len(pref)):
                    if pref[i] == pref[j]:
                        continue
                    if pref[i] > pref[j]:
                        loss_terms.append(F.relu(float(args.pair_margin) - logits[i] + logits[j]))
                    else:
                        loss_terms.append(F.relu(float(args.pair_margin) - logits[j] + logits[i]))
            if not loss_terms:
                continue
            loss = torch.mean(torch.stack(loss_terms))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())

    with torch.no_grad():
        prior_serialized = {
            f"{int(a)}|{int(b)}": float(v)
            for (a, b), v in pair_recovery_prior.items()
        }
        selected_config = {
            "reranker_type": "structural_mlp",
            "structural_mlp_w1": mlp[0].weight.detach().cpu().tolist(),
            "structural_mlp_b1": mlp[0].bias.detach().cpu().tolist(),
            "structural_mlp_w2": mlp[2].weight.detach().cpu().view(-1).tolist(),
            "structural_mlp_b2": float(mlp[2].bias.detach().cpu().view(-1)[0]),
            "pair_recovery_prior": prior_serialized,
        }

    out = {
        "model_type": "learned_hypergraph_reranker",
        "selected_config": selected_config,
        "train_task_count": len(train_rows),
        "config": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "pair_margin": float(args.pair_margin),
            "rerank_candidate_count": int(args.rerank_candidate_count),
            "feature_set": str(args.feature_set),
            "feature_names": feature_keys,
            "target_definition": "lexicographic_preference: F1 > compactness > low_penalty",
        },
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
