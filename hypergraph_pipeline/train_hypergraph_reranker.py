#!/usr/bin/env python3
"""Train a lightweight hypergraph reranker on PET beam candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def f1_score(pred: list[int], gold: list[int]) -> float:
    p = set(int(x) for x in pred)
    g = set(int(x) for x in gold)
    inter = len(p & g)
    if not p or not g or inter == 0:
        return 0.0
    prec = inter / len(p)
    rec = inter / len(g)
    return float(2.0 * prec * rec / (prec + rec))


def set_metrics(pred: list[int], gold: list[int]) -> dict[str, float]:
    p = set(int(x) for x in pred)
    g = set(int(x) for x in gold)
    inter = len(p & g)
    precision = inter / max(len(p), 1)
    recall = inter / max(len(g), 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def risk_cost(skill_meta: list[dict[str, Any]], idx: list[int]) -> tuple[float, float]:
    risks = []
    costs = []
    for x in idx:
        row = skill_meta[int(x)] if 0 <= int(x) < len(skill_meta) else {}
        try:
            risks.append(float(row.get("permission_risk_value", 0.0)))
        except Exception:
            risks.append(0.0)
        try:
            costs.append(float(row.get("token_cost_value", 0.0)))
        except Exception:
            costs.append(0.0)
    return (float(max(risks) if risks else 0.0), float(sum(costs) / max(len(costs), 1)))


def candidate_preference(
    *,
    f1: float,
    size_gap: float,
    risk_max: float,
    cost_mean: float,
    utility_mode: str,
) -> float:
    if utility_mode != "manual":
        raise ValueError(
            f"Unsupported utility_mode={utility_mode}. "
            "Only 'manual' is enabled in the official pipeline."
        )
    return 100.0 * float(f1) - 1.0 * float(size_gap) - 0.1 * float(risk_max) - 0.05 * float(cost_mean)


def zero_dropped_features(feats: list[list[float]], drop_feature_ids: set[int]) -> list[list[float]]:
    if not drop_feature_ids:
        return feats
    out = []
    for row in feats:
        copied = list(row)
        for idx in drop_feature_ids:
            copied[idx] = 0.0
        out.append(copied)
    return out


def build_features(states: list[dict[str, Any]], skill_emb: np.ndarray, skill_meta: list[dict[str, Any]], task_vec: np.ndarray) -> list[list[float]]:
    if not states:
        return []
    scores = [float(s.get("pet_score", 0.0)) for s in states]
    max_score = max(scores)
    min_score = min(scores)
    denom = max(max_score - min_score, 1e-6)
    node_support: dict[int, float] = {}
    pair_support: dict[tuple[int, int], float] = {}
    for s in states:
        idx = [int(x) for x in s.get("selected_indices_47153space", [])]
        w = (float(s.get("pet_score", 0.0)) - min_score) / denom
        for x in idx:
            node_support[x] = node_support.get(x, 0.0) + w
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                k = pair_key(idx[i], idx[j])
                pair_support[k] = pair_support.get(k, 0.0) + w
    max_node = max(node_support.values()) if node_support else 1.0
    max_pair = max(pair_support.values()) if pair_support else 1.0

    feats = []
    for s in states:
        idx = [int(x) for x in s.get("selected_indices_47153space", [])]
        if not idx:
            feats.append([0.0] * len(FEATURE_NAMES))
            continue
        node_score = sum(node_support.get(x, 0.0) / max_node for x in idx) / max(len(idx), 1)
        pairs = [pair_key(idx[i], idx[j]) for i in range(len(idx)) for j in range(i + 1, len(idx))]
        pair_score = sum(pair_support.get(p, 0.0) / max_pair for p in pairs) / max(len(pairs), 1) if pairs else 0.0
        if len(idx) > 1:
            vec = skill_emb[np.asarray(idx, dtype=np.int64)]
            sim = vec @ vec.T
            tri = sim[np.triu_indices(sim.shape[0], k=1)]
            red = float(np.max(tri))
            mean_sim = float(np.mean(tri))
        else:
            red = 0.0
            mean_sim = 0.0
        risk_max, cost_mean = risk_cost(skill_meta, idx)
        task_sims = skill_emb[np.asarray(idx, dtype=np.int64)] @ task_vec
        coverage_max = float(np.max(task_sims)) if len(task_sims) else 0.0
        coverage_mean = float(np.mean(task_sims)) if len(task_sims) else 0.0
        feats.append(
            [
                float(s.get("pet_score", 0.0)),
                float(node_score),
                float(pair_score),
                float(red),
                float(mean_sim),
                float(len(idx)) / 10.0,
                float(risk_max),
                float(cost_mean),
                coverage_max,
                coverage_mean,
            ]
        )
    return feats


def build_candidate_rows(
    *,
    beam: dict[str, Any],
    gold_by_task: dict[int, list[int]],
    skill_emb: np.ndarray,
    skill_meta: list[dict[str, Any]],
    task_emb: np.ndarray,
    drop_feature_ids: set[int],
    utility_mode: str,
) -> tuple[list[list[list[float]]], list[list[float]], list[list[list[int]]], int]:
    feature_rows = []
    pref_rows = []
    bundle_rows = []
    task_count = 0
    for task in beam.get("tasks", []):
        ti = int(task.get("task_index", -1))
        if ti not in gold_by_task:
            continue
        states = task.get("beam", {}).get("final_top_states", []) or []
        if len(states) < 2:
            continue
        feats = build_features(states, skill_emb, skill_meta, task_emb[ti])
        feats = zero_dropped_features(feats, drop_feature_ids)
        prefs = []
        bundles = []
        for s in states:
            idx = [int(x) for x in s.get("selected_indices_47153space", [])]
            f1 = f1_score(idx, gold_by_task[ti])
            size_gap = abs(len(idx) - len(gold_by_task[ti]))
            risk_max, cost_mean = risk_cost(skill_meta, idx)
            prefs.append(
                candidate_preference(
                    f1=f1,
                    size_gap=size_gap,
                    risk_max=risk_max,
                    cost_mean=cost_mean,
                    utility_mode=utility_mode,
                )
            )
            bundles.append(idx)
        feature_rows.append(feats)
        pref_rows.append(prefs)
        bundle_rows.append(bundles)
        task_count += 1
    return feature_rows, pref_rows, bundle_rows, task_count


@torch.no_grad()
def evaluate_candidate_rows(
    *,
    model: torch.nn.Module,
    feature_rows: list[list[list[float]]],
    bundle_rows: list[list[list[int]]],
    gold_rows: list[list[int]],
) -> dict[str, float]:
    rows = []
    for feats, bundles, gold in zip(feature_rows, bundle_rows, gold_rows):
        if not feats:
            continue
        x = torch.tensor(feats, dtype=torch.float32)
        scores = model(x).squeeze(-1)
        best = int(torch.argmax(scores).item())
        pred = bundles[best]
        sm = set_metrics(pred, gold)
        size_gap = len(pred) - len(gold)
        rows.append(
            {
                **sm,
                "pred_size": float(len(pred)),
                "gold_size": float(len(gold)),
                "size_gap": float(size_gap),
                "size_mae": float(abs(size_gap)),
            }
        )
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "pred_size": 0.0, "size_mae": 0.0}
    keys = ["precision", "recall", "f1", "pred_size", "gold_size", "size_gap", "size_mae"]
    return {k: float(sum(r[k] for r in rows) / len(rows)) for k in keys}


def export_config(
    model: torch.nn.Sequential,
    drop_features: list[str],
    selected_epoch: int,
    val_metrics: dict[str, float] | None,
    utility_mode: str,
) -> dict[str, Any]:
    return {
        "feature_names": FEATURE_NAMES,
        "drop_features": drop_features,
        "selected_epoch": int(selected_epoch),
        "selection_metric": "val_f1" if val_metrics is not None else "final_epoch",
        "val_metrics": val_metrics,
        "utility_mode": utility_mode,
        "mlp_w1": model[0].weight.detach().cpu().tolist(),
        "mlp_b1": model[0].bias.detach().cpu().tolist(),
        "mlp_w2": model[2].weight.detach().cpu().view(-1).tolist(),
        "mlp_b2": float(model[2].bias.detach().cpu().view(-1)[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hypergraph reranker.")
    parser.add_argument("--beam-json", required=True)
    parser.add_argument("--val-beam-json", default="", help="Optional validation beam JSON for epoch selection and early stopping.")
    parser.add_argument("--gold-json", default="../../data/benchmark_gold_manifest.json")
    parser.add_argument("--skill-json", default="../../data/benchmark_merged_skills.json")
    parser.add_argument("--skill-emb", default="../../data/benchmark_merged_skill_embeddings.npy")
    parser.add_argument("--task-emb", default="../../data/task_embeddings_bigmodel_512_f32_v2.npy")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument(
        "--utility-mode",
        choices=("manual",),
        default="manual",
        help="Locked to manual for the official line.",
    )
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--drop-features", default="", help="Comma-separated feature names to zero out for feature-set selection/ablation.")
    parser.add_argument("--output-config", default="../output/hypergraph_reranker_config.json")
    parser.add_argument("--output-report", default="../output/hypergraph_reranker_train_report.json")
    args = parser.parse_args()
    torch.manual_seed(int(args.seed))

    beam = load_json(Path(args.beam_json))
    val_beam = load_json(Path(args.val_beam_json)) if args.val_beam_json else None
    gold = load_json(Path(args.gold_json))
    skill_meta = load_json(Path(args.skill_json))
    skill_emb = np.load(Path(args.skill_emb)).astype(np.float32, copy=False)
    skill_emb = skill_emb / np.clip(np.linalg.norm(skill_emb, axis=1, keepdims=True), 1e-12, None)
    task_emb = np.load(Path(args.task_emb)).astype(np.float32, copy=False)
    task_emb = task_emb / np.clip(np.linalg.norm(task_emb, axis=1, keepdims=True), 1e-12, None)
    gold_by_task = {int(x["task_index"]): [int(v) for v in x["gold_merged_skill_indices"]] for x in gold}

    drop_features = [x.strip() for x in str(args.drop_features).split(",") if x.strip()]
    unknown = sorted(set(drop_features) - set(FEATURE_NAMES))
    if unknown:
        raise RuntimeError(f"Unknown drop feature(s): {unknown}. Valid: {FEATURE_NAMES}")
    drop_feature_ids = {FEATURE_NAMES.index(x) for x in drop_features}

    feature_rows, pref_rows, bundle_rows, task_count = build_candidate_rows(
        beam=beam,
        gold_by_task=gold_by_task,
        skill_emb=skill_emb,
        skill_meta=skill_meta,
        task_emb=task_emb,
        drop_feature_ids=drop_feature_ids,
        utility_mode=str(args.utility_mode),
    )

    if not feature_rows:
        raise RuntimeError("No candidate rows for hypergraph training.")
    train_gold_rows = []
    for task in beam.get("tasks", []):
        ti = int(task.get("task_index", -1))
        states = task.get("beam", {}).get("final_top_states", []) or []
        if ti in gold_by_task and len(states) >= 2:
            train_gold_rows.append(gold_by_task[ti])

    val_feature_rows: list[list[list[float]]] = []
    val_bundle_rows: list[list[list[int]]] = []
    val_gold_rows: list[list[int]] = []
    val_task_count = 0
    if val_beam is not None:
        val_feature_rows, _val_pref_rows, val_bundle_rows, val_task_count = build_candidate_rows(
            beam=val_beam,
            gold_by_task=gold_by_task,
            skill_emb=skill_emb,
            skill_meta=skill_meta,
            task_emb=task_emb,
            drop_feature_ids=drop_feature_ids,
            utility_mode=str(args.utility_mode),
        )
        for task in val_beam.get("tasks", []):
            ti = int(task.get("task_index", -1))
            states = task.get("beam", {}).get("final_top_states", []) or []
            if ti in gold_by_task and len(states) >= 2:
                val_gold_rows.append(gold_by_task[ti])

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    history = []
    best_state = None
    best_epoch = 0
    best_val_metrics: dict[str, float] | None = None
    best_val_f1 = float("-inf")
    bad_checks = 0
    for epoch in range(int(args.epochs)):
        total = 0.0
        pair_count = 0
        for feats, prefs in zip(feature_rows, pref_rows):
            x = torch.tensor(feats, dtype=torch.float32)
            y = model(x).squeeze(-1)
            terms = []
            for i in range(len(prefs)):
                for j in range(len(prefs)):
                    if prefs[i] <= prefs[j]:
                        continue
                    terms.append(F.softplus(-(y[i] - y[j])))
            if not terms:
                continue
            loss = torch.stack(terms).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            pair_count += len(terms)
        avg = total / max(len(feature_rows), 1)
        item: dict[str, Any] = {"epoch": epoch + 1, "loss": avg, "pair_count": pair_count}
        should_eval = val_beam is not None and ((epoch + 1) % max(int(args.eval_every), 1) == 0 or epoch + 1 == int(args.epochs))
        if should_eval:
            val_metrics = evaluate_candidate_rows(
                model=model,
                feature_rows=val_feature_rows,
                bundle_rows=val_bundle_rows,
                gold_rows=val_gold_rows,
            )
            item["val_metrics"] = val_metrics
            val_f1 = float(val_metrics["f1"])
            if val_f1 > best_val_f1 + float(args.min_delta):
                best_val_f1 = val_f1
                best_val_metrics = val_metrics
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_checks = 0
            else:
                bad_checks += 1
            item["best_epoch"] = best_epoch
            item["bad_checks"] = bad_checks
        history.append(item)
        if (epoch + 1) % 50 == 0:
            suffix = f" val_f1={best_val_f1:.6f} best_epoch={best_epoch}" if val_beam is not None else ""
            print(f"epoch={epoch+1} loss={avg:.6f} pairs={pair_count}{suffix}")
        if val_beam is not None and bad_checks >= int(args.patience):
            print(f"Early stop at epoch={epoch+1}; best_epoch={best_epoch} best_val_f1={best_val_f1:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = history[-1]["epoch"]

    train_metrics = evaluate_candidate_rows(
        model=model,
        feature_rows=feature_rows,
        bundle_rows=bundle_rows,
        gold_rows=train_gold_rows,
    )
    config = export_config(model, drop_features, best_epoch, best_val_metrics, str(args.utility_mode))
    report = {
        "beam_json": args.beam_json,
        "val_beam_json": args.val_beam_json,
        "train_task_count": task_count,
        "val_task_count": val_task_count,
        "candidate_task_count": len(feature_rows),
        "drop_features": drop_features,
        "utility_mode": str(args.utility_mode),
        "seed": int(args.seed),
        "early_stopping": {
            "enabled": bool(val_beam is not None),
            "eval_every": int(args.eval_every),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "best_epoch": int(best_epoch),
            "best_val_metrics": best_val_metrics,
        },
        "train_metrics_at_selected_epoch": train_metrics,
        "history": history,
    }
    out_cfg = Path(args.output_config)
    out_rep = Path(args.output_report)
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    out_rep.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved config: {out_cfg}")
    print(f"Saved report: {out_rep}")


if __name__ == "__main__":
    main()
