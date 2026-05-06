#!/usr/bin/env python3
"""
Unified evaluation entrypoint for a beam-search result.

This script can emit:
1. descriptive inference metrics for any task set
2. benchmark-style set-match metrics when gold labels are available

Typical usage:
- 89-task production/inference set: descriptive metrics only
- 61-task labeled set: descriptive metrics + benchmark metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def set_metrics(pred: list[int], gold: list[int]) -> dict[str, float]:
    p = set(int(x) for x in pred)
    g = set(int(x) for x in gold)
    inter = len(p & g)
    precision = inter / max(len(p), 1)
    recall = inter / max(len(g), 1)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    jaccard = inter / max(len(p | g), 1)
    return {
        "bundle_precision": float(precision),
        "bundle_recall": float(recall),
        "bundle_f1": float(f1),
        "bundle_jaccard": float(jaccard),
        "bundle_size_mae": float(abs(len(p) - len(g))),
    }


def avg_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = list(rows[0].keys())
    return {k: float(sum(r[k] for r in rows) / len(rows)) for k in keys}


def resolve_selected_indices(beam: dict, use_rerank: bool) -> list[int]:
    if use_rerank:
        return [int(x) for x in beam.get("selected_indices_47153space_after_rerank", beam.get("selected_indices_47153space", []))]
    return [int(x) for x in beam.get("selected_indices_47153space", [])]


def build_descriptive_row(task: dict, use_rerank: bool) -> dict[str, float | int | str]:
    b = task.get("beam", {})
    suffix = "_after_rerank" if use_rerank else ""
    return {
        "task_index": int(task.get("task_index", 0)),
        "task_label": task.get("task_label"),
        "bundle_size": int(b.get(f"selected_count{suffix}", len(resolve_selected_indices(b, use_rerank)))),
        "semantic_score": float(b.get(f"final_semantic_score{suffix}", b.get("final_semantic_score", 0.0))),
        "structural_score": float(b.get(f"final_structural_score{suffix}", b.get("final_structural_score", 0.0))),
        "final_bundle_score": float(b.get(f"final_bundle_score{suffix}", b.get("final_bundle_score", 0.0))),
        "risk_penalty": float(b.get(f"final_risk_penalty{suffix}", b.get("final_risk_penalty", 0.0))),
        "token_penalty": float(b.get(f"final_token_penalty{suffix}", b.get("final_token_penalty", 0.0))),
    }


def eval_snapshot(args: argparse.Namespace) -> None:
    paper_root = Path(__file__).resolve().parents[1]

    def _resolve_under_paper(raw_path: str, *, allow_missing: bool) -> Path:
        p = Path(raw_path)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        try:
            p.relative_to(paper_root)
        except ValueError as exc:
            raise ValueError(f"path must be inside paper root: {paper_root}, got {p}") from exc
        if not allow_missing and not p.exists():
            raise FileNotFoundError(f"path not found: {p}")
        return p

    beam_path = _resolve_under_paper(args.beam_json, allow_missing=False)
    output_path = _resolve_under_paper(args.output_json, allow_missing=True)
    beam = json.loads(beam_path.read_text(encoding="utf-8"))
    descriptive_rows = []
    benchmark_rows = []
    per_task = []

    gold_by_task = None
    split = None
    train_set: set[int] = set()
    test_set: set[int] = set()
    if args.gold_manifest:
        gold_manifest_path = _resolve_under_paper(args.gold_manifest, allow_missing=False)
        gold_manifest = json.loads(gold_manifest_path.read_text(encoding="utf-8"))
        gold_by_task = {int(x["task_index"]): [int(v) for v in x["gold_merged_skill_indices"]] for x in gold_manifest}
    if args.split_json:
        split_path = _resolve_under_paper(args.split_json, allow_missing=False)
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_set = set(int(x) for x in split.get("train_task_indices", []))
        test_set = set(int(x) for x in split.get("test_task_indices", []))

    bench_train_rows = []
    bench_test_rows = []

    for task in beam.get("tasks", []):
        b = task.get("beam", {})
        selected = resolve_selected_indices(b, args.use_rerank)
        desc = build_descriptive_row(task, args.use_rerank)
        descriptive_rows.append(desc)

        row = {
            "task_index": int(task.get("task_index", 0)),
            "task_label": task.get("task_label"),
            "selected_indices": selected,
            **desc,
        }

        if gold_by_task is not None and row["task_index"] in gold_by_task:
            gold = gold_by_task[row["task_index"]]
            bench = set_metrics(selected, gold)
            benchmark_rows.append(bench)
            row["gold_indices"] = gold
            row.update(bench)
            if row["task_index"] in train_set:
                bench_train_rows.append(bench)
            if row["task_index"] in test_set:
                bench_test_rows.append(bench)

        per_task.append(row)

    descriptive_summary = {}
    if descriptive_rows:
        for key in ("bundle_size", "semantic_score", "structural_score", "final_bundle_score", "risk_penalty", "token_penalty"):
            descriptive_summary[key] = float(sum(float(r[key]) for r in descriptive_rows) / len(descriptive_rows))

    out: dict[str, object] = {
        "task_count": len(per_task),
        "selection_mode": "rerank_top1" if args.use_rerank else "original_top1",
        "metric_definition": {
            "descriptive_primary": ["semantic_score", "structural_score", "final_bundle_score"],
            "descriptive_supporting": ["bundle_size", "risk_penalty", "token_penalty"],
        },
        "descriptive_summary": descriptive_summary,
        "per_task": per_task,
    }

    if benchmark_rows:
        out["metric_definition"] = {
            **out["metric_definition"],
            "benchmark_primary": ["bundle_f1", "bundle_jaccard"],
            "benchmark_secondary": ["bundle_precision", "bundle_recall", "bundle_size_mae"],
        }
        out["benchmark_summary"] = {
            "all_tasks": avg_metrics(benchmark_rows),
            "train_split": avg_metrics(bench_train_rows) if split else {},
            "test_split": avg_metrics(bench_test_rows) if split else {},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved evaluation: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate descriptive and optional benchmark metrics for a beam JSON.")
    parser.add_argument("--beam-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--gold-manifest", default=None, help="Optional gold manifest to also compute benchmark metrics.")
    parser.add_argument("--split-json", default=None, help="Optional train/test split for benchmark summary.")
    parser.add_argument("--use-rerank", action="store_true", help="Evaluate reranked top1 fields when present.")
    args = parser.parse_args()
    eval_snapshot(args)


if __name__ == "__main__":
    main()
