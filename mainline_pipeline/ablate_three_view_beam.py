#!/usr/bin/env python3
"""
Run beam-search ablations over the three scoring views:
1) skill-task
2) skill-bundle (attention + PCA)
3) bundle-task (MLP)

For each variant:
- write a temporary scorer_config
- run beam search on benchmark assets
- run top-k hypergraph rerank
- evaluate benchmark metrics
- compute missing-gold recovery metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_VIEWS = {
    "task_only": (1.0, 0.0, 0.0),
    "sb_only": (0.0, 1.0, 0.0),
    "bt_only": (0.0, 0.0, 1.0),
    "task_sb": (0.5, 0.5, 0.0),
    "task_bt": (0.5, 0.0, 0.5),
    "sb_bt": (0.0, 0.5, 0.5),
    "all_three": (0.25, 0.50, 0.25),
}


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def compute_recovery(top20_json: Path, beam_json: Path, gold_manifest: Path) -> dict[str, float]:
    top20 = json.loads(top20_json.read_text(encoding="utf-8"))
    beam = json.loads(beam_json.read_text(encoding="utf-8"))
    gold = json.loads(gold_manifest.read_text(encoding="utf-8"))
    gold_by_task = {int(x["task_index"]): [int(v) for v in x["gold_merged_skill_indices"]] for x in gold}
    beam_by_task = {int(t["task_index"]): t["beam"] for t in beam["tasks"]}

    task_count = 0
    total_missing = 0
    total_recovered = 0
    full_recovery = 0
    partial_recovery = 0
    for task in top20["results"]:
        ti = int(task["task_index"])
        gold_list = [int(x) for x in gold_by_task[ti]]
        g = len(gold_list)
        ranked = [int(x["skill_index"]) for x in task["topk"]]
        topg = ranked[:g]
        missing = [x for x in gold_list if x not in topg]
        if not missing:
            continue
        task_count += 1
        total_missing += len(missing)
        final_pred = set(int(x) for x in beam_by_task[ti].get("selected_indices_47153space_after_rerank", beam_by_task[ti].get("selected_indices_47153space", [])))
        recovered = [x for x in missing if x in final_pred]
        total_recovered += len(recovered)
        if recovered:
            partial_recovery += 1
        if len(recovered) == len(missing):
            full_recovery += 1

    return {
        "missing_skill_recovery_rate": float(total_recovered / max(total_missing, 1)),
        "task_partial_or_full_recovery_rate": float(partial_recovery / max(task_count, 1)),
        "task_full_recovery_rate": float(full_recovery / max(task_count, 1)),
        "tasks_with_gold_missing_from_topg": int(task_count),
        "total_missing_gold_skills": int(total_missing),
        "total_recovered_missing_gold_skills": int(total_recovered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate the three beam scoring views on benchmark assets.")
    parser.add_argument("--base-config", default="data/default_bundle_scorer_config.json")
    parser.add_argument("--skills-json", default="data/bundle_input/skills_with_risk.json")
    parser.add_argument("--task-json", default="data/tasks_all_metadata_for_embedding_v2.json")
    parser.add_argument("--task-emb", default="data/task_embeddings_bigmodel_512_f32_v2.npy")
    parser.add_argument("--skill-emb", default="data/bundle_input/skills_embedding.npy")
    parser.add_argument("--gold-manifest", default="data/benchmark_gold_manifest.json")
    parser.add_argument("--split-json", default="data/benchmark_train_test_split.json")
    parser.add_argument("--rerank-config", default="data/benchmark_gold_aware_top4_rerank_config_bs9_pca_blend06_exact.json")
    parser.add_argument("--beam-size", type=int, default=12)
    parser.add_argument("--final-candidate-count", type=int, default=6)
    parser.add_argument("--final-selection-mode", choices=("global_topk", "diverse_topk"), default="diverse_topk")
    parser.add_argument("--rerank-candidate-count", type=int, default=6)
    parser.add_argument("--work-dir", default="temp_ablation_three_views")
    parser.add_argument("--summary-json", default="temp_ablation_three_views/summary.json")
    parser.add_argument("--summary-csv", default="temp_ablation_three_views/summary.csv")
    args = parser.parse_args()

    cwd = Path.cwd()
    work_dir = cwd / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = json.loads((cwd / args.base_config).read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for name, (w_task, w_sb, w_bt) in DEFAULT_VIEWS.items():
        cfg = dict(base_cfg)
        cfg["semantic_task_weight"] = float(w_task)
        cfg["semantic_attention_weight"] = float(w_sb)
        cfg["semantic_bundle_weight"] = float(w_bt)
        cfg["semantic_weight"] = 1.0
        cfg["structural_weight"] = 0.0
        cfg_path = work_dir / f"{name}_config.json"
        beam_path = work_dir / f"{name}_beam.json"
        hyper_path = work_dir / f"{name}_hyper.json"
        rerank_path = work_dir / f"{name}_reranked.json"
        eval_path = work_dir / f"{name}_eval.json"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        run(
            [
                "python",
                "search_skill_bundles.py",
                "--skills-json",
                args.skills_json,
                "--task-json",
                args.task_json,
                "--task-emb",
                args.task_emb,
                "--skill-emb",
                args.skill_emb,
                "--beam-size",
                str(args.beam_size),
                "--final-candidate-count",
                str(args.final_candidate_count),
                "--final-selection-mode",
                args.final_selection_mode,
                "--scorer-config",
                str(cfg_path),
                "--output",
                str(beam_path),
            ],
            cwd,
        )
        run(
            [
                "python",
                "build_hypergraph_candidates.py",
                "--beam-json",
                str(beam_path),
                "--skills-json",
                args.skills_json,
                "--output-json",
                str(hyper_path),
            ],
            cwd,
        )
        run(
            [
                "python",
                "apply_bundle_reranker.py",
                "--beam-json",
                str(beam_path),
                "--hypergraph-json",
                str(hyper_path),
                "--config-json",
                args.rerank_config,
                "--output-json",
                str(rerank_path),
                "--rerank-candidate-count",
                str(args.rerank_candidate_count),
            ],
            cwd,
        )
        run(
            [
                "python",
                "evaluate_mainline_predictions.py",
                "--beam-json",
                str(rerank_path),
                "--gold-manifest",
                args.gold_manifest,
                "--split-json",
                args.split_json,
                "--use-rerank",
                "--output-json",
                str(eval_path),
            ],
            cwd,
        )

        eval_obj = json.loads(eval_path.read_text(encoding="utf-8"))
        summary = eval_obj["benchmark_summary"]
        recovery = compute_recovery(cwd / args.top20_json, rerank_path, cwd / args.gold_manifest)
        row = {
            "variant": name,
            "semantic_task_weight": w_task,
            "semantic_attention_weight": w_sb,
            "semantic_bundle_weight": w_bt,
            **{f"all_{k}": v for k, v in summary["all_tasks"].items()},
            **{f"test_{k}": v for k, v in summary["test_split"].items()},
            **recovery,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    out_json = cwd / args.summary_json
    out_csv = cwd / args.summary_csv
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved ablation summary: {out_json}")
    print(f"Saved ablation csv    : {out_csv}")


if __name__ == "__main__":
    main()
