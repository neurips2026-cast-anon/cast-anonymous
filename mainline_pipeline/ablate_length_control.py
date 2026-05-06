#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run length-control ablations for beam scorer.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workdir", default="../")
    parser.add_argument("--skills-json", default="../data/bundle_input/skills_with_risk.json")
    parser.add_argument("--task-json", default="../data/tasks_all_metadata_for_embedding_v2.json")
    parser.add_argument("--task-emb", default="../data/task_embeddings_bigmodel_512_f32_v2.npy")
    parser.add_argument("--skill-emb", default="../data/bundle_input/skills_embedding.npy")
    parser.add_argument("--gold-manifest", default="../data/benchmark_gold_manifest.json")
    parser.add_argument("--split-json", default="../data/benchmark_train_test_split.json")
    parser.add_argument("--base-scorer-config", default="../output/trained_scorer_config_quick3.json")
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=12)
    parser.add_argument("--stop-threshold", type=float, default=0.25)
    parser.add_argument("--min-bundle-size", type=int, default=2)
    parser.add_argument("--final-candidate-count", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.workdir).resolve()
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = json.loads(Path(args.base_scorer_config).resolve().read_text(encoding="utf-8"))

    experiments = [
        {
            "name": "keep_scorer_remove_beam",
            "cfg_updates": {},
            "min_balanced_gain": -999.0,
        },
        {
            "name": "keep_beam_remove_scorer",
            "cfg_updates": {"gain_threshold": 0.0, "growth": 0.0},
            "min_balanced_gain": -0.75,
        },
    ]

    summary = {}
    for exp in experiments:
        cfg = dict(base_cfg)
        cfg.update(exp["cfg_updates"])
        cfg_path = output_dir / f"{exp['name']}_scorer_config.json"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        beam_json = output_dir / f"{exp['name']}_beam.json"
        eval_json = output_dir / f"{exp['name']}_eval.json"

        run(
            [
                args.python,
                str((root / "code" / "search_skill_bundles.py").resolve()),
                "--skills-json",
                str(Path(args.skills_json).resolve()),
                "--task-json",
                str(Path(args.task_json).resolve()),
                "--task-emb",
                str(Path(args.task_emb).resolve()),
                "--skill-emb",
                str(Path(args.skill_emb).resolve()),
                "--max-candidates",
                str(args.max_candidates),
                "--beam-size",
                str(args.beam_size),
                "--stop-threshold",
                str(args.stop_threshold),
                "--min-bundle-size",
                str(args.min_bundle_size),
                "--min-balanced-gain",
                str(exp["min_balanced_gain"]),
                "--final-candidate-count",
                str(args.final_candidate_count),
                "--final-selection-mode",
                "diverse_topk",
                "--scorer-config",
                str(cfg_path),
                "--output",
                str(beam_json),
            ],
            root,
        )

        run(
            [
                args.python,
                str((root / "code" / "evaluate_mainline_predictions.py").resolve()),
                "--beam-json",
                str(beam_json),
                "--gold-manifest",
                str(Path(args.gold_manifest).resolve()),
                "--split-json",
                str(Path(args.split_json).resolve()),
                "--output-json",
                str(eval_json),
            ],
            root,
        )

        summary[exp["name"]] = json.loads(eval_json.read_text(encoding="utf-8"))

    summary_path = output_dir / "length_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
