#!/usr/bin/env python3
"""
Apply compact pair-graph-first reranker to beam candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _mlp_score(features: list[float], cfg: dict) -> float:
    import math
    w1 = cfg.get("structural_mlp_w1")
    b1 = cfg.get("structural_mlp_b1")
    w2 = cfg.get("structural_mlp_w2")
    b2 = cfg.get("structural_mlp_b2")
    if w1 is None or b1 is None or w2 is None or b2 is None:
        return 0.0
    h = []
    for row, bias in zip(w1, b1):
        s = sum(float(a) * float(x) for a, x in zip(row, features)) + float(bias)
        h.append(math.tanh(s))
    y = sum(float(a) * float(x) for a, x in zip(w2, h)) + float(b2)
    return float(y)


def _resolve_feature_vector(cfg: dict, feature_bank: dict[str, float]) -> list[float]:
    """Resolve feature vector by config feature_names, fallback by legacy in-dim."""
    feature_names = cfg.get("feature_names")
    if isinstance(feature_names, list) and feature_names:
        return [float(feature_bank.get(str(k), 0.0)) for k in feature_names]
    w1 = cfg.get("structural_mlp_w1")
    in_dim = 0
    if isinstance(w1, list) and w1 and isinstance(w1[0], list):
        in_dim = len(w1[0])
    if in_dim == 3:
        keys = ("edge_mean", "hyper_support", "edge_progress")
    elif in_dim == 4:
        keys = ("edge_mean", "hyper_support", "edge_progress", "edge_recovery_prior")
    else:
        # Legacy default (also used as fallback).
        keys = ("node_stability", "edge_mean", "edge_worst", "edge_consistency", "hyper_support", "edge_progress")
    vec = [float(feature_bank.get(k, 0.0)) for k in keys]
    if in_dim > 0 and len(vec) != in_dim:
        if len(vec) > in_dim:
            vec = vec[:in_dim]
        else:
            vec = vec + [0.0] * (in_dim - len(vec))
    return vec


def main() -> None:
    paper_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Apply compact pair-graph reranker to beam JSON.")
    parser.add_argument("--beam-json", required=True)
    parser.add_argument("--hypergraph-json", required=True)
    parser.add_argument("--config-json", required=True, help="Report JSON with selected_config or raw config fields")
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--rerank-candidate-count",
        type=int,
        default=4,
        help="How many final candidates to rerank per task.",
    )
    args = parser.parse_args()

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
    hypergraph_path = _resolve_under_paper(args.hypergraph_json, allow_missing=False)
    config_path = _resolve_under_paper(args.config_json, allow_missing=False)
    output_path = _resolve_under_paper(args.output_json, allow_missing=True)

    beam_data = json.loads(beam_path.read_text(encoding="utf-8"))
    hgraph = json.loads(hypergraph_path.read_text(encoding="utf-8"))
    cfg_raw = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = cfg_raw.get("selected_config", cfg_raw.get("gold_aware_rerank_config", cfg_raw.get("config", cfg_raw)))

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
    max_hyper_cooccur = 1.0
    for h in hgraph.get("hyperedges", []):
        key = tuple(int(x) for x in h.get("skill_indices", []))
        hyper_lookup[key] = h
        max_hyper_cooccur = max(max_hyper_cooccur, float(h.get("cooccur_count", 0.0)))
    pair_prior_raw = cfg.get("pair_recovery_prior", {}) if isinstance(cfg, dict) else {}
    pair_prior: dict[tuple[int, int], float] = {}
    if isinstance(pair_prior_raw, dict):
        for k, v in pair_prior_raw.items():
            if not isinstance(k, str) or "|" not in k:
                continue
            a_str, b_str = k.split("|", 1)
            try:
                a = int(a_str)
                b = int(b_str)
            except ValueError:
                continue
            pair_prior[_pair_key(a, b)] = float(v)

    for task in beam_data.get("tasks", []):
        beam = dict(task.get("beam", {}))
        top_candidates = beam.get("final_top_states") or beam.get("final_top4_states") or beam.get("global_top_beam_states", [])[: int(args.rerank_candidate_count)]
        enriched = []

        for state in top_candidates[: int(args.rerank_candidate_count)]:
            indices = [int(x) for x in state.get("selected_indices_47153space", [])]
            if not indices:
                enriched.append({**state, "rerank_score": 0.0})
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
            pair_recovery_vals = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pair_key = _pair_key(indices[i], indices[j])
                    edge = edge_lookup.get(pair_key)
                    if edge:
                        co_rates.append(float(edge.get("cooccur_count", 0.0)) / max_cooccur)
                        pair_drop_rates.append(float(edge.get("drop_rate", 0.0)))
                        pair_continue_gains.append(float(edge.get("avg_continuation_gain", 0.0)))
                    pair_recovery_vals.append(float(pair_prior.get(pair_key, 0.0)))

            avg_appear = (sum(skill_appear) / len(skill_appear)) if skill_appear else 0.0
            avg_drop = (sum(skill_drop) / len(skill_drop)) if skill_drop else 0.0
            avg_displaced = (sum(skill_displaced) / len(skill_displaced)) if skill_displaced else 0.0
            avg_co = (sum(co_rates) / len(co_rates)) if co_rates else 0.0
            avg_pair_drop = (sum(pair_drop_rates) / len(pair_drop_rates)) if pair_drop_rates else 0.0
            avg_pair_continue_gain = (sum(pair_continue_gains) / len(pair_continue_gains)) if pair_continue_gains else 0.0

            hkey = tuple(sorted(indices))
            hinfo = hyper_lookup.get(hkey, {})
            hyper_support = float(hinfo.get("cooccur_count", 0.0)) / max_hyper_cooccur if hinfo else 0.0

            node_stability = float(avg_appear - 0.5 * avg_drop - 0.5 * avg_displaced)
            edge_mean = float(avg_co)
            edge_worst = float(min(co_rates)) if co_rates else 0.0
            edge_consistency = float(avg_co - 0.5 * avg_pair_drop)
            edge_progress = float(avg_pair_continue_gain)
            edge_recovery_prior = float(sum(pair_recovery_vals) / len(pair_recovery_vals)) if pair_recovery_vals else 0.0
            feature_bank = {
                "node_stability": float(node_stability),
                "edge_mean": float(edge_mean),
                "edge_worst": float(edge_worst),
                "edge_consistency": float(edge_consistency),
                "hyper_support": float(hyper_support),
                "edge_progress": float(edge_progress),
                "edge_recovery_prior": float(edge_recovery_prior),
                "hyper_gap": float(hyper_support - edge_mean),
                "bottleneck_gap": float(edge_mean - edge_worst),
            }
            feat = _resolve_feature_vector(cfg, feature_bank)
            rerank_score = _mlp_score(feat, cfg)
            enriched.append(
                {
                    **state,
                    "rerank_score": float(rerank_score),
                    "_edge_mean": float(edge_mean),
                    "_hyper_support": float(hyper_support),
                    "_edge_progress": float(edge_progress),
                    "_edge_recovery_prior": float(edge_recovery_prior),
                }
            )

        enriched.sort(key=lambda s: float(s.get("rerank_score", 0.0)), reverse=True)
        beam["final_top4_states_reranked"] = enriched[:4]
        beam["final_top4_reranked_indices_47153space"] = [
            [int(x) for x in s.get("selected_indices_47153space", [])] for s in enriched[:4]
        ]
        beam["final_top_states_reranked"] = enriched
        beam["final_top_reranked_indices_47153space"] = [
            [int(x) for x in s.get("selected_indices_47153space", [])] for s in enriched
        ]
        if enriched:
            top = enriched[0]
            beam["selected_indices_47153space_after_rerank"] = [int(x) for x in top.get("selected_indices_47153space", [])]
            beam["selected_count_after_rerank"] = len(beam["selected_indices_47153space_after_rerank"])
            beam["final_bundle_score_after_rerank"] = float(top.get("bundle_score", 0.0))
            beam["final_semantic_score_after_rerank"] = float(top.get("semantic_score", 0.0))
            beam["final_structural_score_after_rerank"] = float(top.get("structural_score", 0.0))
            beam["final_risk_penalty_after_rerank"] = float(top.get("risk_penalty", 0.0))
            beam["final_token_penalty_after_rerank"] = float(top.get("token_penalty", 0.0))
        task["beam"] = beam

    beam_data["gold_aware_rerank_config"] = cfg
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(beam_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
