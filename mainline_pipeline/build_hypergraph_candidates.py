#!/usr/bin/env python3
"""
Build a hypergraph view of beam top-4 bundles per round.

Nodes: skills
Edges: skill co-occurrence inside top-4 bundles (pairwise + sparse bundle hyperedges)
Also records per-skill appearance frequency and drop signals between rounds.

This version emphasizes pairwise structure:
- raw co-occurrence count
- conditional support
- PMI / NPMI
- sparse hyperedges only when repeated enough
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _sorted_key(indices: list[int]) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in indices))


def _mean_or_zero(total: float, count: int) -> float:
    return float(total / count) if count else 0.0


def _safe_log(x: float) -> float:
    import math

    return math.log(max(x, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build top-4 beam hypergraph from beam-search traces.")
    parser.add_argument("--beam-json", default="../output/beam_search.json")
    parser.add_argument("--skills-json", default="../data/bundle_input/skills_with_risk.json")
    parser.add_argument("--output-json", default="../output/hypergraph_candidates.json")
    parser.add_argument("--min-hyperedge-count", type=int, default=2)
    args = parser.parse_args()

    beam_data = json.loads(Path(args.beam_json).read_text(encoding="utf-8"))
    skills = json.loads(Path(args.skills_json).read_text(encoding="utf-8"))

    node_stats: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "appear_rounds": 0,
            "total_rounds": 0,
            "first_round": None,
            "last_round": None,
            "drop_count": 0,
            "below_cutoff_count": 0,
            "displaced_count": 0,
        }
    )
    edge_stats: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {
            "cooccur_count": 0,
            "bundle_score_sum": 0.0,
            "semantic_score_sum": 0.0,
            "structural_score_sum": 0.0,
            "attention_score_sum": 0.0,
            "drop_count": 0,
            "below_cutoff_drop_count": 0,
            "displaced_drop_count": 0,
            "continue_count": 0,
            "continuation_gain_sum": 0.0,
        }
    )
    hyper_stats: dict[tuple[int, ...], dict[str, Any]] = defaultdict(
        lambda: {
            "cooccur_count": 0,
            "bundle_score_sum": 0.0,
        }
    )
    total_rounds_observed = 0

    for task_entry in beam_data.get("tasks", []):
        steps = task_entry.get("beam", {}).get("steps", []) or []
        prev_round_skills: set[int] = set()
        prev_round_best_score: dict[int, float] = {}
        prev_round_pairs: set[tuple[int, int]] = set()
        prev_pair_best_score: dict[tuple[int, int], float] = {}

        for step in steps:
            states = step.get("states", []) or []
            if not states:
                continue

            round_num = int(step.get("round", 0))
            total_rounds_observed += 1
            round_cutoff = min(float(s.get("bundle_score", 0.0)) for s in states)

            round_skills: set[int] = set()
            round_best_score: dict[int, float] = {}
            round_pairs: set[tuple[int, int]] = set()
            round_pair_best_score: dict[tuple[int, int], float] = {}

            for state in states:
                indices = [int(x) for x in state.get("selected_indices_47153space", [])]
                if not indices:
                    continue
                score = float(state.get("bundle_score", 0.0))
                semantic = float(state.get("semantic_score", 0.0))
                structural = float(state.get("structural_score", 0.0))
                attention = float(state.get("attention_score", 0.0))

                for idx in indices:
                    round_skills.add(idx)
                    if idx not in round_best_score or score > round_best_score[idx]:
                        round_best_score[idx] = score

                # pairwise edges
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        key = _pair_key(indices[i], indices[j])
                        round_pairs.add(key)
                        if key not in round_pair_best_score or score > round_pair_best_score[key]:
                            round_pair_best_score[key] = score
                        es = edge_stats[key]
                        es["cooccur_count"] += 1
                        es["bundle_score_sum"] += score
                        es["semantic_score_sum"] += semantic
                        es["structural_score_sum"] += structural
                        es["attention_score_sum"] += attention

                # hyperedge (bundle as a whole)
                hkey = _sorted_key(indices)
                hs = hyper_stats[hkey]
                hs["cooccur_count"] += 1
                hs["bundle_score_sum"] += score

            # update node stats
            for idx, st in node_stats.items():
                st["total_rounds"] += 1
                if idx in round_skills:
                    st["appear_rounds"] += 1
                    if st["first_round"] is None or round_num < st["first_round"]:
                        st["first_round"] = round_num
                    st["last_round"] = round_num

            # drop signals (skill present last round but not this round)
            dropped = prev_round_skills - round_skills
            for idx in dropped:
                st = node_stats[idx]
                st["drop_count"] += 1
                last_score = prev_round_best_score.get(idx, 0.0)
                if last_score < round_cutoff:
                    st["below_cutoff_count"] += 1
                else:
                    st["displaced_count"] += 1

            # pair-level drop/displacement signals
            dropped_pairs = prev_round_pairs - round_pairs
            for key in dropped_pairs:
                es = edge_stats[key]
                es["drop_count"] += 1
                last_score = prev_pair_best_score.get(key, 0.0)
                if last_score < round_cutoff:
                    es["below_cutoff_drop_count"] += 1
                else:
                    es["displaced_drop_count"] += 1

            # pair-level continuation signals: pair survives to next round and still participates in better bundles
            continued_pairs = prev_round_pairs & round_pairs
            for key in continued_pairs:
                es = edge_stats[key]
                es["continue_count"] += 1
                es["continuation_gain_sum"] += float(round_pair_best_score.get(key, 0.0) - prev_pair_best_score.get(key, 0.0))

            prev_round_skills = round_skills
            prev_round_best_score = round_best_score
            prev_round_pairs = round_pairs
            prev_pair_best_score = round_pair_best_score

    # build output
    skill_nodes: list[dict[str, Any]] = []
    for idx, st in sorted(node_stats.items()):
        meta = skills[idx] if 0 <= idx < len(skills) else {}
        appear = int(st["appear_rounds"])
        total = int(st["total_rounds"])
        skill_nodes.append(
            {
                "skill_index": int(idx),
                "skill_name": meta.get("name"),
                "description": meta.get("description"),
                "appear_rounds": appear,
                "total_rounds": total,
                "appear_rate": float(appear / total) if total else 0.0,
                "first_round": st["first_round"],
                "last_round": st["last_round"],
                "drop_count": int(st["drop_count"]),
                "below_cutoff_count": int(st["below_cutoff_count"]),
                "displaced_count": int(st["displaced_count"]),
            }
        )

    edges: list[dict[str, Any]] = []
    for (a, b), es in sorted(edge_stats.items()):
        c = int(es["cooccur_count"])
        a_rounds = int(node_stats[a]["appear_rounds"])
        b_rounds = int(node_stats[b]["appear_rounds"])
        total = max(int(total_rounds_observed), 1)
        p_ab = c / total
        p_a = max(a_rounds / total, 1e-12)
        p_b = max(b_rounds / total, 1e-12)
        pmi = _safe_log(p_ab / max(p_a * p_b, 1e-12))
        npmi = pmi / max(-_safe_log(p_ab), 1e-12) if p_ab > 0 else 0.0
        edges.append(
            {
                "src_skill_index": int(a),
                "dst_skill_index": int(b),
                "cooccur_count": c,
                "support": float(c / total),
                "conditional_src_to_dst": float(c / max(a_rounds, 1)),
                "conditional_dst_to_src": float(c / max(b_rounds, 1)),
                "pmi": float(pmi),
                "npmi": float(npmi),
                "drop_rate": float(es["drop_count"] / max(c, 1)),
                "below_cutoff_drop_rate": float(es["below_cutoff_drop_count"] / max(c, 1)),
                "displaced_drop_rate": float(es["displaced_drop_count"] / max(c, 1)),
                "continue_rate": float(es["continue_count"] / max(c, 1)),
                "avg_continuation_gain": _mean_or_zero(es["continuation_gain_sum"], int(es["continue_count"])),
                "avg_bundle_score": _mean_or_zero(es["bundle_score_sum"], c),
                "avg_semantic_score": _mean_or_zero(es["semantic_score_sum"], c),
                "avg_structural_score": _mean_or_zero(es["structural_score_sum"], c),
                "avg_attention_score": _mean_or_zero(es["attention_score_sum"], c),
            }
        )

    hyperedges: list[dict[str, Any]] = []
    for key, hs in sorted(hyper_stats.items()):
        c = int(hs["cooccur_count"])
        if c < int(args.min_hyperedge_count):
            continue
        hyperedges.append(
            {
                "skill_indices": list(key),
                "arity": len(key),
                "cooccur_count": c,
                "avg_bundle_score": _mean_or_zero(hs["bundle_score_sum"], c),
            }
        )

    out = {
        "description": "Top-4 beam hypergraph (per round). Nodes are skills; edges are co-occurrence in top-4 bundles.",
        "inputs": {"beam_json": args.beam_json},
        "summary": {
            "observed_round_count": int(total_rounds_observed),
            "skill_node_count": len(skill_nodes),
            "edge_count": len(edges),
            "hyperedge_count": len(hyperedges),
        },
        "skill_nodes": skill_nodes,
        "edges": edges,
        "hyperedges": hyperedges,
        "limitations": [
            "Drop reasons are approximated by comparing last-seen bundle_score to round cutoff.",
            "Only top-4 bundle states are observed; rejected expansions are not included.",
        ],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"skills={len(skill_nodes)} edges={len(edges)} hyperedges={len(hyperedges)}")


if __name__ == "__main__":
    main()
