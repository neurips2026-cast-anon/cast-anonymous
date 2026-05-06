#!/usr/bin/env python3
"""
Search task-specific skill bundles from full skill library via beam search.

Logic:
1. top candidates still come from task-skill cosine similarity
2. initial beam is seeded by the highest-ranked `beam_size` single-skill bundles
3. each round expands every beam state by adding one unseen skill
4. all expanded bundles are scored, top `beam_size` are kept
5. stop when the aggregate gain from previous top-k to new top-k is too small
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bundle_scorer_stage05 import l2norm_rows, l2norm_vec, score_bundle


@dataclass
class BeamState:
    indices: tuple[int, ...]
    result: Any


def _norm_skill_name(raw: str) -> str:
    return " ".join((raw or "").strip().lower().split())


def _canonicalize_indices(indices: tuple[int, ...], skills_json: list[dict[str, Any]]) -> tuple[int, ...]:
    seen_names: set[str] = set()
    kept: list[int] = []
    for idx in indices:
        if not (0 <= int(idx) < len(skills_json)):
            continue
        name = _norm_skill_name(str(skills_json[int(idx)].get("name") or skills_json[int(idx)].get("skill_name") or idx))
        if name in seen_names:
            continue
        seen_names.add(name)
        kept.append(int(idx))
    return tuple(kept)


def _score_state(
    task_vec: np.ndarray,
    skill_emb: np.ndarray,
    indices: tuple[int, ...],
    semantic_weight: float,
    structural_weight: float,
    scorer_config: dict[str, float] | None = None,
    complement_low: float = 0.35,
    redundancy_threshold: float = 0.85,
    token_cost_values: np.ndarray | None = None,
    permission_risk_values: np.ndarray | None = None,
    risk_lambda: float = 0.01,
    token_lambda: float = 0.005,
):
    trial_vecs = skill_emb[np.asarray(indices, dtype=np.int64)]
    return score_bundle(
        task_vec=task_vec,
        skill_vecs=trial_vecs,
        semantic_weight=semantic_weight,
        structural_weight=structural_weight,
        complement_low=complement_low,
        redundancy_threshold=redundancy_threshold,
        scorer_config=scorer_config,
        token_cost_values=token_cost_values,
        permission_risk_values=permission_risk_values,
        risk_lambda=risk_lambda,
        token_lambda=token_lambda,
    )


def _state_payload(state: BeamState) -> dict[str, Any]:
    r = state.result
    return {
        "selected_indices_47153space": [int(x) for x in state.indices],
        "selected_count": len(state.indices),
        "bundle_score": float(r.final_score),
        "skill_task_score": float(r.skill_task_score),
        "bundle_task_score": float(r.bundle_task_score),
        "task_bundle_score": float(r.task_bundle_score),
        "attention_score": float(r.attention_score),
        "semantic_score": float(r.semantic_score),
        "structural_score": float(r.structural_score),
        "risk_penalty": float(r.risk_penalty),
        "token_penalty": float(r.token_penalty),
        "length_penalty": float(r.length_penalty),
        "p_syn": float(r.p_syn),
        "p_red": float(r.p_red),
        "attention_weights": [float(x) for x in r.attention_weights.tolist()],
    }


def _sort_states(states: list[BeamState]) -> list[BeamState]:
    return sorted(
        states,
        key=lambda s: (
            float(s.result.final_score),
            float(s.result.semantic_score),
            -len(s.indices),
            tuple(-x for x in s.indices),
        ),
        reverse=True,
    )


def _dedupe_states(states: list[BeamState], skills_json: list[dict[str, Any]]) -> list[BeamState]:
    kept: dict[tuple[int, ...], BeamState] = {}
    for state in _sort_states(states):
        key = _canonicalize_indices(state.indices, skills_json)
        if key and key not in kept:
            kept[key] = BeamState(indices=key, result=state.result)
    return _sort_states(list(kept.values()))


def _build_rescue_pair_pool(
    task_vec: np.ndarray,
    candidates: list[dict[str, Any]],
    skills_json: list[dict[str, Any]],
    skill_emb: np.ndarray,
    *,
    semantic_weight: float,
    structural_weight: float,
    scorer_config: dict[str, float] | None,
    complement_low: float,
    redundancy_threshold: float,
    risk_lambda: float,
    token_lambda: float,
    topn_seed: int,
) -> list[BeamState]:
    seeds = candidates[: max(2, int(topn_seed))]
    rescue_states: list[BeamState] = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            idx_a = int(seeds[i]["skill_index_47153"])
            idx_b = int(seeds[j]["skill_index_47153"])
            indices = (idx_a, idx_b)
            token_vals = np.asarray(
                [float(seeds[i].get("token_cost_value", 0.0)), float(seeds[j].get("token_cost_value", 0.0))],
                dtype=np.float32,
            )
            risk_vals = np.asarray(
                [float(seeds[i].get("permission_risk_value", 0.0)), float(seeds[j].get("permission_risk_value", 0.0))],
                dtype=np.float32,
            )
            result = _score_state(
                task_vec=task_vec,
                skill_emb=skill_emb,
                indices=indices,
                semantic_weight=semantic_weight,
                structural_weight=structural_weight,
                scorer_config=scorer_config,
                complement_low=complement_low,
                redundancy_threshold=redundancy_threshold,
                token_cost_values=token_vals,
                permission_risk_values=risk_vals,
                risk_lambda=risk_lambda,
                token_lambda=token_lambda,
            )
            rescue_states.append(BeamState(indices=indices, result=result))
    return _dedupe_states(rescue_states, skills_json)


def _resolve_scorer_config_path(raw_path: str) -> str:
    paper_root = Path(__file__).resolve().parents[1]
    if raw_path:
        cfg_path = Path(raw_path)
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        else:
            cfg_path = cfg_path.resolve()
        try:
            cfg_path.relative_to(paper_root)
        except ValueError as exc:
            raise ValueError(
                f"--scorer-config must be inside paper root: {paper_root}, got {cfg_path}"
            ) from exc
        return str(cfg_path)
    default_path = paper_root / "data" / "default_bundle_scorer_config.json"
    if default_path.exists():
        return str(default_path)
    return raw_path


def main() -> None:
    paper_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build task bundles from full skill library via beam search.")
    parser.add_argument("--skills-json", default=str(paper_root / "data" / "bundle_input" / "skills_with_risk.json"))
    parser.add_argument("--task-json", default=str(paper_root / "data" / "tasks_all_metadata_for_embedding_v2.json"))
    parser.add_argument("--task-emb", default=str(paper_root / "data" / "task_embeddings_bigmodel_512_f32_v2.npy"))
    parser.add_argument("--skill-emb", default=str(paper_root / "data" / "bundle_input" / "skills_embedding.npy"))
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=20,
        help="Top-k candidate pool size by task-skill similarity before beam expansion.",
    )
    parser.add_argument("--beam-size", type=int, default=9)
    parser.add_argument(
        "--final-candidate-count",
        type=int,
        default=6,
        help="How many final beam states to retain for downstream reranking.",
    )
    parser.add_argument(
        "--final-selection-mode",
        choices=("global_topk", "diverse_topk"),
        default="global_topk",
        help="How to retain final candidates: plain top-k or ensure each length keeps at least one state first.",
    )
    parser.add_argument("--stop-threshold", type=float, default=0.25)
    parser.add_argument("--min-bundle-size", type=int, default=2)
    parser.add_argument("--semantic-weight", type=float, default=0.70)
    parser.add_argument("--structural-weight", type=float, default=0.30)
    parser.add_argument("--complement-low", type=float, default=0.35)
    parser.add_argument("--redundancy-threshold", type=float, default=0.85)
    parser.add_argument("--risk-lambda", type=float, default=0.01)
    parser.add_argument("--token-lambda", type=float, default=0.005)
    parser.add_argument(
        "--min-balanced-gain",
        type=float,
        default=-0.75,
        help="Allow slight score drops during bundle growth. Expansions with gain below this threshold are pruned once bundle size exceeds min_bundle_size.",
    )
    parser.add_argument("--output", default=str(paper_root / "output" / "beam_search.json"))
    parser.add_argument("--scorer-config", default="")
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

    skills_json_path = _resolve_under_paper(args.skills_json, allow_missing=False)
    task_json_path = _resolve_under_paper(args.task_json, allow_missing=False)
    task_emb_path = _resolve_under_paper(args.task_emb, allow_missing=False)
    skill_emb_path = _resolve_under_paper(args.skill_emb, allow_missing=False)
    output_path = _resolve_under_paper(args.output, allow_missing=True)

    skills_json = json.loads(skills_json_path.read_text(encoding="utf-8"))
    tasks = json.loads(task_json_path.read_text(encoding="utf-8"))
    task_emb = l2norm_rows(np.load(task_emb_path).astype(np.float32, copy=False))
    skill_emb = l2norm_rows(np.load(skill_emb_path).astype(np.float32, copy=False))
    if len(tasks) != task_emb.shape[0]:
        raise ValueError(
            f"task count mismatch: task json has {len(tasks)}, task embedding has {task_emb.shape[0]}"
        )
    if len(skills_json) != skill_emb.shape[0]:
        raise ValueError(
            f"skill count mismatch: skills json has {len(skills_json)}, skill embedding has {skill_emb.shape[0]}"
        )
    resolved_scorer_config = _resolve_scorer_config_path(args.scorer_config)
    scorer_config = None
    if resolved_scorer_config:
        scorer_config = json.loads(Path(resolved_scorer_config).read_text(encoding="utf-8"))

    out_tasks: list[dict[str, Any]] = []

    for i, task in enumerate(tasks):
        task_label = str(task.get("task_id", f"task_{i+1}"))
        task_vec = l2norm_vec(task_emb[i])
        sim_to_all = skill_emb @ task_vec
        k = min(int(args.max_candidates), int(skill_emb.shape[0]))
        if k <= 0:
            raise ValueError("--max-candidates must be >= 1")
        if k == skill_emb.shape[0]:
            top_idx = np.argsort(-sim_to_all)
        else:
            part = np.argpartition(-sim_to_all, k - 1)[:k]
            top_idx = part[np.argsort(-sim_to_all[part])]

        candidates: list[dict[str, Any]] = []
        seen_candidate_names: set[str] = set()
        missing_risk_label = 0
        missing_token_label = 0
        risk_lookup: dict[str, float] = {"A": 0.0, "B": 0.5, "C": 1.0, "UNKNOWN": 0.0}
        token_lookup: dict[str, float] = {"A": 0.0, "B": 0.25, "C": 0.5, "UNKNOWN": 0.0}
        for rank, si in enumerate(top_idx.tolist(), start=1):
            skill_row = skills_json[int(si)]
            canonical_name = _norm_skill_name(str(skill_row.get("name") or skill_row.get("skill_name") or si))
            if canonical_name in seen_candidate_names:
                continue
            seen_candidate_names.add(canonical_name)
            raw_risk_level = skill_row.get("permission_risk_level")
            raw_token_level = skill_row.get("token_cost_level")
            if raw_risk_level in (None, "", []):
                missing_risk_label += 1
            if raw_token_level in (None, "", []):
                missing_token_label += 1
            risk_level = str(raw_risk_level).upper() if raw_risk_level not in (None, "", []) else "UNKNOWN"
            token_level = str(raw_token_level).upper() if raw_token_level not in (None, "", []) else "UNKNOWN"
            candidates.append(
                {
                    "rank_in_candidate_pool": rank,
                    "skill_index_47153": int(si),
                    "sim_to_task": float(sim_to_all[int(si)]),
                    "original_task_skill_score": float(sim_to_all[int(si)]),
                    # Legacy aliases kept only for backward compatibility with old analyzers.
                    # These values are task-similarity, not true gold-similarity.
                    "sim_to_gold_target": float(sim_to_all[int(si)]),
                    "sim_to_gold_max": float(sim_to_all[int(si)]),
                    "sim_to_gold_mean": float(sim_to_all[int(si)]),
                    "sim_to_each_gold_skill": {},
                    "legacy_gold_similarity_alias": True,
                    "permission_risk_level": risk_level,
                    "token_cost_level": token_level,
                    "permission_risk_value": float(skill_row.get("permission_risk_value", risk_lookup.get(risk_level, 0.0))),
                    "token_cost_value": float(skill_row.get("token_cost_value", token_lookup.get(token_level, 0.0))),
                    "permission_risk_note": str(skill_row.get("permission_risk_note", "")),
                    "token_cost_note": str(skill_row.get("token_cost_note", "")),
                    "review_status": str(skill_row.get("review_status", "auto_full_library")),
                }
            )
        candidate_by_index = {int(c["skill_index_47153"]): c for c in candidates}

        beam_steps: list[dict[str, Any]] = []
        initial_candidates = candidates[: max(0, args.beam_size)]
        current_beam: list[BeamState] = []
        for seed in initial_candidates:
            si = int(seed["skill_index_47153"])
            seed_token = np.asarray([float(seed.get("token_cost_value", 0.0))], dtype=np.float32)
            seed_risk = np.asarray([float(seed.get("permission_risk_value", 0.0))], dtype=np.float32)
            result = _score_state(
                task_vec=task_vec,
                skill_emb=skill_emb,
                indices=(si,),
                semantic_weight=args.semantic_weight,
                structural_weight=args.structural_weight,
                scorer_config=scorer_config,
                complement_low=args.complement_low,
                redundancy_threshold=args.redundancy_threshold,
                token_cost_values=seed_token,
                permission_risk_values=seed_risk,
                risk_lambda=args.risk_lambda,
                token_lambda=args.token_lambda,
            )
            current_beam.append(BeamState(indices=_canonicalize_indices((si,), skills_json), result=result))

        current_beam = _dedupe_states(current_beam, skills_json)[: args.beam_size]
        global_top_states: list[BeamState] = current_beam[:]
        beam_steps.append(
            {
                "round": 1,
                "action": "seed",
                "beam_size": int(args.beam_size),
                "topk_total_score": float(sum(s.result.final_score for s in current_beam)),
                "global_topk_total_score": float(sum(s.result.final_score for s in global_top_states)),
                "states": [_state_payload(s) for s in current_beam],
                "global_top_states": [_state_payload(s) for s in global_top_states],
                "score_type": "bundle_encoder_score",
            }
        )

        round_idx = 1
        stop_reason = "exhausted_candidates"

        while current_beam:
            expansions: list[BeamState] = []
            seen_state_indices: set[tuple[int, ...]] = set()

            for state in current_beam:
                selected = set(state.indices)
                for cand in candidates:
                    si = int(cand["skill_index_47153"])
                    if si in selected:
                        continue
                    new_indices = _canonicalize_indices(tuple(list(state.indices) + [si]), skills_json)
                    if len(new_indices) != len(state.indices) + 1:
                        continue
                    if new_indices in seen_state_indices:
                        continue
                    seen_state_indices.add(new_indices)
                    selected_candidate_items = [
                        candidate_by_index[int(idx)]
                        for idx in new_indices
                    ]
                    token_vals = np.asarray(
                        [float(c.get("token_cost_value", 0.0)) for c in selected_candidate_items],
                        dtype=np.float32,
                    )
                    risk_vals = np.asarray(
                        [float(c.get("permission_risk_value", 0.0)) for c in selected_candidate_items],
                        dtype=np.float32,
                    )
                    result = _score_state(
                        task_vec=task_vec,
                        skill_emb=skill_emb,
                        indices=new_indices,
                        semantic_weight=args.semantic_weight,
                        structural_weight=args.structural_weight,
                        scorer_config=scorer_config,
                        complement_low=args.complement_low,
                        redundancy_threshold=args.redundancy_threshold,
                        token_cost_values=token_vals,
                        permission_risk_values=risk_vals,
                        risk_lambda=args.risk_lambda,
                        token_lambda=args.token_lambda,
                    )
                    final_gain = float(result.final_score - state.result.final_score)
                    if len(new_indices) > args.min_bundle_size and final_gain < float(args.min_balanced_gain):
                        continue
                    expansions.append(BeamState(indices=new_indices, result=result))

            if not expansions:
                break

            next_beam = _sort_states(expansions)[: args.beam_size]
            prev_total = float(sum(s.result.final_score for s in current_beam))
            next_total = float(sum(s.result.final_score for s in next_beam))
            total_gain = next_total - prev_total
            round_idx += 1
            global_top_states = _dedupe_states(global_top_states + next_beam, skills_json)[: args.beam_size]
            global_total = float(sum(s.result.final_score for s in global_top_states))

            beam_steps.append(
                {
                    "round": int(round_idx),
                    "action": "expand",
                    "previous_topk_total_score": prev_total,
                    "new_topk_total_score": next_total,
                    "global_topk_total_score": global_total,
                    "topk_total_gain": float(total_gain),
                    "threshold": float(args.stop_threshold),
                    "states": [_state_payload(s) for s in next_beam],
                    "global_top_states": [_state_payload(s) for s in global_top_states],
                    "score_type": "bundle_encoder_score",
                }
            )

            best_bundle_size = max((len(s.indices) for s in next_beam), default=0)
            if total_gain < args.stop_threshold and best_bundle_size >= args.min_bundle_size:
                stop_reason = "topk_gain_below_threshold"
                break

            current_beam = next_beam

        final_beam = _sort_states(current_beam)[: args.beam_size]
        global_top_states = _dedupe_states(global_top_states + final_beam, skills_json)[: args.beam_size]
        eligible_states = [s for s in global_top_states if len(s.indices) >= args.min_bundle_size]
        if not eligible_states:
            eligible_states = [s for s in final_beam if len(s.indices) >= args.min_bundle_size]
        if len(eligible_states) < max(2, int(args.final_candidate_count)) and len(candidates) >= args.min_bundle_size:
            rescue_states = _build_rescue_pair_pool(
                task_vec=task_vec,
                candidates=candidates,
                skills_json=skills_json,
                skill_emb=skill_emb,
                semantic_weight=args.semantic_weight,
                structural_weight=args.structural_weight,
                scorer_config=scorer_config,
                complement_low=args.complement_low,
                redundancy_threshold=args.redundancy_threshold,
                risk_lambda=args.risk_lambda,
                token_lambda=args.token_lambda,
                topn_seed=min(12, len(candidates)),
            )
            if rescue_states:
                eligible_states = _dedupe_states(eligible_states + rescue_states, skills_json)
                if stop_reason == "exhausted_candidates":
                    stop_reason = "rescue_pair_pool"
                elif stop_reason == "topk_gain_below_threshold":
                    stop_reason = "topk_gain_below_threshold_with_rescue"

        if args.final_selection_mode == "diverse_topk":
            best_per_len: dict[int, BeamState] = {}
            for state in eligible_states:
                l = len(state.indices)
                if l not in best_per_len or float(state.result.final_score) > float(best_per_len[l].result.final_score):
                    best_per_len[l] = state
            diverse_seed = _sort_states(list(best_per_len.values()))
            diverse_keys = {tuple(s.indices) for s in diverse_seed}
            remaining = [s for s in _sort_states(eligible_states) if tuple(s.indices) not in diverse_keys]
            final_candidate_states = (diverse_seed + remaining)[: max(1, int(args.final_candidate_count))]
        else:
            final_candidate_states = _sort_states(eligible_states)[: max(1, int(args.final_candidate_count))]

        best_state = final_candidate_states[0] if final_candidate_states else None

        if best_state is None:
            final_bundle_score = 0.0
            final_semantic_score = 0.0
            final_structural_score = 0.0
            final_p_syn = 0.0
            final_p_red = 0.0
            final_attention_weights: list[float] = []
            selected_idx: list[int] = []
            final_top_states: list[dict[str, Any]] = []
            final_top_indices: list[list[int]] = []
            final_top_scores: list[float] = []
            final_top_lengths: list[int] = []
        else:
            final_bundle_score = float(best_state.result.final_score)
            final_semantic_score = float(best_state.result.semantic_score)
            final_structural_score = float(best_state.result.structural_score)
            final_p_syn = float(best_state.result.p_syn)
            final_p_red = float(best_state.result.p_red)
            final_attention_weights = [float(x) for x in best_state.result.attention_weights.tolist()]
            selected_idx = [int(x) for x in best_state.indices]
            final_top_states = [_state_payload(s) for s in final_candidate_states]
            final_top_indices = [[int(x) for x in s.indices] for s in final_candidate_states]
            final_top_scores = [float(s.result.final_score) for s in final_candidate_states]
            final_top_lengths = [len(s.indices) for s in final_candidate_states]

        out_tasks.append(
            {
                "task_index": i,
                "task_id": task.get("id"),
                "task_label": task_label,
                "candidates_count": len(candidates),
                "missing_permission_risk_labels": int(missing_risk_label),
                "missing_token_cost_labels": int(missing_token_label),
                "candidates_sorted_by_task_similarity": candidates,
                # Legacy key retained for compatibility. Prefer candidates_sorted_by_task_similarity.
                "candidates_sorted_by_gold_similarity": candidates,
                "beam": {
                    "beam_size": int(args.beam_size),
                    "stop_threshold": float(args.stop_threshold),
                    "min_bundle_size": int(args.min_bundle_size),
                    "stop_reason": stop_reason,
                    "selected_indices_47153space": selected_idx,
                    "selected_count": len(selected_idx),
                    "final_top4_selected_indices_47153space": final_top_indices[:4],
                    "final_top4_count": min(4, len(final_top_indices)),
                    "final_top4_scores": final_top_scores[:4],
                    "final_top4_lengths": final_top_lengths[:4],
                    "final_top4_states": final_top_states[:4],
                    "final_candidate_count": len(final_top_indices),
                    "final_top_states": final_top_states,
                    "final_top_selected_indices_47153space": final_top_indices,
                    "final_top_scores": final_top_scores,
                    "final_top_lengths": final_top_lengths,
                    "final_bundle_score": final_bundle_score,
                    "final_skill_task_score": float(best_state.result.skill_task_score) if best_state is not None else 0.0,
                    "final_bundle_task_score": float(best_state.result.bundle_task_score) if best_state is not None else 0.0,
                    "final_task_bundle_score": float(best_state.result.task_bundle_score) if best_state is not None else 0.0,
                    "final_attention_score": float(best_state.result.attention_score) if best_state is not None else 0.0,
                    "final_semantic_score": final_semantic_score,
                    "final_structural_score": final_structural_score,
                    "final_length_penalty": float(best_state.result.length_penalty) if best_state is not None else 0.0,
                    "final_risk_penalty": float(best_state.result.risk_penalty) if best_state is not None else 0.0,
                    "final_token_penalty": float(best_state.result.token_penalty) if best_state is not None else 0.0,
                    "final_p_syn": final_p_syn,
                    "final_p_red": final_p_red,
                    "final_attention_weights": final_attention_weights,
                    "final_score_to_task": final_semantic_score,
                    # Legacy field name kept for compatibility with old reports.
                    "final_score_to_gold_target": final_bundle_score,
                    "global_best_selected_indices_47153space": [int(x) for x in best_state.indices] if best_state is not None else [],
                    "global_best_round_agnostic": True,
                    "global_top_beam_states": [_state_payload(s) for s in global_top_states],
                    "final_beam_states": [_state_payload(s) for s in final_beam],
                    "steps": beam_steps,
                },
            }
        )

    out = {
        "inputs": {
            "skills_json": str(skills_json_path),
            "scorer_config_json": resolved_scorer_config,
            "task_json": str(task_json_path),
            "task_emb": str(task_emb_path),
            "skill_emb": str(skill_emb_path),
        },
        "config": {
            "candidate_source": "full_skill_library",
            "candidate_metric": "task_skill_cosine",
            "legacy_fields_note": "gold_similarity-named fields are compatibility aliases and actually store task-similarity based values.",
            "bundle_selection_metric": "bundle_encoder_score",
            "max_candidates": int(args.max_candidates),
            "beam_size": int(args.beam_size),
            "final_candidate_count": int(args.final_candidate_count),
            "final_selection_mode": args.final_selection_mode,
            "stop_threshold": float(args.stop_threshold),
            "semantic_weight": float(args.semantic_weight),
            "structural_weight": float(args.structural_weight),
            "complement_low": float(args.complement_low),
            "redundancy_threshold": float(args.redundancy_threshold),
            "risk_lambda": float(args.risk_lambda),
            "token_lambda": float(args.token_lambda),
            "min_balanced_gain": float(args.min_balanced_gain),
        },
        "tasks": out_tasks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved beam: {output_path}")


if __name__ == "__main__":
    main()
