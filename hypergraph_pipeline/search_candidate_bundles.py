#!/usr/bin/env python3
"""Beam search with the Typed LightGCN PET scorer."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cast_relevance3_model import TaskPETRelevance3Scorer, load_pet_graph_tensors, l2norm_rows


def load_dual_tower_class():
    path = Path(__file__).resolve().parent / "12_train_task_skill_retriever.py"
    spec = importlib.util.spec_from_file_location("task_skill_retriever_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.DualTowerRetriever


def load_residual_retriever_class():
    path = Path(__file__).resolve().parent / "14_train_residual_task_skill_retriever.py"
    spec = importlib.util.spec_from_file_location("residual_retriever_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.ResidualRetriever


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_vec_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), eps)


def skill_name(skills: list[dict[str, Any]], idx: int) -> str:
    if 0 <= idx < len(skills):
        return str(skills[idx].get("name") or f"skill_{idx}")
    return f"skill_{idx}"


def safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def coverage_score(task_vec: np.ndarray, skill_emb: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    sims = skill_emb[np.asarray(indices, dtype=np.int64)] @ task_vec
    # Additive task coverage: adding a relevant skill should create positive
    # marginal gain. Final ranking is still done by the PET scorer.
    return float(np.sum(np.clip(sims, 0.0, None)))


class InferencePETScorer:
    def __init__(
        self,
        *,
        model: TaskPETRelevance3Scorer,
        train_graph: Any,
        train_enc: dict[str, torch.Tensor],
        full_skill_emb: np.ndarray,
        skill_meta: list[dict[str, Any]],
        device: torch.device,
        retrieved_prompt_topm: int = 4,
    ) -> None:
        self.model = model
        self.graph = train_graph
        self.enc = train_enc
        self.device = device
        self.full_skill_emb = torch.tensor(full_skill_emb, dtype=torch.float32, device=device)
        self.full_skill_emb = l2norm_rows(self.full_skill_emb)
        self.skill_meta = skill_meta
        self.retrieved_prompt_topm = int(retrieved_prompt_topm)
        self.active_map = {
            int(g): i for i, g in enumerate(train_graph.active_skill_global_indices.detach().cpu().tolist())
        }

    def _skill_views(self, global_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = []
        ts = []
        bs = []
        for gid in global_indices:
            gid = int(gid)
            raw_vec = self.full_skill_emb[gid]
            raw.append(raw_vec)
            local = self.active_map.get(gid)
            if local is None:
                ts.append(raw_vec)
                bs.append(raw_vec)
            else:
                ts.append(self.enc["skill_ts"][local])
                bs.append(self.enc["skill_bs"][local])
        return torch.stack(raw, dim=0), torch.stack(ts, dim=0), torch.stack(bs, dim=0)

    def _bundle_features(self, global_indices: list[int], raw_skill_vecs: torch.Tensor) -> torch.Tensor:
        risks = []
        costs = []
        for gid in global_indices:
            meta = self.skill_meta[int(gid)] if 0 <= int(gid) < len(self.skill_meta) else {}
            risks.append(safe_float(meta.get("permission_risk_value", 0.0)))
            costs.append(safe_float(meta.get("token_cost_value", 0.0)))
        size = float(len(global_indices))
        risk_mean = float(sum(risks) / max(len(risks), 1))
        risk_max = float(max(risks) if risks else 0.0)
        cost_mean = float(sum(costs) / max(len(costs), 1))
        cost_max = float(max(costs) if costs else 0.0)
        if len(global_indices) > 1:
            sim = raw_skill_vecs @ raw_skill_vecs.T
            tri = sim[torch.triu_indices(sim.shape[0], sim.shape[1], offset=1, device=sim.device).unbind()]
            pair_mean = float(tri.mean().detach().cpu())
            pair_max = float(tri.max().detach().cpu())
            pair_min = float(tri.min().detach().cpu())
        else:
            pair_mean = pair_max = pair_min = 0.0
        return torch.tensor(
            [[size / 10.0, risk_mean, risk_max, cost_mean, cost_max, pair_mean, pair_max, pair_min]],
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def score(
        self,
        task_vec_np: np.ndarray,
        global_indices: list[int],
        *,
        prompt_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        return self.score_many(task_vec_np, [global_indices], prompt_indices=prompt_indices)[0] if global_indices else {"score": 0.0}

    @torch.no_grad()
    def score_many(
        self,
        task_vec_np: np.ndarray,
        bundles: list[list[int]],
        *,
        prompt_indices: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        if not bundles:
            return []
        max_len = max(len(b) for b in bundles)
        batch = len(bundles)
        ids = torch.zeros((batch, max_len), dtype=torch.long, device=self.device)
        mask = torch.zeros((batch, max_len), dtype=torch.float32, device=self.device)
        for i, b in enumerate(bundles):
            ids[i, : len(b)] = torch.tensor([int(x) for x in b], dtype=torch.long, device=self.device)
            mask[i, : len(b)] = 1.0

        raw_s = self.full_skill_emb[ids] * mask[:, :, None]
        t0 = torch.tensor(task_vec_np, dtype=torch.float32, device=self.device).reshape(1, -1)
        skill_context = None
        if getattr(self.model, "task_prompt_mode", "none") == "retrieved" and prompt_indices:
            valid_prompt = [int(x) for x in prompt_indices[: self.retrieved_prompt_topm] if 0 <= int(x) < self.full_skill_emb.shape[0]]
            if valid_prompt:
                skill_context = self.full_skill_emb[torch.tensor(valid_prompt, dtype=torch.long, device=self.device)].mean(dim=0, keepdim=True)
        feat_rows = [self._bundle_features(b, raw_s[i, : len(b)]) for i, b in enumerate(bundles)]
        feat = torch.cat(feat_rows, dim=0)
        final, stats = self.model.score_dynamic_bundles(
            l2norm_rows(t0),
            raw_s,
            mask,
            feat,
            skill_context=skill_context,
        )

        out = []
        for i, b in enumerate(bundles):
            view_scores = stats.get("view_scores")
            view_gate = stats.get("view_gate")
            experts = [float(x) for x in view_scores[i].detach().cpu().tolist()] if view_scores is not None else []
            gate = [float(x) for x in view_gate[i].detach().cpu().tolist()] if view_gate is not None else []
            out.append({
                "score": float(final[i].item()),
                "view_scores": experts,
                "view_gate": gate,
                "experts": experts,
                "gate": gate,
                "bundle_size": len(b),
            })
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="PET scorer beam search.")
    parser.add_argument("--task-json", default="../../data/tasks_all_metadata_for_embedding_v2.json")
    parser.add_argument("--task-emb", default="../../data/task_embeddings_bigmodel_512_f32_v2.npy")
    parser.add_argument("--skill-json", default="../../data/benchmark_merged_skills.json")
    parser.add_argument("--skill-emb", default="../../data/benchmark_merged_skill_embeddings.npy")
    parser.add_argument("--train-split-dir", default="../data/splits/train")
    parser.add_argument("--paper-data", default="../../data")
    parser.add_argument("--model-path", default="../output/typed_lightgcn_pet_l2.pt")
    parser.add_argument("--structure-mode", choices=("soft", "learned_red"), default="soft")
    parser.add_argument("--view-mode", choices=("full", "task_skill_only"), default="full")
    parser.add_argument("--score-view-mode", choices=("full", "tb_only", "ts_only", "bs_only"), default="full")
    parser.add_argument("--expert-subset", choices=("full", "rel_only", "rel_struct", "rel_safe"), default="full")
    parser.add_argument("--task-prompt-mode", choices=("none", "prompt", "retrieved"), default="none")
    parser.add_argument("--prompt-count", type=int, default=4)
    parser.add_argument("--retrieved-prompt-topm", type=int, default=4)
    parser.add_argument("--retriever-model", default="")
    parser.add_argument("--retriever-type", choices=("dual", "residual"), default="dual")
    parser.add_argument("--retriever-blend", type=float, default=0.0, help="Blend learned retriever score with raw embedding similarity.")
    parser.add_argument("--split-json", default="../../data/benchmark_train_test_split.json")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument(
        "--max-bundle-size",
        type=int,
        default=5,
        help="Maximum bundle length to expand to.",
    )
    parser.add_argument("--min-coverage-gain", type=float, default=0.0)
    parser.add_argument("--min-bundle-size", type=int, default=2)
    parser.add_argument("--final-candidate-count", type=int, default=10)
    parser.add_argument(
        "--final-selection-mode",
        choices=("score", "length_diverse"),
        default="length_diverse",
        help="Keep best candidates overall or reserve one candidate per length before filling by score.",
    )
    parser.add_argument(
        "--no-direct-fallback",
        dest="no_direct_fallback",
        action="store_true",
        default=True,
        help="Disable direct top-k fallback candidates (clean default).",
    )
    parser.add_argument(
        "--allow-direct-fallback",
        dest="no_direct_fallback",
        action="store_false",
        help="Enable direct top-k fallback candidates (non-clean ablation).",
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default="../output/pet_beam_search.json")
    args = parser.parse_args()
    if int(args.min_bundle_size) < 1:
        raise ValueError("--min-bundle-size must be at least 1")
    if int(args.max_bundle_size) < int(args.min_bundle_size):
        raise ValueError("--max-bundle-size must be at least --min-bundle-size")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tasks = load_json(Path(args.task_json))
    task_emb = np.load(Path(args.task_emb)).astype(np.float32, copy=False)
    skill_emb = np.load(Path(args.skill_emb)).astype(np.float32, copy=False)
    skill_emb = skill_emb / np.clip(np.linalg.norm(skill_emb, axis=1, keepdims=True), 1e-12, None)
    skill_meta = load_json(Path(args.skill_json))
    retriever = None
    skill_retriever_emb = None
    if args.retriever_model and Path(args.retriever_model).exists():
        if args.retriever_type == "residual":
            RetrieverClass = load_residual_retriever_class()
            retriever = RetrieverClass(dim=task_emb.shape[1], alpha=0.05).to(device)
        else:
            RetrieverClass = load_dual_tower_class()
            retriever = RetrieverClass(dim=task_emb.shape[1], out_dim=256).to(device)
        rckpt = torch.load(Path(args.retriever_model), map_location=device)
        retriever.load_state_dict(rckpt.get("model_state_dict", rckpt), strict=True)
        retriever.eval()
        if args.retriever_type == "dual":
            with torch.no_grad():
                skill_retriever_emb = retriever.encode_skill(torch.tensor(skill_emb, dtype=torch.float32, device=device)).detach().cpu().numpy()
        else:
            skill_retriever_emb = None

    train_graph = load_pet_graph_tensors(
        pet_split_dir=Path(args.train_split_dir),
        paper_data_dir=Path(args.paper_data),
        device=device,
        retrieved_prompt_topm=int(args.retrieved_prompt_topm),
    )
    model = TaskPETRelevance3Scorer(
        dim=train_graph.task_emb.shape[1],
        num_layers=2,
        structure_mode=str(args.structure_mode),
        view_mode=str(args.view_mode),
        score_view_mode=str(args.score_view_mode),
        expert_subset=str(args.expert_subset),
        task_prompt_mode=str(args.task_prompt_mode),
        prompt_count=int(args.prompt_count),
    ).to(device)
    model_path = Path(args.model_path)
    loaded_model = False
    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        current = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
        model.load_state_dict(compatible, strict=False)
        loaded_model = True
    model.eval()
    with torch.no_grad():
        train_enc = model.encode(train_graph)
    scorer = InferencePETScorer(
        model=model,
        train_graph=train_graph,
        train_enc=train_enc,
        full_skill_emb=skill_emb,
        skill_meta=skill_meta,
        device=device,
        retrieved_prompt_topm=int(args.retrieved_prompt_topm),
    )

    split_filter: set[int] | None = None
    if args.split != "all":
        split_data = load_json(Path(args.split_json))
        split_filter = set(int(x) for x in split_data.get(f"{args.split}_task_indices", []))

    out_tasks = []
    task_iter = [(i, t) for i, t in enumerate(tasks) if split_filter is None or i in split_filter]
    if args.max_tasks:
        task_iter = task_iter[: int(args.max_tasks)]
    for ti, task in task_iter:
        tv = norm_vec_np(task_emb[ti])
        if retriever is not None and float(args.retriever_blend) > 0.0:
            with torch.no_grad():
                if args.retriever_type == "dual":
                    assert skill_retriever_emb is not None
                    tz = retriever.encode_task(torch.tensor(tv, dtype=torch.float32, device=device).reshape(1, -1)).detach().cpu().numpy()[0]
                    retr_sims = skill_retriever_emb @ tz
                else:
                    # Residual retriever already includes raw cosine + bounded delta.
                    chunks = []
                    tt = torch.tensor(tv, dtype=torch.float32, device=device).reshape(1, -1)
                    for st in range(0, skill_emb.shape[0], 8192):
                        se = torch.tensor(skill_emb[st:st+8192], dtype=torch.float32, device=device)
                        chunks.append(retriever(tt.expand(se.shape[0], -1), se).detach().cpu().numpy())
                    retr_sims = np.concatenate(chunks, axis=0)
            raw_sims = skill_emb @ tv
            a = max(0.0, min(1.0, float(args.retriever_blend)))
            sims = (1.0 - a) * raw_sims + a * retr_sims
        else:
            sims = skill_emb @ tv
        topn = min(int(args.max_candidates), sims.shape[0])
        idx = np.argpartition(-sims, topn - 1)[:topn]
        idx = idx[np.argsort(-sims[idx])]
        candidates = [int(x) for x in idx.tolist()]
        min_bundle_size = int(args.min_bundle_size)
        max_bundle_size = int(args.max_bundle_size)
        search_max_bundle_size = min(max_bundle_size, len(candidates))

        score_cache: dict[tuple[int, ...], dict[str, Any]] = {}

        def score_cached(bundle_list: list[list[int]]) -> list[dict[str, Any]]:
            missing = []
            missing_keys = []
            for b in bundle_list:
                key = tuple(sorted(int(x) for x in b))
                if key not in score_cache:
                    missing.append(list(key))
                    missing_keys.append(key)
            if missing:
                vals = scorer.score_many(tv, missing, prompt_indices=candidates)
                for k, v in zip(missing_keys, vals):
                    score_cache[k] = v
            return [score_cache[tuple(sorted(int(x) for x in b))] for b in bundle_list]

        seed_bundles = [[int(sidx)] for sidx in candidates[: int(args.beam_size)]]
        seed_scores = score_cached(seed_bundles)
        beam = [(b, sc) for b, sc in zip(seed_bundles, seed_scores)]
        beam.sort(key=lambda x: x[1]["score"], reverse=True)
        final_pool = list(beam)

        for _round in range(2, search_max_bundle_size + 1):
            expanded_parent_coverage: dict[tuple[int, ...], float] = {}
            for bundle, _sc in beam:
                used = set(bundle)
                parent_cov = coverage_score(tv, skill_emb, bundle)
                for sidx in candidates:
                    if sidx in used:
                        continue
                    nb = tuple(sorted(bundle + [sidx]))
                    if nb in expanded_parent_coverage and expanded_parent_coverage[nb] >= parent_cov:
                        continue
                    expanded_parent_coverage[nb] = parent_cov
            expanded_keys = list(expanded_parent_coverage.keys())
            if not expanded_keys:
                break
            expanded_scores = score_cached([list(k) for k in expanded_keys])
            kept = []
            for k, v in zip(expanded_keys, expanded_scores):
                cov_gain = coverage_score(tv, skill_emb, list(k)) - float(expanded_parent_coverage[k])
                if cov_gain >= float(args.min_coverage_gain):
                    vv = dict(v)
                    vv["coverage_gain_from_parent"] = cov_gain
                    kept.append((list(k), vv))
            if not kept:
                break
            beam = kept
            beam.sort(key=lambda x: x[1]["score"], reverse=True)
            beam = beam[: int(args.beam_size)]
            final_pool.extend(beam)

        if not args.no_direct_fallback:
            fallback_bundles = []
            for k in (2, 3, 4):
                if min_bundle_size <= k <= search_max_bundle_size and len(candidates) >= k:
                    fallback_bundles.append(candidates[:k])
            fallback_scores = score_cached(fallback_bundles)
            for b, sc in zip(fallback_bundles, fallback_scores):
                vv = dict(sc)
                vv["candidate_source"] = f"direct_top{len(b)}"
                final_pool.append((list(b), vv))

        dedup = {}
        for bundle, sc in final_pool:
            key = tuple(bundle)
            if key not in dedup or sc["score"] > dedup[key]["score"]:
                dedup[key] = sc
        states = []
        for bundle, sc in dedup.items():
            if len(bundle) < min_bundle_size:
                continue
            states.append(
                {
                    "selected_indices_47153space": [int(x) for x in bundle],
                    "selected_names": [skill_name(skill_meta, int(x)) for x in bundle],
                    "selected_count": len(bundle),
                    "pet_score": float(sc["score"]),
                    "experts": sc.get("experts", []),
                    "gate": sc.get("gate", []),
                }
            )
        states.sort(key=lambda x: x["pet_score"], reverse=True)
        if args.final_selection_mode == "length_diverse":
            best_by_len = {}
            for st in states:
                l = int(st["selected_count"])
                if l not in best_by_len or float(st["pet_score"]) > float(best_by_len[l]["pet_score"]):
                    best_by_len[l] = st
            seed = sorted(best_by_len.values(), key=lambda x: x["pet_score"], reverse=True)
            used = {tuple(x["selected_indices_47153space"]) for x in seed}
            rest = [x for x in states if tuple(x["selected_indices_47153space"]) not in used]
            states = (seed + rest)[: int(args.final_candidate_count)]
        else:
            states = states[: int(args.final_candidate_count)]
        out_tasks.append(
            {
                "task_index": ti,
                "task_id": task.get("id"),
                "task_label": task.get("task_id"),
                "model_loaded": loaded_model,
                "retriever_loaded": retriever is not None,
                "retriever_type": args.retriever_type,
                "retriever_blend": float(args.retriever_blend),
                "candidates": candidates,
                "beam": {
                    "min_bundle_size": min_bundle_size,
                    "max_bundle_size": max_bundle_size,
                    "search_max_bundle_size": search_max_bundle_size,
                    "final_top_states": states,
                    "selected_indices_47153space": states[0]["selected_indices_47153space"] if states else [],
                    "final_pet_score": states[0]["pet_score"] if states else 0.0,
                },
            }
        )

    config = vars(args).copy()
    config["max_bundle_size_policy"] = "hard_cap"
    out = {
        "config": config,
        "model_loaded": loaded_model,
        "tasks": out_tasks,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path} tasks={len(out_tasks)} model_loaded={loaded_model}")


if __name__ == "__main__":
    main()
