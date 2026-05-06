#!/usr/bin/env python3
"""Typed LightGCN PET-style scorer.

This module defines the model only. Training/search scripts can import:
- TypedLightGCNEncoder
- TaskPETScorer
- load_pet_graph_tensors
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_SKILL_EDGE_TYPES = [
    "task_gold_skill",
    "task_top20_skill",
]

BUNDLE_SKILL_EDGE_TYPES = [
    "bundle_member",
]

TASK_BUNDLE_EDGE_TYPES = [
    "gold_bundle",
]


@dataclass
class PETGraphTensors:
    task_emb: torch.Tensor
    skill_emb: torch.Tensor
    bundle_skill_indices: list[list[int]]
    bundle_global_skill_indices: list[list[int]]
    bundle_skill_padded: torch.Tensor
    bundle_skill_mask: torch.Tensor
    bundle_member_edge: torch.Tensor
    task_bundle_edges: dict[str, torch.Tensor]
    task_skill_edges: dict[str, torch.Tensor]
    task_prompt_edges: torch.Tensor
    bundle_skill_edges: dict[str, torch.Tensor]
    bundle_task_index: torch.Tensor
    bundle_role_is_positive: torch.Tensor
    bundle_features: torch.Tensor
    active_skill_global_indices: torch.Tensor


def l2norm_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, dim=1, keepdim=True), min=eps)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _edge_index_from_rows(
    rows: list[dict[str, Any]],
    *,
    left_key: str,
    right_key: str,
    edge_type_key: str,
    edge_types: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, list[list[int]]] = {t: [[], []] for t in edge_types}
    for row in rows:
        et = str(row[edge_type_key])
        if et not in out:
            continue
        out[et][0].append(int(row[left_key]))
        out[et][1].append(int(row[right_key]))
    return {
        et: torch.tensor(v, dtype=torch.long, device=device)
        if v[0]
        else torch.empty((2, 0), dtype=torch.long, device=device)
        for et, v in out.items()
    }


def load_pet_graph_tensors(
    *,
    pet_split_dir: str | Path,
    paper_data_dir: str | Path,
    device: str | torch.device = "cuda",
    retrieved_prompt_topm: int = 4,
) -> PETGraphTensors:
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    pet_split_dir = Path(pet_split_dir)
    paper_data_dir = Path(paper_data_dir)

    task_nodes = json.loads((pet_split_dir / "task_nodes.json").read_text(encoding="utf-8"))
    bundle_nodes = json.loads((pet_split_dir / "bundle_nodes.json").read_text(encoding="utf-8"))
    task_bundle_rows = _read_jsonl(pet_split_dir / "task_bundle_interactions.jsonl")
    task_skill_rows = _read_jsonl(pet_split_dir / "task_skill_interactions.jsonl")
    bundle_skill_rows = _read_jsonl(pet_split_dir / "bundle_skill_interactions.jsonl")

    full_task_emb = np.load(paper_data_dir / "task_embeddings_bigmodel_512_f32_v2.npy").astype(np.float32, copy=False)
    full_skill_emb = np.load(paper_data_dir / "benchmark_merged_skill_embeddings.npy").astype(np.float32, copy=False)

    task_indices = [int(x["task_index"]) for x in task_nodes]
    task_emb = torch.tensor(full_task_emb[np.asarray(task_indices, dtype=np.int64)], dtype=torch.float32, device=device)
    task_emb = l2norm_rows(task_emb)

    global_task_to_local = {g: i for i, g in enumerate(task_indices)}
    global_bundle_to_local = {int(x["bundle_id"]): i for i, x in enumerate(bundle_nodes)}

    # Training uses the active skill-induced subgraph only. Full-universe
    # fallback for inference can be handled by search code with raw embeddings.
    active_skill_ids: set[int] = set()
    for row in task_skill_rows:
        gti = int(row["task_index"])
        if gti in global_task_to_local and str(row["edge_type"]) in {"task_gold_skill", "task_top20_skill"}:
            active_skill_ids.add(int(row["skill_index"]))
    for row in bundle_skill_rows:
        gbid = int(row["bundle_id"])
        if gbid in global_bundle_to_local:
            active_skill_ids.add(int(row["skill_index"]))
    active_skill_global = sorted(active_skill_ids)
    global_skill_to_local = {g: i for i, g in enumerate(active_skill_global)}
    skill_emb = torch.tensor(
        full_skill_emb[np.asarray(active_skill_global, dtype=np.int64)],
        dtype=torch.float32,
        device=device,
    )
    skill_emb = l2norm_rows(skill_emb)

    # Reindex task/bundle ids in split rows to local graph ids.
    # Message passing uses only stable positive/candidate structure:
    # - task-bundle: gold only
    # - task-skill: gold + top20
    # Negative labels are reserved for scorer/loss, not propagation.
    tb_reindexed = []
    for row in task_bundle_rows:
        val = float(row["interaction"])
        if val != 1.0:
            continue
        tb_reindexed.append(
            {
                "task": global_task_to_local[int(row["task_index"])],
                "bundle": global_bundle_to_local[int(row["bundle_id"])],
                "edge_type": "gold_bundle",
            }
        )

    ts_reindexed = []
    prompt_reindexed = []
    for row in task_skill_rows:
        gti = int(row["task_index"])
        if gti not in global_task_to_local:
            continue
        edge_type = str(row["edge_type"])
        if edge_type not in {"task_gold_skill", "task_top20_skill"}:
            continue
        item = {
            "task": global_task_to_local[gti],
            "skill": global_skill_to_local[int(row["skill_index"])],
            "edge_type": edge_type,
        }
        ts_reindexed.append(item)
        if edge_type == "task_top20_skill":
            try:
                rank = int(row.get("retrieval_rank", 10**9))
            except Exception:
                rank = 10**9
            if rank <= int(retrieved_prompt_topm):
                prompt_reindexed.append(item)

    bs_reindexed = []
    for row in bundle_skill_rows:
        gbid = int(row["bundle_id"])
        if gbid not in global_bundle_to_local:
            continue
        bs_reindexed.append(
            {
                "bundle": global_bundle_to_local[gbid],
                "skill": global_skill_to_local[int(row["skill_index"])],
                "edge_type": "bundle_member",
            }
        )

    bundle_skill_indices: list[list[int]] = []
    bundle_global_skill_indices: list[list[int]] = []
    bundle_task_index: list[int] = []
    bundle_role_is_positive: list[int] = []
    bundle_features: list[list[float]] = []
    for b in bundle_nodes:
        global_indices = [int(x) for x in b["skill_indices"]]
        bundle_global_skill_indices.append(global_indices)
        bundle_skill_indices.append([global_skill_to_local[int(x)] for x in global_indices])
        bundle_task_index.append(global_task_to_local[int(b["task_index"])])
        bundle_role_is_positive.append(1 if str(b["role"]) == "positive" else 0)
        skill_indices = global_indices
        risk_vals = []
        cost_vals = []
        # Risk/cost features are read from the full skill metadata.
        # Missing values are treated as zero-risk/zero-cost.
        skill_meta_path = paper_data_dir / "benchmark_merged_skills.json"
        # Placeholder; actual metadata is loaded once below.
        bundle_features.append([float(len(skill_indices)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    skill_meta = json.loads((paper_data_dir / "benchmark_merged_skills.json").read_text(encoding="utf-8"))
    for bi, skill_indices in enumerate(bundle_skill_indices):
        risks = []
        costs = []
        for sidx in skill_indices:
            meta = skill_meta[int(sidx)] if 0 <= int(sidx) < len(skill_meta) else {}
            try:
                risks.append(float(meta.get("permission_risk_value", 0.0)))
            except Exception:
                risks.append(0.0)
            try:
                costs.append(float(meta.get("token_cost_value", 0.0)))
            except Exception:
                costs.append(0.0)
        size = float(len(skill_indices))
        risk_mean = float(sum(risks) / max(len(risks), 1))
        risk_max = float(max(risks) if risks else 0.0)
        cost_mean = float(sum(costs) / max(len(costs), 1))
        cost_max = float(max(costs) if costs else 0.0)
        if len(skill_indices) > 1:
            vecs = full_skill_emb[np.asarray(skill_indices, dtype=np.int64)]
            vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
            sim = vecs @ vecs.T
            tri = sim[np.triu_indices(sim.shape[0], k=1)]
            pair_mean = float(np.mean(tri))
            pair_max = float(np.max(tri))
            pair_min = float(np.min(tri))
        else:
            pair_mean = 0.0
            pair_max = 0.0
            pair_min = 0.0
        bundle_features[bi] = [size / 10.0, risk_mean, risk_max, cost_mean, cost_max, pair_mean, pair_max, pair_min]

    member_left: list[int] = []
    member_right: list[int] = []
    for local_bid, skill_indices in enumerate(bundle_skill_indices):
        for sidx in skill_indices:
            member_left.append(local_bid)
            member_right.append(int(sidx))
    bundle_member_edge = (
        torch.tensor([member_left, member_right], dtype=torch.long, device=device)
        if member_left
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    task_prompt_edge = (
        torch.tensor(
            [[int(x["task"]) for x in prompt_reindexed], [int(x["skill"]) for x in prompt_reindexed]],
            dtype=torch.long,
            device=device,
        )
        if prompt_reindexed
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    max_bundle_len = max((len(x) for x in bundle_skill_indices), default=1)
    bundle_skill_padded = torch.zeros((len(bundle_skill_indices), max_bundle_len), dtype=torch.long, device=device)
    bundle_skill_mask = torch.zeros((len(bundle_skill_indices), max_bundle_len), dtype=torch.float32, device=device)
    for bi, skill_indices in enumerate(bundle_skill_indices):
        if not skill_indices:
            continue
        ids = torch.tensor(skill_indices, dtype=torch.long, device=device)
        bundle_skill_padded[bi, : ids.numel()] = ids
        bundle_skill_mask[bi, : ids.numel()] = 1.0

    return PETGraphTensors(
        task_emb=task_emb,
        skill_emb=skill_emb,
        bundle_skill_indices=bundle_skill_indices,
        bundle_global_skill_indices=bundle_global_skill_indices,
        bundle_skill_padded=bundle_skill_padded,
        bundle_skill_mask=bundle_skill_mask,
        bundle_member_edge=bundle_member_edge,
        task_bundle_edges=_edge_index_from_rows(
            tb_reindexed,
            left_key="task",
            right_key="bundle",
            edge_type_key="edge_type",
            edge_types=TASK_BUNDLE_EDGE_TYPES,
            device=device,
        ),
        task_skill_edges=_edge_index_from_rows(
            ts_reindexed,
            left_key="task",
            right_key="skill",
            edge_type_key="edge_type",
            edge_types=TASK_SKILL_EDGE_TYPES,
            device=device,
        ),
        task_prompt_edges=task_prompt_edge,
        bundle_skill_edges=_edge_index_from_rows(
            bs_reindexed,
            left_key="bundle",
            right_key="skill",
            edge_type_key="edge_type",
            edge_types=BUNDLE_SKILL_EDGE_TYPES,
            device=device,
        ),
        bundle_task_index=torch.tensor(bundle_task_index, dtype=torch.long, device=device),
        bundle_role_is_positive=torch.tensor(bundle_role_is_positive, dtype=torch.float32, device=device),
        bundle_features=torch.tensor(bundle_features, dtype=torch.float32, device=device),
        active_skill_global_indices=torch.tensor(active_skill_global, dtype=torch.long, device=device),
    )


class TypedBipartiteLightGCN(nn.Module):
    """One typed bipartite LightGCN encoder with scalar edge-type gates."""

    def __init__(self, edge_types: list[str], num_layers: int = 2) -> None:
        super().__init__()
        self.edge_types = list(edge_types)
        self.num_layers = int(num_layers)
        self.edge_logits = nn.Parameter(torch.zeros(len(self.edge_types), dtype=torch.float32))

    def propagate(
        self,
        left_x: torch.Tensor,
        right_x: torch.Tensor,
        edges_by_type: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_layers = [left_x]
        right_layers = [right_x]
        cur_left = left_x
        cur_right = right_x
        edge_weights = torch.sigmoid(self.edge_logits)

        for _ in range(self.num_layers):
            next_left = torch.zeros_like(cur_left)
            next_right = torch.zeros_like(cur_right)
            left_deg = torch.zeros(cur_left.shape[0], dtype=cur_left.dtype, device=cur_left.device)
            right_deg = torch.zeros(cur_right.shape[0], dtype=cur_right.dtype, device=cur_right.device)

            for k, et in enumerate(self.edge_types):
                edge = edges_by_type.get(et)
                if edge is None or edge.numel() == 0:
                    continue
                lidx = edge[0]
                ridx = edge[1]
                w = edge_weights[k]
                next_left.index_add_(0, lidx, w * cur_right[ridx])
                next_right.index_add_(0, ridx, w * cur_left[lidx])
                left_deg.index_add_(0, lidx, torch.ones_like(lidx, dtype=cur_left.dtype) * w)
                right_deg.index_add_(0, ridx, torch.ones_like(ridx, dtype=cur_right.dtype) * w)

            next_left = next_left / torch.clamp(left_deg[:, None], min=1.0)
            next_right = next_right / torch.clamp(right_deg[:, None], min=1.0)
            # Isolated nodes keep their previous representation.
            next_left = torch.where(left_deg[:, None] > 0, next_left, cur_left)
            next_right = torch.where(right_deg[:, None] > 0, next_right, cur_right)
            next_left = l2norm_rows(next_left)
            next_right = l2norm_rows(next_right)
            left_layers.append(next_left)
            right_layers.append(next_right)
            cur_left, cur_right = next_left, next_right

        return l2norm_rows(torch.stack(left_layers, dim=0).mean(dim=0)), l2norm_rows(torch.stack(right_layers, dim=0).mean(dim=0))


class TaskPETScorer(nn.Module):
    """Typed LightGCN + task-conditioned three-view scorer."""

    def __init__(
        self,
        dim: int = 512,
        num_layers: int = 2,
        gate_temperature: float = 1.0,
        structure_mode: str = "soft",
        view_mode: str = "full",
        relevance_expert_mode: str = "linear",
        expert_mode: str = "legacy",
        task_prompt_mode: str = "none",
        prompt_count: int = 4,
    ) -> None:
        super().__init__()
        self.gate_temperature = float(gate_temperature)
        if structure_mode not in {"soft", "learned_red"}:
            raise ValueError(f"Unknown structure_mode: {structure_mode}")
        if view_mode not in {"full", "task_skill_only"}:
            raise ValueError(f"Unknown view_mode: {view_mode}")
        if relevance_expert_mode not in {"linear", "mlp"}:
            raise ValueError(f"Unknown relevance_expert_mode: {relevance_expert_mode}")
        if expert_mode not in {"legacy", "role_specific", "role_smooth"}:
            raise ValueError(f"Unknown expert_mode: {expert_mode}")
        if task_prompt_mode not in {"none", "prompt", "retrieved"}:
            raise ValueError(f"Unknown task_prompt_mode: {task_prompt_mode}")
        self.structure_mode = structure_mode
        self.view_mode = view_mode
        self.relevance_expert_mode = relevance_expert_mode
        self.expert_mode = expert_mode
        self.task_prompt_mode = task_prompt_mode
        self.tb_encoder = TypedBipartiteLightGCN(TASK_BUNDLE_EDGE_TYPES, num_layers=num_layers)
        self.ts_encoder = TypedBipartiteLightGCN(TASK_SKILL_EDGE_TYPES, num_layers=num_layers)
        self.bs_encoder = TypedBipartiteLightGCN(BUNDLE_SKILL_EDGE_TYPES, num_layers=num_layers)
        # Gate is task-conditioned and bundle-aware. It selects among three
        # interpretable experts: relevance, structure/minimality, safety-cost.
        self.gate = nn.Sequential(
            nn.Linear(dim * 4 + 8, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
        )
        self.tb_mlp = nn.Sequential(nn.Linear(dim * 4, 64), nn.Tanh(), nn.Linear(64, 1))
        self.bs_mlp = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
        # Interpretable expert parameters. Each expert is a small learnable
        # formula rather than a black-box MLP.
        self.rel_logits = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.rel_role_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.rel_mlp = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
        self.struct_pos_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.struct_role_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.struct_smooth_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        # Keep structure penalties light at initialization. The previous
        # zero initialization maps to softplus(0)=0.69, which can immediately
        # clamp the structure expert to zero on high-similarity bundles.
        self.struct_neg_raw = nn.Parameter(torch.full((2,), -2.0, dtype=torch.float32))
        # Start safety-cost penalties conservatively; they are learnable.
        self.safety_raw = nn.Parameter(torch.full((4,), -2.0, dtype=torch.float32))
        self.risk_agg_logits = nn.Parameter(torch.tensor([0.6, 0.4], dtype=torch.float32))
        self.cost_agg_logits = nn.Parameter(torch.tensor([0.7, 0.3], dtype=torch.float32))
        self.gain_threshold_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        # Optional clean structure variant: learn the high-redundancy threshold
        # inside a bounded [0.75, 0.95] interval, initialized at 0.85.
        self.redundancy_threshold_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.redundancy_mix_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.task_prompt_tokens = nn.Parameter(torch.randn(int(prompt_count), dim, dtype=torch.float32) * 0.02)
        self.task_prompt_mlp = nn.Sequential(nn.Linear(dim * 4, 64), nn.Tanh(), nn.Linear(64, dim))
        self.task_prompt_norm = nn.LayerNorm(dim)
        self.task_prompt_scale_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        # DAM-inspired task-conditioned skill attention for bundle encoding.
        self.attn_task_proj = nn.Linear(dim, 64, bias=False)
        self.attn_skill_proj = nn.Linear(dim, 64, bias=False)
        self.attn_out = nn.Linear(64, 1, bias=False)

    @staticmethod
    def retrieved_skill_context(
        task_x: torch.Tensor,
        skill_x: torch.Tensor,
        prompt_edges: torch.Tensor,
    ) -> torch.Tensor:
        ctx = torch.zeros_like(task_x)
        deg = torch.zeros(task_x.shape[0], dtype=task_x.dtype, device=task_x.device)
        if prompt_edges is not None and prompt_edges.numel() > 0:
            tidx = prompt_edges[0]
            sidx = prompt_edges[1]
            ctx.index_add_(0, tidx, skill_x[sidx])
            deg.index_add_(0, tidx, torch.ones_like(tidx, dtype=task_x.dtype))
        ctx = ctx / torch.clamp(deg[:, None], min=1.0)
        return l2norm_rows(ctx)

    def adapt_task_embeddings(self, task_x: torch.Tensor, skill_context: torch.Tensor | None = None) -> torch.Tensor:
        task_x = l2norm_rows(task_x)
        if self.task_prompt_mode == "none":
            return task_x
        if self.task_prompt_mode == "retrieved":
            if skill_context is None:
                return task_x
            scale = torch.sigmoid(self.task_prompt_scale_raw)
            return l2norm_rows(task_x + scale * l2norm_rows(skill_context))
        prompt = l2norm_rows(self.task_prompt_tokens)
        attn = torch.softmax(task_x @ prompt.T, dim=1)
        prompt_ctx = attn @ prompt
        prompt_in = torch.cat([task_x, prompt_ctx, torch.abs(task_x - prompt_ctx), task_x * prompt_ctx], dim=1)
        delta = self.task_prompt_mlp(prompt_in)
        scale = torch.sigmoid(self.task_prompt_scale_raw)
        return l2norm_rows(self.task_prompt_norm(task_x + scale * delta))

    def relevance_expert_score(self, rel_feat: torch.Tensor) -> torch.Tensor:
        if self.relevance_expert_mode == "mlp":
            return torch.sigmoid(self.rel_mlp(rel_feat).squeeze(-1))
        rel_w = F.softmax(self.rel_logits, dim=0)
        return torch.clamp(torch.sum(rel_feat * rel_w.reshape(1, -1), dim=1), 0.0, 1.0)

    def role_specific_expert_scores(
        self,
        *,
        score_tb: torch.Tensor,
        score_ts: torch.Tensor,
        score_bs: torch.Tensor,
        pair_mean: torch.Tensor,
        novelty: torch.Tensor,
        redundancy_proxy: torch.Tensor,
        redundancy_excess: torch.Tensor,
        attention_coverage: torch.Tensor,
        risk_agg: torch.Tensor,
        cost_agg: torch.Tensor,
        risk_max: torch.Tensor,
        cost_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rel_consistency = torch.clamp(1.0 - torch.abs(score_tb - score_ts), 0.0, 1.0)
        rel_feat = torch.stack([score_tb, score_ts, rel_consistency], dim=1)
        rel_w = F.softmax(self.rel_role_logits, dim=0)
        score_rel = torch.clamp(torch.sum(rel_feat * rel_w.reshape(1, -1), dim=1), 0.0, 1.0)

        if self.expert_mode == "role_smooth":
            novelty_smooth = torch.sqrt(torch.clamp(1.0 - pair_mean, min=0.0, max=1.0))
            smooth_redundancy_proxy = torch.clamp(0.5 * redundancy_proxy + 0.5 * pair_mean, 0.0, 1.0)
            redundancy_threshold = 0.75 + 0.20 * torch.sigmoid(self.redundancy_threshold_raw)
            redundancy_excess = torch.clamp(
                (smooth_redundancy_proxy - redundancy_threshold) / torch.clamp(1.0 - redundancy_threshold, min=1e-6),
                0.0,
                1.0,
            )
            struct_feat = torch.stack([score_bs, novelty_smooth, attention_coverage], dim=1)
            struct_w = F.softmax(self.struct_smooth_logits, dim=0)
        else:
            struct_feat = torch.stack([score_bs, novelty], dim=1)
            struct_w = F.softmax(self.struct_role_logits, dim=0)
        struct_reward = torch.clamp(torch.sum(struct_feat * struct_w.reshape(1, -1), dim=1), 0.0, 1.0)
        struct_neg_w = F.softplus(self.struct_neg_raw)
        structure_penalty = struct_neg_w[0] * redundancy_excess
        score_struct = torch.clamp(struct_reward * torch.exp(-structure_penalty), 0.0, 1.0)

        safety_w = F.softplus(self.safety_raw)
        safety_penalty = (
            safety_w[0] * risk_agg
            + safety_w[1] * cost_agg
            + safety_w[2] * risk_max
            + safety_w[3] * cost_max
        )
        score_safe = torch.clamp(1.0 - torch.sigmoid(safety_penalty), 0.0, 1.0)
        return score_rel, score_struct, score_safe, structure_penalty

    def encode(self, graph: PETGraphTensors) -> dict[str, torch.Tensor]:
        # Initial bundle embeddings are mean-pooled skill embeddings, computed
        # with one vectorized scatter instead of a Python loop over bundles.
        edge = graph.bundle_member_edge
        num_bundles = len(graph.bundle_skill_indices)
        bundle_x = torch.zeros(
            (num_bundles, graph.skill_emb.shape[1]),
            dtype=graph.skill_emb.dtype,
            device=graph.skill_emb.device,
        )
        deg = torch.zeros(num_bundles, dtype=graph.skill_emb.dtype, device=graph.skill_emb.device)
        if edge.numel() > 0:
            bidx = edge[0]
            sidx = edge[1]
            bundle_x.index_add_(0, bidx, graph.skill_emb[sidx])
            deg.index_add_(0, bidx, torch.ones_like(bidx, dtype=graph.skill_emb.dtype))
        bundle_x = bundle_x / torch.clamp(deg[:, None], min=1.0)
        bundle_x = l2norm_rows(bundle_x)

        skill_context = None
        if self.task_prompt_mode == "retrieved":
            skill_context = self.retrieved_skill_context(graph.task_emb, graph.skill_emb, graph.task_prompt_edges)
        task_x = self.adapt_task_embeddings(graph.task_emb, skill_context)

        if self.view_mode == "task_skill_only":
            task_ts, skill_ts = self.ts_encoder.propagate(task_x, graph.skill_emb, graph.task_skill_edges)
            edge = graph.bundle_member_edge
            num_bundles = len(graph.bundle_skill_indices)
            ts_bundle = torch.zeros(
                (num_bundles, skill_ts.shape[1]),
                dtype=skill_ts.dtype,
                device=skill_ts.device,
            )
            deg = torch.zeros(num_bundles, dtype=skill_ts.dtype, device=skill_ts.device)
            if edge.numel() > 0:
                bidx = edge[0]
                sidx = edge[1]
                ts_bundle.index_add_(0, bidx, skill_ts[sidx])
                deg.index_add_(0, bidx, torch.ones_like(bidx, dtype=skill_ts.dtype))
            ts_bundle = l2norm_rows(ts_bundle / torch.clamp(deg[:, None], min=1.0))
            align_sum = torch.zeros(num_bundles, dtype=skill_ts.dtype, device=skill_ts.device)
            if edge.numel() > 0:
                bidx = edge[0]
                sidx = edge[1]
                align_sum.index_add_(0, bidx, torch.sum(skill_ts[sidx] * ts_bundle[bidx], dim=1))
            member_align_mean = torch.sigmoid(align_sum / torch.clamp(deg, min=1.0))
            return {
                "task_tb": task_ts,
                "bundle_tb": ts_bundle,
                "task_ts": task_ts,
                "skill_ts": skill_ts,
                "bundle_bs": ts_bundle,
                "skill_bs": skill_ts,
                "bundle_init": bundle_x,
                "bundle_ts_mean": ts_bundle,
                "bundle_bs_member_align_mean": member_align_mean,
                "task_input": task_x,
            }

        task_tb, bundle_tb = self.tb_encoder.propagate(task_x, bundle_x, graph.task_bundle_edges)
        task_ts, skill_ts = self.ts_encoder.propagate(task_x, graph.skill_emb, graph.task_skill_edges)
        bundle_bs, skill_bs = self.bs_encoder.propagate(bundle_x, graph.skill_emb, graph.bundle_skill_edges)
        # Precompute bundle-level means for vectorized scoring.
        edge = graph.bundle_member_edge
        num_bundles = len(graph.bundle_skill_indices)
        ts_bundle = torch.zeros(
            (num_bundles, skill_ts.shape[1]),
            dtype=skill_ts.dtype,
            device=skill_ts.device,
        )
        bs_align_sum = torch.zeros(num_bundles, dtype=skill_ts.dtype, device=skill_ts.device)
        deg = torch.zeros(num_bundles, dtype=skill_ts.dtype, device=skill_ts.device)
        if edge.numel() > 0:
            bidx = edge[0]
            sidx = edge[1]
            ts_bundle.index_add_(0, bidx, skill_ts[sidx])
            bs_align_sum.index_add_(0, bidx, torch.sum(skill_bs[sidx] * bundle_bs[bidx], dim=1))
            deg.index_add_(0, bidx, torch.ones_like(bidx, dtype=skill_ts.dtype))
        ts_bundle = l2norm_rows(ts_bundle / torch.clamp(deg[:, None], min=1.0))
        bs_member_align_mean = torch.sigmoid(bs_align_sum / torch.clamp(deg, min=1.0))
        return {
            "task_tb": task_tb,
            "bundle_tb": bundle_tb,
            "task_ts": task_ts,
            "skill_ts": skill_ts,
            "bundle_bs": bundle_bs,
            "skill_bs": skill_bs,
            "bundle_init": bundle_x,
            "bundle_ts_mean": ts_bundle,
            "bundle_bs_member_align_mean": bs_member_align_mean,
            "task_input": task_x,
        }

    def score_bundle_ids(
        self,
        graph: PETGraphTensors,
        enc: dict[str, torch.Tensor],
        task_local_idx: torch.Tensor,
        bundle_local_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        t_tb = enc["task_tb"][task_local_idx]
        # Task-conditioned attentive bundle embeddings for all scored pairs.
        # This used to loop over every pair and launch many tiny kernels; the
        # padded bundle member table keeps the operation batched on device.
        member_ids = graph.bundle_skill_padded[bundle_local_idx]
        member_mask = graph.bundle_skill_mask[bundle_local_idx]
        denom = torch.clamp(member_mask.sum(dim=1), min=1.0)
        s_tb = enc["skill_bs"][member_ids]
        q = self.attn_task_proj(t_tb).unsqueeze(1)
        k = self.attn_skill_proj(s_tb)
        logits = self.attn_out(torch.tanh(q + k)).squeeze(-1)
        logits = logits.masked_fill(member_mask <= 0, -1e9)
        w = torch.softmax(logits, dim=1) * member_mask
        w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1e-12)
        attn_entropy = -torch.sum(w * torch.log(torch.clamp(w, min=1e-8)), dim=1)
        attention_coverage = torch.clamp(torch.exp(attn_entropy) / 5.0, 0.0, 1.0)
        b_tb = l2norm_rows(torch.sum(w[:, :, None] * s_tb, dim=1))
        raw_s = graph.skill_emb[member_ids]
        b0_attn = l2norm_rows(torch.sum(w[:, :, None] * raw_s, dim=1))
        # For TB main score, use the bundle representation from the TB view
        # directly (no attentive skill aggregation for this head).
        b_tb_direct = enc["bundle_tb"][bundle_local_idx]
        score_tb = torch.sigmoid(
            self.tb_mlp(torch.cat([t_tb, b_tb_direct, torch.abs(t_tb - b_tb_direct), t_tb * b_tb_direct], dim=1)).squeeze(-1)
        )

        # Task-skill view uses task_ts against member skill_ts directly
        # instead of first collapsing members into a bundle_ts vector.
        t_ts = enc["task_ts"][task_local_idx]
        s_ts = enc["skill_ts"][member_ids]
        member_dot = torch.sum(s_ts * t_ts[:, None, :], dim=2)
        score_ts_t = torch.sigmoid(torch.sum(member_dot * member_mask, dim=1) / denom)
        member_absdiff = torch.mean(torch.abs(s_ts - t_ts[:, None, :]), dim=2)
        sim_absdiff = torch.clamp(1.0 - (torch.sum(member_absdiff * member_mask, dim=1) / denom) / 2.0, 0.0, 1.0)
        member_prod = torch.mean(s_ts * t_ts[:, None, :], dim=2)
        sim_prod = torch.clamp((torch.sum(member_prod * member_mask, dim=1) / denom + 1.0) / 2.0, 0.0, 1.0)
        member_l2 = torch.norm(s_ts - t_ts[:, None, :], dim=2)
        sim_l2 = torch.clamp(1.0 - (torch.sum(member_l2 * member_mask, dim=1) / denom) / 2.0, 0.0, 1.0)
        bs_feat = torch.stack(
            [
                enc["bundle_bs_member_align_mean"][bundle_local_idx],
                graph.bundle_features[bundle_local_idx, 5],
                graph.bundle_features[bundle_local_idx, 6],
                graph.bundle_features[bundle_local_idx, 7],
            ],
            dim=1,
        )
        score_bs_t = torch.sigmoid(self.bs_mlp(bs_feat).squeeze(-1))
        risk_mean = graph.bundle_features[bundle_local_idx, 1]
        risk_max = graph.bundle_features[bundle_local_idx, 2]
        cost_mean = graph.bundle_features[bundle_local_idx, 3]
        cost_max = graph.bundle_features[bundle_local_idx, 4]
        risk_w = F.softmax(self.risk_agg_logits, dim=0)
        cost_w = F.softmax(self.cost_agg_logits, dim=0)
        risk_agg = risk_w[0] * risk_mean + risk_w[1] * risk_max
        cost_agg = cost_w[0] * cost_mean + cost_w[1] * cost_max
        pair_mean = graph.bundle_features[bundle_local_idx, 5]
        pair_max = graph.bundle_features[bundle_local_idx, 6]
        pair_min = graph.bundle_features[bundle_local_idx, 7]
        redundancy_mix = torch.sigmoid(self.redundancy_mix_raw)
        redundancy_proxy = torch.clamp(pair_max, 0.0, 1.0)
        novelty_proxy = torch.clamp(1.0 - pair_mean, 0.0, 1.0)
        gain_thr = 0.05 + 0.90 * torch.sigmoid(self.gain_threshold_raw)
        low_gain_proxy = torch.clamp((gain_thr - score_ts_t) / torch.clamp(gain_thr, min=1e-6), 0.0, 1.0)
        size_norm = graph.bundle_features[bundle_local_idx, 0]
        gain_size_penalty = low_gain_proxy * torch.clamp(size_norm, 0.0, 1.0)
        t0 = enc["task_input"][task_local_idx]
        b0 = b0_attn
        gate_input = torch.cat(
            [
                t0,
                b0,
                torch.abs(t0 - b0),
                t0 * b0,
                graph.bundle_features[bundle_local_idx],
            ],
            dim=1,
        )
        gate = F.softmax(self.gate(gate_input) / max(self.gate_temperature, 1e-6), dim=1)
        struct_neg_w = F.softplus(self.struct_neg_raw)
        coverage_mean = score_ts_t
        coverage_floor = score_ts_t
        unique_coverage = score_bs_t
        redundant_low_gain_proxy = torch.zeros_like(score_ts_t)
        if self.structure_mode == "learned_red":
            redundancy_threshold = 0.75 + 0.20 * torch.sigmoid(self.redundancy_threshold_raw)
            redundancy_excess_max = torch.clamp(
                (torch.clamp(pair_max, 0.0, 1.0) - redundancy_threshold)
                / torch.clamp(1.0 - redundancy_threshold, min=1e-6),
                0.0,
                1.0,
            )
            redundancy_excess_mean = torch.clamp(
                (torch.clamp(pair_mean, 0.0, 1.0) - redundancy_threshold)
                / torch.clamp(1.0 - redundancy_threshold, min=1e-6),
                0.0,
                1.0,
            )
        else:
            redundancy_threshold = torch.tensor(0.85, dtype=score_ts_t.dtype, device=score_ts_t.device)
            redundancy_excess_max = torch.clamp((torch.clamp(pair_max, 0.0, 1.0) - 0.85) / 0.15, 0.0, 1.0)
            redundancy_excess_mean = torch.clamp((torch.clamp(pair_mean, 0.0, 1.0) - 0.85) / 0.15, 0.0, 1.0)
        redundancy_excess = torch.clamp(
            redundancy_mix * redundancy_excess_max + (1.0 - redundancy_mix) * redundancy_excess_mean,
            0.0,
            1.0,
        )

        if self.expert_mode in {"role_specific", "role_smooth"}:
            score_rel_t, score_struct_t, score_safe_t, structure_penalty = self.role_specific_expert_scores(
                score_tb=score_tb,
                score_ts=score_ts_t,
                score_bs=score_bs_t,
                pair_mean=pair_mean,
                novelty=novelty_proxy,
                redundancy_proxy=redundancy_proxy,
                redundancy_excess=redundancy_excess,
                attention_coverage=attention_coverage,
                risk_agg=risk_agg,
                cost_agg=cost_agg,
                risk_max=risk_max,
                cost_max=cost_max,
            )
        else:
            rel_feat = torch.stack([score_tb, sim_absdiff, sim_prod, sim_l2], dim=1)
            score_rel_t = self.relevance_expert_score(rel_feat)

            struct_pos = torch.stack([score_bs_t, novelty_proxy], dim=1)
            struct_pos_w = F.softmax(self.struct_pos_logits, dim=0)
            struct_reward = torch.clamp(torch.sum(struct_pos * struct_pos_w.reshape(1, -1), dim=1), 0.0, 1.0)
            if self.structure_mode == "learned_red":
                structure_penalty = struct_neg_w[0] * redundancy_excess + struct_neg_w[1] * gain_size_penalty
            else:
                structure_penalty = struct_neg_w[0] * redundancy_excess + struct_neg_w[1] * gain_size_penalty
            score_struct_t = torch.clamp(struct_reward * torch.exp(-structure_penalty), 0.0, 1.0)

            safety_w = F.softplus(self.safety_raw)
            safety_penalty = (
                safety_w[0] * risk_agg
                + safety_w[1] * cost_agg
                + safety_w[2] * risk_max
                + safety_w[3] * cost_max
            )
            score_safe_t = torch.clamp(1.0 - torch.sigmoid(safety_penalty), 0.0, 1.0)
        expert_scores = torch.stack([score_rel_t, score_struct_t, score_safe_t], dim=1)
        final = torch.clamp(torch.sum(gate * expert_scores, dim=1), 0.0, 1.0)
        return final, {
            "score_tb": score_tb,
            "score_tb_bundle_direct": score_tb,
            "score_ts": score_ts_t,
            "score_bs": score_bs_t,
            "score_relevance": score_rel_t,
            "score_structure": score_struct_t,
            "score_safety_cost": score_safe_t,
            "novelty_proxy": novelty_proxy,
            "attention_coverage": attention_coverage,
            "redundancy_proxy": redundancy_proxy,
            "redundancy_mix": redundancy_mix,
            "redundancy_excess_max": redundancy_excess_max,
            "redundancy_excess_mean": redundancy_excess_mean,
            "redundancy_excess": redundancy_excess,
            "redundancy_threshold": redundancy_threshold,
            "coverage_mean": coverage_mean,
            "coverage_floor": coverage_floor,
            "unique_coverage": unique_coverage,
            "low_gain_proxy": low_gain_proxy,
            "gain_size_penalty": gain_size_penalty,
            "structure_penalty": structure_penalty,
            "risk_agg": risk_agg,
            "cost_agg": cost_agg,
            "risk_agg_weights": risk_w,
            "cost_agg_weights": cost_w,
            "gain_threshold": gain_thr,
            "expert_scores": expert_scores,
            "gate": gate,
        }
