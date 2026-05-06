#!/usr/bin/env python3
"""Typed LightGCN PET scorer with explicit 3-base-score + 3-expert routing.

Implemented to match the clean PET document protocol:
- Base scores: `s_TB`, `s_TS`, `s_BS`
- Experts: relevance (`E_rel`), structure (`E_struct`), safety-cost (`E_safe`)
- Final score: dynamic sparse routing gate over the three experts
  `gate_rel * E_rel + gate_struct * E_struct + gate_safe * E_safe`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typed_lightgcn_model import (
    PETGraphTensors,
    TASK_BUNDLE_EDGE_TYPES,
    TASK_SKILL_EDGE_TYPES,
    BUNDLE_SKILL_EDGE_TYPES,
    TypedBipartiteLightGCN,
    l2norm_rows,
    load_pet_graph_tensors,
)


class TaskPETRelevance3Scorer(nn.Module):
    """Typed LightGCN PET scorer with a three-subscore relevance head."""

    def __init__(
        self,
        dim: int = 512,
        num_layers: int = 2,
        gate_temperature: float = 1.0,
        structure_mode: str = "soft",
        view_mode: str = "full",
        score_view_mode: str = "full",
        expert_subset: str = "full",
        relevance_expert_mode: str = "linear",
        expert_mode: str = "legacy",
        task_prompt_mode: str = "none",
        prompt_count: int = 4,
        graph_path_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.gate_temperature = float(gate_temperature)
        if structure_mode not in {"soft", "learned_red"}:
            raise ValueError(f"Unknown structure_mode: {structure_mode}")
        if view_mode not in {"full", "task_skill_only"}:
            raise ValueError(f"Unknown view_mode: {view_mode}")
        if score_view_mode not in {"full", "tb_only", "ts_only", "bs_only"}:
            raise ValueError(f"Unknown score_view_mode: {score_view_mode}")
        if expert_subset not in {"full", "rel_only", "rel_struct", "rel_safe"}:
            raise ValueError(f"Unknown expert_subset: {expert_subset}")
        if task_prompt_mode not in {"none", "prompt", "retrieved"}:
            raise ValueError(f"Unknown task_prompt_mode: {task_prompt_mode}")
        self.structure_mode = structure_mode
        self.view_mode = view_mode
        self.score_view_mode = score_view_mode
        self.expert_subset = expert_subset
        self.relevance_expert_mode = relevance_expert_mode
        self.expert_mode = expert_mode
        self.task_prompt_mode = task_prompt_mode
        self.graph_path_dropout = float(graph_path_dropout)

        self.tb_encoder = TypedBipartiteLightGCN(TASK_BUNDLE_EDGE_TYPES, num_layers=num_layers)
        self.ts_encoder = TypedBipartiteLightGCN(TASK_SKILL_EDGE_TYPES, num_layers=num_layers)
        self.bs_encoder = TypedBipartiteLightGCN(BUNDLE_SKILL_EDGE_TYPES, num_layers=num_layers)

        self.gate = nn.Sequential(
            nn.Linear(dim * 4 + 8, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
        )
        self.tb_mlp = nn.Sequential(nn.Linear(dim * 4, 64), nn.Tanh(), nn.Linear(64, 1))
        self.bs_base_mlp = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
        self.rel_expert_logits = nn.Parameter(torch.tensor([2.0, 1.8, 1.2, 0.8, 0.8, 0.8], dtype=torch.float32))
        self.struct_minimality_logits = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.struct_positive_logits = nn.Parameter(torch.tensor([1.2, 1.0, 1.0], dtype=torch.float32))
        self.struct_redundancy_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.safety_raw = nn.Parameter(torch.full((4,), -2.0, dtype=torch.float32))
        self.risk_agg_logits = nn.Parameter(torch.tensor([0.6, 0.4], dtype=torch.float32))
        self.cost_agg_logits = nn.Parameter(torch.tensor([0.7, 0.3], dtype=torch.float32))
        self.redundancy_threshold_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.task_prompt_tokens = nn.Parameter(torch.randn(int(prompt_count), dim, dtype=torch.float32) * 0.02)
        self.task_prompt_mlp = nn.Sequential(nn.Linear(dim * 4, 64), nn.Tanh(), nn.Linear(64, dim))
        self.task_prompt_norm = nn.LayerNorm(dim)
        self.task_prompt_scale_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        self.attn_task_proj = nn.Linear(dim, 64, bias=False)
        self.attn_skill_proj = nn.Linear(dim, 64, bias=False)
        self.attn_out = nn.Linear(64, 1, bias=False)

    def task_conditioned_bundle_attention(
        self,
        task_x: torch.Tensor,
        skill_x: torch.Tensor,
        skill_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.attn_task_proj(task_x).unsqueeze(1)
        k = self.attn_skill_proj(skill_x)
        logits = self.attn_out(torch.tanh(q + k)).squeeze(-1)
        logits = logits.masked_fill(skill_mask <= 0, -1e9)
        w = torch.softmax(logits, dim=1) * skill_mask
        w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1e-12)
        bundle_x = l2norm_rows(torch.sum(w[:, :, None] * skill_x, dim=1))
        return bundle_x, w

    @staticmethod
    def relation_match_features(
        task_bundle_x: torch.Tensor,
        bundle_x: torch.Tensor,
        task_skill_x: torch.Tensor,
        skill_pool_x: torch.Tensor,
    ) -> torch.Tensor:
        tb_bundle_dot = torch.clamp((torch.sum(task_bundle_x * bundle_x, dim=1) + 1.0) / 2.0, 0.0, 1.0)
        tb_bundle_abs = torch.clamp(1.0 - torch.mean(torch.abs(task_bundle_x - bundle_x), dim=1) / 2.0, 0.0, 1.0)
        ts_bundle_dot = torch.clamp((torch.sum(task_skill_x * bundle_x, dim=1) + 1.0) / 2.0, 0.0, 1.0)
        ts_bundle_abs = torch.clamp(1.0 - torch.mean(torch.abs(task_skill_x - bundle_x), dim=1) / 2.0, 0.0, 1.0)
        tb_skill_prod = torch.sigmoid(torch.mean(task_bundle_x * skill_pool_x, dim=1))
        ts_skill_prod = torch.sigmoid(torch.mean(task_skill_x * skill_pool_x, dim=1))
        return torch.stack(
            [tb_bundle_dot, tb_bundle_abs, ts_bundle_dot, ts_bundle_abs, tb_skill_prod, ts_skill_prod],
            dim=1,
        )

    def apply_graph_path_dropout(self, graph_available: torch.Tensor) -> torch.Tensor:
        if self.training and self.graph_path_dropout > 0.0:
            keep_prob = max(0.0, min(1.0, 1.0 - self.graph_path_dropout))
            keep = torch.bernoulli(torch.full_like(graph_available, keep_prob))
            return graph_available * keep
        return graph_available

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

    def encode(self, graph: PETGraphTensors) -> dict[str, torch.Tensor]:
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
        bundle_x = l2norm_rows(bundle_x / torch.clamp(deg[:, None], min=1.0))

        skill_context = None
        if self.task_prompt_mode == "retrieved":
            skill_context = self.retrieved_skill_context(graph.task_emb, graph.skill_emb, graph.task_prompt_edges)
        task_x = self.adapt_task_embeddings(graph.task_emb, skill_context)

        if self.view_mode == "task_skill_only":
            task_ts, skill_ts = self.ts_encoder.propagate(task_x, graph.skill_emb, graph.task_skill_edges)
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
                "bundle_ts_mean": ts_bundle,
                "bundle_bs_member_align_mean": member_align_mean,
                "task_input": task_x,
            }

        task_tb, bundle_tb = self.tb_encoder.propagate(task_x, bundle_x, graph.task_bundle_edges)
        task_ts, skill_ts = self.ts_encoder.propagate(task_x, graph.skill_emb, graph.task_skill_edges)
        bundle_bs, skill_bs = self.bs_encoder.propagate(bundle_x, graph.skill_emb, graph.bundle_skill_edges)

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
        member_ids = graph.bundle_skill_padded[bundle_local_idx]
        member_mask = graph.bundle_skill_mask[bundle_local_idx]
        denom = torch.clamp(member_mask.sum(dim=1, keepdim=True), min=1.0)

        # 8.1: task-bundle score (s_TB).
        t_tb = enc["task_tb"][task_local_idx]
        s_bs_members = enc["skill_bs"][member_ids]
        b_tb_attn, w = self.task_conditioned_bundle_attention(t_tb, s_bs_members, member_mask)
        b_tb_attn = l2norm_rows(torch.sum(w[:, :, None] * s_bs_members, dim=1))
        tb_feat = torch.cat([t_tb, b_tb_attn, torch.abs(t_tb - b_tb_attn), t_tb * b_tb_attn], dim=1)
        score_tb = torch.sigmoid(self.tb_mlp(tb_feat).squeeze(-1))  # s_TB

        # 8.2: task-skill pooled score (s_TS).
        t_ts = enc["task_ts"][task_local_idx]
        s_ts_members = enc["skill_ts"][member_ids]
        b_ts = l2norm_rows(torch.sum(s_ts_members * member_mask[:, :, None], dim=1) / denom)
        score_ts = torch.sigmoid(torch.sum(t_ts * b_ts, dim=1))  # s_TS = sigmoid(t_TS^T b_TS)

        pair_mean = graph.bundle_features[bundle_local_idx, 5]
        pair_max = graph.bundle_features[bundle_local_idx, 6]
        pair_min = graph.bundle_features[bundle_local_idx, 7]
        b_bs_dyn, w_dyn = self.task_conditioned_bundle_attention(t_tb, s_bs_members, member_mask)
        b_bs_graph = enc["bundle_bs"][bundle_local_idx]
        member_align = torch.sigmoid(
            torch.sum(s_bs_members * b_bs_graph[:, None, :], dim=2).sum(dim=1) / torch.clamp(member_mask.sum(dim=1), min=1.0)
        )
        # 8.3: bundle-internal score (s_BS).
        bs_base_feat = torch.stack([member_align, torch.clamp(pair_mean, 0.0, 1.0), torch.clamp(pair_max, 0.0, 1.0), torch.clamp(pair_min, 0.0, 1.0)], dim=1)
        score_bs = torch.sigmoid(self.bs_base_mlp(bs_base_feat).squeeze(-1))  # s_BS

        if self.score_view_mode == "tb_only":
            score_ts = torch.zeros_like(score_ts)
            score_bs = torch.zeros_like(score_bs)
        elif self.score_view_mode == "ts_only":
            score_tb = torch.zeros_like(score_tb)
            score_bs = torch.zeros_like(score_bs)
        elif self.score_view_mode == "bs_only":
            score_tb = torch.zeros_like(score_tb)
            score_ts = torch.zeros_like(score_ts)

        # 9.1 relevance expert: weighted_sum(6 relevance features).
        rel_f4 = torch.clamp(1.0 - torch.mean(torch.abs(t_ts - b_ts), dim=1) / 2.0, 0.0, 1.0)
        rel_f5 = torch.clamp((torch.mean(t_ts * b_ts, dim=1) + 1.0) / 2.0, 0.0, 1.0)
        rel_f6 = torch.clamp(1.0 - torch.norm(t_ts - b_ts, dim=1) / 2.0, 0.0, 1.0)
        rel_features = torch.stack([score_tb, score_ts, score_bs, rel_f4, rel_f5, rel_f6], dim=1)
        rel_w = F.softmax(self.rel_expert_logits, dim=0)
        score_rel = torch.clamp(torch.sum(rel_features * rel_w[None, :], dim=1), 0.0, 1.0)

        # 9.2 structure expert.
        novelty = torch.clamp(1.0 - pair_mean, 0.0, 1.0)
        min_w = F.softmax(self.struct_minimality_logits, dim=0)
        minimality = torch.clamp(min_w[0] * score_ts + min_w[1] * score_bs, 0.0, 1.0)
        pos_w = F.softmax(self.struct_positive_logits, dim=0)
        positive_structure = torch.clamp(pos_w[0] * score_bs + pos_w[1] * minimality + pos_w[2] * novelty, 0.0, 1.0)
        redundancy_threshold = 0.75 + 0.20 * torch.sigmoid(self.redundancy_threshold_raw)
        redundancy_excess = torch.clamp(
            (torch.clamp(pair_max, 0.0, 1.0) - redundancy_threshold) / torch.clamp(1.0 - redundancy_threshold, min=1e-6),
            0.0,
            1.0,
        )
        redundancy_penalty = F.softplus(self.struct_redundancy_raw) * redundancy_excess
        score_struct = torch.clamp(positive_structure * torch.exp(-redundancy_penalty), 0.0, 1.0)

        # 9.3 safety-cost expert.
        risk_mean = graph.bundle_features[bundle_local_idx, 1]
        risk_max = graph.bundle_features[bundle_local_idx, 2]
        cost_mean = graph.bundle_features[bundle_local_idx, 3]
        cost_max = graph.bundle_features[bundle_local_idx, 4]
        risk_w = F.softmax(self.risk_agg_logits, dim=0)
        cost_w = F.softmax(self.cost_agg_logits, dim=0)
        risk_agg = risk_w[0] * risk_mean + risk_w[1] * risk_max
        cost_agg = cost_w[0] * cost_mean + cost_w[1] * cost_max
        safe_w = F.softplus(self.safety_raw)
        safety_penalty = (
            safe_w[0] * risk_agg
            + safe_w[1] * cost_agg
            + safe_w[2] * risk_max
            + safe_w[3] * cost_max
        )
        score_safe = torch.clamp(1.0 - torch.sigmoid(safety_penalty), 0.0, 1.0)

        # 10: dynamic expert fusion gate.
        raw_s = graph.skill_emb[member_ids]
        b0 = l2norm_rows(torch.sum(w_dyn[:, :, None] * raw_s, dim=1))
        t0 = enc["task_input"][task_local_idx]
        gate_input = torch.cat([t0, b0, torch.abs(t0 - b0), t0 * b0, graph.bundle_features[bundle_local_idx]], dim=1)
        view_gate = F.softmax(self.gate(gate_input) / max(self.gate_temperature, 1e-6), dim=1)
        expert_scores = torch.stack([score_rel, score_struct, score_safe], dim=1)
        if self.expert_subset == "rel_only":
            expert_mask = torch.tensor([1.0, 0.0, 0.0], dtype=expert_scores.dtype, device=expert_scores.device)
        elif self.expert_subset == "rel_struct":
            expert_mask = torch.tensor([1.0, 1.0, 0.0], dtype=expert_scores.dtype, device=expert_scores.device)
        elif self.expert_subset == "rel_safe":
            expert_mask = torch.tensor([1.0, 0.0, 1.0], dtype=expert_scores.dtype, device=expert_scores.device)
        else:
            expert_mask = torch.tensor([1.0, 1.0, 1.0], dtype=expert_scores.dtype, device=expert_scores.device)
        view_gate = view_gate * expert_mask[None, :]
        view_gate = view_gate / torch.clamp(view_gate.sum(dim=1, keepdim=True), min=1e-12)
        final = torch.clamp(torch.sum(view_gate * expert_scores, dim=1), 0.0, 1.0)

        return final, {
            "score_relevance": score_rel,
            "score_task_bundle": score_tb,
            "score_task_skill": score_ts,
            "score_bundle_internal": score_bs,
            "score_skill_bundle": score_struct,
            "score_safe": score_safe,
            "experts": expert_scores,
            "novelty": novelty,
            "minimality": minimality,
            "positive_structure": positive_structure,
            "redundancy_threshold": redundancy_threshold,
            "redundancy_excess": redundancy_excess,
            "redundancy_penalty": redundancy_penalty,
            "safety_penalty": safety_penalty,
            "view_scores": expert_scores,
            "view_gate": view_gate,
        }

    def score_dynamic_bundles(
        self,
        task_x: torch.Tensor,
        skill_x: torch.Tensor,
        skill_mask: torch.Tensor,
        bundle_features: torch.Tensor,
        skill_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score dynamic bundles that do not have graph-side task/bundle nodes.

        This is the inference path for unseen tasks and newly generated
        candidate bundles. Task representations come from the semantic task
        embedding after optional prompt adaptation; skill representations come
        from semantic skill embeddings; bundle representations are constructed
        on the fly by pooling member skills.
        """
        task_x = self.adapt_task_embeddings(task_x, skill_context)
        batch = skill_x.shape[0]
        t_tb = task_x.expand(batch, -1)
        t_ts = t_tb
        denom = torch.clamp(skill_mask.sum(dim=1, keepdim=True), min=1.0)

        b_tb_attn, w = self.task_conditioned_bundle_attention(t_tb, skill_x, skill_mask)
        tb_feat = torch.cat([t_tb, b_tb_attn, torch.abs(t_tb - b_tb_attn), t_tb * b_tb_attn], dim=1)
        score_tb = torch.sigmoid(self.tb_mlp(tb_feat).squeeze(-1))

        b_ts = l2norm_rows(torch.sum(skill_x * skill_mask[:, :, None], dim=1) / denom)
        score_ts = torch.sigmoid(torch.sum(t_ts * b_ts, dim=1))

        pair_mean = bundle_features[:, 5]
        pair_max = bundle_features[:, 6]
        pair_min = bundle_features[:, 7]
        b_bs_dyn, _ = self.task_conditioned_bundle_attention(t_tb, skill_x, skill_mask)
        member_align_dyn = torch.sigmoid(
            torch.sum(skill_x * b_bs_dyn[:, None, :], dim=2).sum(dim=1) / torch.clamp(skill_mask.sum(dim=1), min=1.0)
        )
        bs_base_feat = torch.stack([member_align_dyn, torch.clamp(pair_mean, 0.0, 1.0), torch.clamp(pair_max, 0.0, 1.0), torch.clamp(pair_min, 0.0, 1.0)], dim=1)
        score_bs = torch.sigmoid(self.bs_base_mlp(bs_base_feat).squeeze(-1))

        if self.score_view_mode == "tb_only":
            score_ts = torch.zeros_like(score_ts)
            score_bs = torch.zeros_like(score_bs)
        elif self.score_view_mode == "ts_only":
            score_tb = torch.zeros_like(score_tb)
            score_bs = torch.zeros_like(score_bs)
        elif self.score_view_mode == "bs_only":
            score_tb = torch.zeros_like(score_tb)
            score_ts = torch.zeros_like(score_ts)

        rel_f4 = torch.clamp(1.0 - torch.mean(torch.abs(t_ts - b_ts), dim=1) / 2.0, 0.0, 1.0)
        rel_f5 = torch.clamp((torch.mean(t_ts * b_ts, dim=1) + 1.0) / 2.0, 0.0, 1.0)
        rel_f6 = torch.clamp(1.0 - torch.norm(t_ts - b_ts, dim=1) / 2.0, 0.0, 1.0)
        rel_features = torch.stack([score_tb, score_ts, score_bs, rel_f4, rel_f5, rel_f6], dim=1)
        rel_w = F.softmax(self.rel_expert_logits, dim=0)
        score_rel = torch.clamp(torch.sum(rel_features * rel_w[None, :], dim=1), 0.0, 1.0)

        novelty = torch.clamp(1.0 - pair_mean, 0.0, 1.0)
        min_w = F.softmax(self.struct_minimality_logits, dim=0)
        minimality = torch.clamp(min_w[0] * score_ts + min_w[1] * score_bs, 0.0, 1.0)
        pos_w = F.softmax(self.struct_positive_logits, dim=0)
        positive_structure = torch.clamp(pos_w[0] * score_bs + pos_w[1] * minimality + pos_w[2] * novelty, 0.0, 1.0)
        redundancy_threshold = 0.75 + 0.20 * torch.sigmoid(self.redundancy_threshold_raw)
        redundancy_excess = torch.clamp(
            (torch.clamp(pair_max, 0.0, 1.0) - redundancy_threshold) / torch.clamp(1.0 - redundancy_threshold, min=1e-6),
            0.0,
            1.0,
        )
        redundancy_penalty = F.softplus(self.struct_redundancy_raw) * redundancy_excess
        score_struct = torch.clamp(positive_structure * torch.exp(-redundancy_penalty), 0.0, 1.0)

        b0 = l2norm_rows(torch.sum(w[:, :, None] * skill_x, dim=1))
        gate_input = torch.cat([t_tb, b0, torch.abs(t_tb - b0), t_tb * b0, bundle_features], dim=1)
        view_gate = F.softmax(self.gate(gate_input) / max(self.gate_temperature, 1e-6), dim=1)

        risk_mean = bundle_features[:, 1]
        risk_max = bundle_features[:, 2]
        cost_mean = bundle_features[:, 3]
        cost_max = bundle_features[:, 4]
        risk_w = F.softmax(self.risk_agg_logits, dim=0)
        cost_w = F.softmax(self.cost_agg_logits, dim=0)
        risk_agg = risk_w[0] * risk_mean + risk_w[1] * risk_max
        cost_agg = cost_w[0] * cost_mean + cost_w[1] * cost_max
        safe_w = F.softplus(self.safety_raw)
        safety_penalty = (
            safe_w[0] * risk_agg
            + safe_w[1] * cost_agg
            + safe_w[2] * risk_max
            + safe_w[3] * cost_max
        )
        score_safe = torch.clamp(1.0 - torch.sigmoid(safety_penalty), 0.0, 1.0)
        expert_scores = torch.stack([score_rel, score_struct, score_safe], dim=1)
        if self.expert_subset == "rel_only":
            expert_mask = torch.tensor([1.0, 0.0, 0.0], dtype=expert_scores.dtype, device=expert_scores.device)
        elif self.expert_subset == "rel_struct":
            expert_mask = torch.tensor([1.0, 1.0, 0.0], dtype=expert_scores.dtype, device=expert_scores.device)
        elif self.expert_subset == "rel_safe":
            expert_mask = torch.tensor([1.0, 0.0, 1.0], dtype=expert_scores.dtype, device=expert_scores.device)
        else:
            expert_mask = torch.tensor([1.0, 1.0, 1.0], dtype=expert_scores.dtype, device=expert_scores.device)
        view_gate = view_gate * expert_mask[None, :]
        view_gate = view_gate / torch.clamp(view_gate.sum(dim=1, keepdim=True), min=1e-12)
        final = torch.clamp(torch.sum(view_gate * expert_scores, dim=1), 0.0, 1.0)

        return final, {
            "score_relevance": score_rel,
            "score_task_bundle": score_tb,
            "score_task_skill": score_ts,
            "score_bundle_internal": score_bs,
            "score_skill_bundle": score_struct,
            "score_safe": score_safe,
            "experts": expert_scores,
            "member_align_dyn": member_align_dyn,
            "novelty": novelty,
            "minimality": minimality,
            "positive_structure": positive_structure,
            "redundancy_threshold": redundancy_threshold,
            "redundancy_excess": redundancy_excess,
            "redundancy_penalty": redundancy_penalty,
            "safety_penalty": safety_penalty,
            "view_scores": expert_scores,
            "view_gate": view_gate,
        }


__all__ = [
    "TaskPETRelevance3Scorer",
    "PETGraphTensors",
    "TypedBipartiteLightGCN",
    "load_pet_graph_tensors",
    "l2norm_rows",
]
