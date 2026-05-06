#!/usr/bin/env python3
"""
Train internal score parameters with PyTorch.

Loss:
- Listwise softmax over candidate bundles
- Contrastive positive/negative bundle loss from constructed supervision
- Explicit length/risk control loss
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from bundle_scorer_stage05 import score_bundle as stage05_score_bundle


def _set_requires_grad(named_params: list[tuple[str, torch.nn.Parameter]], enabled_prefixes: tuple[str, ...]) -> None:
    for name, param in named_params:
        param.requires_grad = any(name.startswith(prefix) for prefix in enabled_prefixes)


def _build_optimizer(
    named_params: list[tuple[str, torch.nn.Parameter]],
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    params = [p for _name, p in named_params if p.requires_grad]
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def l2norm_vec(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x), min=eps)


def l2norm_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, dim=1, keepdim=True), min=eps)


def set_metrics(pred: list[int], gold: list[int]) -> dict[str, float]:
    p = set(int(x) for x in pred)
    g = set(int(x) for x in gold)
    inter = len(p & g)
    precision = inter / max(len(p), 1)
    recall = inter / max(len(g), 1)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    jaccard = inter / max(len(p | g), 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
        "bundle_size_mae": float(abs(len(p) - len(g))),
    }


def avg(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = list(rows[0].keys())
    return {k: float(sum(r[k] for r in rows) / len(rows)) for k in keys}


@dataclass
class Candidate:
    indices: list[int]
    risk_penalty: float
    token_penalty: float
    base_bundle_score: float


@dataclass
class RoundCandidate:
    indices: list[int]
    risk_penalty: float
    token_penalty: float
    gold_overlap: int
    bundle_size: int
    is_gold_subset: bool
    subset_ratio: float


@dataclass
class ContrastivePair:
    positive: Candidate
    negative: Candidate
    margin: float
    weight: float
    negative_role: str


def softmax_np(z: np.ndarray, temp: float = 1.0) -> np.ndarray:
    t = max(float(temp), 1e-6)
    x = np.asarray(z, dtype=np.float64) / t
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.clip(np.sum(e), 1e-12, None)


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = float(sum(xs) / n)
    my = float(sum(ys) / n)
    num = float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)))
    dx = float(sum((x - mx) ** 2 for x in xs))
    dy = float(sum((y - my) ** 2 for y in ys))
    if dx <= 1e-18 or dy <= 1e-18:
        return 0.0
    return float(num / math.sqrt(dx * dy))


def _rank_desc(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    out = [0] * len(values)
    for r, i in enumerate(order):
        out[i] = r
    return out


def _spearman_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    rx = _rank_desc(xs)
    ry = _rank_desc(ys)
    return _pearson_corr([float(x) for x in rx], [float(y) for y in ry])


def _bundle_quality(indices: list[int], gold_set: set[int], gold_size: int) -> float:
    p = set(int(x) for x in indices)
    if not p:
        return 0.0
    inter = float(len(p & gold_set))
    prec = inter / max(float(len(p)), 1.0)
    rec = inter / max(float(gold_size), 1.0)
    f1 = 0.0 if prec + rec <= 1e-12 else 2.0 * prec * rec / (prec + rec)
    return float(f1)


def _bundle_preference_utility(
    indices: list[int],
    gold_set: set[int],
    gold_size: int,
    risk_penalty: float,
    token_penalty: float,
) -> float:
    f1 = _bundle_quality(indices, gold_set, gold_size)
    size_gap = abs(float(len(indices)) - float(gold_size))
    return 100.0 * f1 - 1.0 * size_gap - 0.02 * float(risk_penalty) - 0.02 * float(token_penalty)


def _build_rerank_preference_pairs(
    rerank_beam: dict[str, Any],
    gold_lookup: dict[int, list[int]],
) -> dict[int, list[tuple[list[int], list[int]]]]:
    out: dict[int, list[tuple[list[int], list[int]]]] = {}
    for task in rerank_beam.get("tasks", []):
        ti = int(task.get("task_index", -1))
        if ti not in gold_lookup:
            continue
        cands = (
            task.get("beam", {}).get("final_top_states_reranked")
            or task.get("beam", {}).get("final_top_states")
            or []
        )
        if len(cands) < 2:
            continue
        gold_set = set(int(x) for x in gold_lookup[ti])
        scored = []
        for c in cands[:6]:
            idx = [int(x) for x in c.get("selected_indices_47153space", [])]
            if not idx:
                continue
            util = _bundle_preference_utility(
                idx,
                gold_set,
                len(gold_set),
                float(c.get("risk_penalty", 0.0)),
                float(c.get("token_penalty", 0.0)),
            )
            scored.append((util, idx))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        pairs = []
        for _u, neg in scored[1:]:
            if neg != best:
                pairs.append((best, neg))
        if pairs:
            out[ti] = pairs
    return out


def _dedupe_candidates(cands: list[Candidate]) -> list[Candidate]:
    kept: dict[tuple[int, ...], Candidate] = {}
    for c in cands:
        key = tuple(sorted(int(x) for x in c.indices))
        old = kept.get(key)
        if old is None or float(c.base_bundle_score) > float(old.base_bundle_score):
            kept[key] = c
    return list(kept.values())


def _contrastive_role(row: dict[str, Any]) -> str:
    supervision = row.get("supervision", {}) or {}
    return str(supervision.get("role", row.get("bundle_type", "")))


def _build_contrastive_pairs(contrastive_rows: list[dict[str, Any]]) -> dict[int, list[ContrastivePair]]:
    by_task: dict[int, list[dict[str, Any]]] = {}
    for row in contrastive_rows:
        if not bool(row.get("valid", True)):
            continue
        ti = int(row.get("task_index", -1))
        if ti < 0 or not row.get("indices"):
            continue
        by_task.setdefault(ti, []).append(row)

    pairs_by_task: dict[int, list[ContrastivePair]] = {}
    for ti, rows in by_task.items():
        positives = []
        negatives = []
        for row in rows:
            role = _contrastive_role(row)
            item = Candidate(
                indices=[int(x) for x in row.get("indices", [])],
                risk_penalty=float(row.get("risk_penalty", row.get("p_red", 0.0))),
                token_penalty=float(row.get("token_penalty", 0.0)),
                base_bundle_score=float(row.get("bundle_score", 0.0)),
            )
            meta = {
                "candidate": item,
                "weight": float((row.get("supervision", {}) or {}).get("weight", 1.0)),
                "margin": float((row.get("supervision", {}) or {}).get("target_margin_from_gold", 0.0)) / 100.0,
                "role": role,
            }
            # Keep only canonical gold positives for contrastive supervision.
            if role in {"gold_bundle"}:
                positives.append(meta)
            else:
                negatives.append(meta)

        task_pairs: list[ContrastivePair] = []
        for pos in positives:
            for neg in negatives:
                task_pairs.append(
                    ContrastivePair(
                        positive=pos["candidate"],
                        negative=neg["candidate"],
                        margin=max(float(neg["margin"]), 1e-3),
                        weight=0.5 * (float(pos["weight"]) + float(neg["weight"])),
                        negative_role=str(neg["role"]),
                    )
                )
        if task_pairs:
            pairs_by_task[ti] = task_pairs
    return pairs_by_task


def _build_old_to_new_index_map(old_skill_emb: np.ndarray, new_skill_emb: np.ndarray) -> dict[int, int]:
    key_to_new: dict[str, int] = {}
    for i in range(new_skill_emb.shape[0]):
        key = hashlib.md5(np.asarray(new_skill_emb[i], dtype=np.float32).tobytes()).hexdigest()
        key_to_new.setdefault(key, i)
    out: dict[int, int] = {}
    for i in range(old_skill_emb.shape[0]):
        key = hashlib.md5(np.asarray(old_skill_emb[i], dtype=np.float32).tobytes()).hexdigest()
        j = key_to_new.get(key)
        if j is not None:
            out[i] = int(j)
    return out


def _prepare_candidate_groups(
    pool: list[Candidate],
    *,
    skill_device: torch.device,
    score_device: torch.device,
) -> list[dict[str, Any]]:
    by_len: dict[int, list[int]] = {}
    for pi, c in enumerate(pool):
        by_len.setdefault(len(c.indices), []).append(pi)
    groups: list[dict[str, Any]] = []
    for pool_indices in by_len.values():
        idx_mat = torch.tensor(
            [pool[j].indices for j in pool_indices],
            dtype=torch.long,
            device=skill_device,
        )
        risk_batch = torch.tensor(
            [float(pool[j].risk_penalty) for j in pool_indices],
            dtype=torch.float32,
            device=score_device,
        )
        token_batch = torch.tensor(
            [float(pool[j].token_penalty) for j in pool_indices],
            dtype=torch.float32,
            device=score_device,
        )
        groups.append(
            {
                "pool_indices": pool_indices,
                "idx_mat": idx_mat,
                "risk_batch": risk_batch,
                "token_batch": token_batch,
            }
        )
    return groups


def _score_candidate_groups(
    model: "LearnableScore",
    task_vec: torch.Tensor,
    skill_emb: torch.Tensor,
    pool: list[Candidate],
    groups: list[dict[str, Any]],
) -> torch.Tensor:
    out: list[torch.Tensor | None] = [None] * len(pool)
    for g in groups:
        svecs_batch = skill_emb[g["idx_mat"]]
        scores = model.score_bundle_batch_same_size(task_vec, svecs_batch, g["risk_batch"], g["token_batch"])
        for kk, pool_idx in enumerate(g["pool_indices"]):
            out[pool_idx] = scores[kk]
    return torch.stack([x for x in out if x is not None], dim=0)


def _prepare_contrastive_groups(
    pairs: list["ContrastivePair"],
    *,
    skill_device: torch.device,
    score_device: torch.device,
) -> list[dict[str, Any]]:
    by_shape: dict[tuple[int, int, bool], list[ContrastivePair]] = {}
    structural_roles = {"structure_negative", "weak_negative", "hard_negative"}
    for pair in pairs:
        key = (
            len(pair.positive.indices),
            len(pair.negative.indices),
            pair.negative_role in structural_roles,
        )
        by_shape.setdefault(key, []).append(pair)
    groups: list[dict[str, Any]] = []
    for (_pos_len, _neg_len, use_struct), grp in by_shape.items():
        pos_idx_mat = torch.tensor(
            [p.positive.indices for p in grp],
            dtype=torch.long,
            device=skill_device,
        )
        neg_idx_mat = torch.tensor(
            [p.negative.indices for p in grp],
            dtype=torch.long,
            device=skill_device,
        )
        groups.append(
            {
                "pos_idx_mat": pos_idx_mat,
                "neg_idx_mat": neg_idx_mat,
                "pos_risk_batch": torch.tensor([float(p.positive.risk_penalty) for p in grp], dtype=torch.float32, device=score_device),
                "pos_token_batch": torch.tensor([float(p.positive.token_penalty) for p in grp], dtype=torch.float32, device=score_device),
                "neg_risk_batch": torch.tensor([float(p.negative.risk_penalty) for p in grp], dtype=torch.float32, device=score_device),
                "neg_token_batch": torch.tensor([float(p.negative.token_penalty) for p in grp], dtype=torch.float32, device=score_device),
                "margin_batch": torch.tensor([float(p.margin) for p in grp], dtype=torch.float32, device=score_device),
                "weight_batch": torch.tensor([float(p.weight) for p in grp], dtype=torch.float32, device=score_device),
                "use_struct_contrast": use_struct,
            }
        )
    return groups


def _filter_attention_pretrain_pairs(pairs: list["ContrastivePair"]) -> list["ContrastivePair"]:
    allowed_negative_roles = {"structure_negative", "hard_negative", "weak_negative"}
    return [p for p in pairs if p.negative_role in allowed_negative_roles]


class LearnableScore(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        risk_lambda_max: float = 0.05,
        token_lambda_max: float = 0.03,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.risk_lambda_max = float(risk_lambda_max)
        self.token_lambda_max = float(token_lambda_max)

        self.tau_raw = torch.nn.Parameter(torch.full((num_heads,), 0.0))

        # Lightweight relation scorer for attention:
        # [task_aff, novelty, redundancy, conflict] -> tiny MLP -> logits,
        # with qkv task-match residual added outside the MLP.
        self.rel_proj = torch.nn.Linear(4, 4)
        self.rel_out = torch.nn.Linear(4, 1)
        # Three-view semantic fusion:
        # [skill-task, skill-bundle(attention), bundle-task]
        self.semantic_logits = torch.nn.Parameter(torch.tensor([0.25, 0.50, 0.25]))
        # Learnable alpha for skill-task branch:
        # score = alpha * max_sim + (1 - alpha) * mean_sim
        self.skill_task_alpha_raw = torch.nn.Parameter(torch.tensor(0.0))
        self.bundle_task_proj = torch.nn.Linear(4, 8)
        self.bundle_task_out = torch.nn.Linear(8, 1)
        # Learnable thresholds used by second-view attention features.
        self.complement_low_raw = torch.nn.Parameter(torch.tensor(0.0))
        self.redundancy_threshold_raw = torch.nn.Parameter(torch.tensor(0.0))

        # penalties (non-negative)
        # constrain to [0, max] via sigmoid.
        self.risk_lambda_raw = torch.nn.Parameter(torch.tensor(0.0))
        self.token_lambda_raw = torch.nn.Parameter(torch.tensor(0.0))

        # gain-threshold-based length penalty config (no hard target size)
        self.gain_threshold = 0.62
        self.growth = 0.008

    def _normalize_rel_feat(self, rel_feat: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(rel_feat, dim=-2, keepdim=True)
        std = torch.std(rel_feat, dim=-2, keepdim=True, unbiased=False)
        return (rel_feat - mean) / torch.clamp(std, min=1e-4)

    def _normalize_relation_residual(self, fallback_logits: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(fallback_logits, dim=-1, keepdim=True)
        std = torch.std(fallback_logits, dim=-1, keepdim=True, unbiased=False)
        return (fallback_logits - mean) / torch.clamp(std, min=1e-4)

    def _relation_logits(self, rel_feat: torch.Tensor, fallback_logits: torch.Tensor) -> torch.Tensor:
        rel_h = torch.tanh(self.rel_proj(self._normalize_rel_feat(rel_feat)))
        rel_logits = self.rel_out(rel_h).squeeze(-1)
        # Keep attention useful even if the MLP is weak by anchoring to task-match logits.
        return rel_logits + 0.5 * self._normalize_relation_residual(fallback_logits)

    def _params_constrained(self) -> dict[str, torch.Tensor]:
        sem = F.softmax(self.semantic_logits, dim=0)
        tau = 0.05 + F.softplus(self.tau_raw)
        risk_lambda = self.risk_lambda_max * torch.sigmoid(self.risk_lambda_raw)
        token_lambda = self.token_lambda_max * torch.sigmoid(self.token_lambda_raw)
        skill_task_alpha = torch.sigmoid(self.skill_task_alpha_raw)
        complement_low = 0.05 + 0.90 * torch.sigmoid(self.complement_low_raw)
        redundancy_threshold = 0.50 + 0.49 * torch.sigmoid(self.redundancy_threshold_raw)
        return {
            "sem": sem,
            "tau": tau,
            "risk_lambda": risk_lambda,
            "token_lambda": token_lambda,
            "skill_task_alpha": skill_task_alpha,
            "complement_low": complement_low,
            "redundancy_threshold": redundancy_threshold,
        }

    def score_bundle(self, task_vec: torch.Tensor, skill_vecs: torch.Tensor, risk_penalty: float, token_penalty: float) -> torch.Tensor:
        p = self._params_constrained()
        t = l2norm_vec(task_vec)
        s = l2norm_rows(skill_vecs)
        m = s.shape[0]
        tau_eff = torch.mean(p["tau"])

        # bundle-task score: mean-pooled bundle vector plus low-dim summary
        # features passed through a small MLP.
        bundle_mean = l2norm_vec(torch.mean(s, dim=0))
        sim_pool = torch.clamp((torch.dot(t, bundle_mean) + 1.0) / 2.0, 0.0, 1.0)

        # Bundle self-attention: contextualize each item with item-item interaction.
        # h_i = SelfAttn(e_i, {e_j}_{j in B})
        self_logits = (s @ s.T) / torch.clamp(tau_eff, min=1e-6)
        self_attn = F.softmax(self_logits, dim=1)
        contextual = l2norm_rows(self_attn @ s)

        # Structure-aware attention:
        # base task relevance + novelty bonus - redundancy penalty - conflict penalty.
        qkv_logits = (contextual @ t) / torch.clamp(tau_eff, min=1e-6)
        cos = contextual @ t  # [-1,1], shape [m]
        cos01 = torch.clamp((cos + 1.0) / 2.0, 0.0, 1.0)
        if m > 1:
            sim = contextual @ contextual.T
            sim = sim - torch.diag_embed(torch.diag(sim))
            sim01 = torch.clamp((sim + 1.0) / 2.0, 0.0, 1.0)
            mean_sim = torch.mean(sim01, dim=1)
            max_sim = torch.max(sim01, dim=1).values
        else:
            mean_sim = torch.zeros_like(cos)
            max_sim = torch.zeros_like(cos)
        mean_sim_scaled = torch.clamp(mean_sim / torch.clamp(p["redundancy_threshold"], min=1e-6), 0.0, 1.0)
        task_low = torch.clamp(
            (torch.clamp(p["complement_low"], min=1e-6) - cos01) / torch.clamp(p["complement_low"], min=1e-6),
            0.0,
            1.0,
        )
        novelty = cos01 * (1.0 - mean_sim_scaled)
        conflict = task_low * mean_sim_scaled
        rel_feat = torch.stack([cos01, novelty, mean_sim_scaled, conflict], dim=1)
        rel_logits = self._relation_logits(rel_feat, qkv_logits)
        alpha_i = F.softmax(rel_logits, dim=0)

        # Attention branch score is computed directly from weighted relation
        # statistics (task-affinity/novelty/redundancy/conflict), instead of
        # a second dot-product over an aggregated semantic vector.
        attn_task_aff = torch.sum(alpha_i * cos01)
        attn_novelty = torch.sum(alpha_i * novelty)
        attn_redundancy = torch.sum(alpha_i * mean_sim_scaled)
        attn_conflict = torch.sum(alpha_i * conflict)
        attention_score = torch.clamp(
            0.5 * (attn_task_aff + attn_novelty)
            + 0.5 * ((1.0 - attn_redundancy) + (1.0 - attn_conflict)) * 0.5,
            0.0,
            1.0,
        )

        # skill-task score: aggregate per-skill relevance to task.
        sim_each = torch.clamp(((s @ t) + 1.0) / 2.0, 0.0, 1.0)
        sim_max = torch.max(sim_each) if sim_each.numel() > 0 else torch.tensor(0.0, device=t.device)
        sim_mean = torch.mean(sim_each) if sim_each.numel() > 0 else torch.tensor(0.0, device=t.device)
        alpha = p["skill_task_alpha"]
        skill_task_score = torch.clamp(alpha * sim_max + (1.0 - alpha) * sim_mean, 0.0, 1.0)
        bundle_feat = torch.stack(
            [
                sim_pool,
                sim_mean,
                sim_max,
                torch.tensor(min(m, 8) / 8.0, dtype=torch.float32, device=t.device),
            ]
        )
        bundle_h = torch.tanh(self.bundle_task_proj(bundle_feat))
        bundle_task_score = torch.sigmoid(self.bundle_task_out(bundle_h)).squeeze(-1)

        sem_w = p["sem"]
        semantic_score = sem_w[0] * skill_task_score + sem_w[1] * attention_score + sem_w[2] * bundle_task_score
        fused = semantic_score

        length_penalty = torch.tensor(0.0, device=t.device)
        if m > 1:
            gain_shortfall = torch.clamp(self.gain_threshold - fused, 0.0, 1.0)
            length_penalty = (m - 1) * self.growth * gain_shortfall

        final = fused - length_penalty - p["risk_lambda"] * float(risk_penalty) - p["token_lambda"] * float(token_penalty)
        return final

    def score_bundle_batch_same_size(
        self,
        task_vec: torch.Tensor,
        skill_vecs_batch: torch.Tensor,
        risk_penalty_batch: torch.Tensor,
        token_penalty_batch: torch.Tensor,
    ) -> torch.Tensor:
        # Batch variant for bundles with the same size: [B, M, D].
        p = self._params_constrained()
        t = l2norm_vec(task_vec)
        s = l2norm_rows(skill_vecs_batch)
        bsz = s.shape[0]
        m = s.shape[1]
        tau_eff = torch.mean(p["tau"])

        bundle_mean = l2norm_rows(torch.mean(s, dim=1))
        sim_pool = torch.clamp((torch.sum(bundle_mean * t.unsqueeze(0), dim=1) + 1.0) / 2.0, 0.0, 1.0)

        self_logits = torch.matmul(s, s.transpose(1, 2)) / torch.clamp(tau_eff, min=1e-6)
        self_attn = F.softmax(self_logits, dim=2)
        contextual = l2norm_rows(torch.matmul(self_attn, s))

        q = t.view(1, -1, 1)
        qkv_logits = (torch.matmul(contextual, q).squeeze(-1)) / torch.clamp(tau_eff, min=1e-6)
        cos = torch.matmul(contextual, q).squeeze(-1)
        cos01 = torch.clamp((cos + 1.0) / 2.0, 0.0, 1.0)
        if m > 1:
            sim = torch.matmul(contextual, contextual.transpose(1, 2))
            eye = torch.eye(m, device=sim.device, dtype=sim.dtype).unsqueeze(0)
            sim = sim - sim * eye
            sim01 = torch.clamp((sim + 1.0) / 2.0, 0.0, 1.0)
            mean_sim = torch.mean(sim01, dim=2)
            max_sim = torch.max(sim01, dim=2).values
        else:
            mean_sim = torch.zeros_like(cos)
            max_sim = torch.zeros_like(cos)
        mean_sim_scaled = torch.clamp(mean_sim / torch.clamp(p["redundancy_threshold"], min=1e-6), 0.0, 1.0)
        task_low = torch.clamp(
            (torch.clamp(p["complement_low"], min=1e-6) - cos01) / torch.clamp(p["complement_low"], min=1e-6),
            0.0,
            1.0,
        )
        novelty = cos01 * (1.0 - mean_sim_scaled)
        conflict = task_low * mean_sim_scaled
        rel_feat = torch.stack([cos01, novelty, mean_sim_scaled, conflict], dim=2)
        rel_logits = self._relation_logits(rel_feat, qkv_logits)
        alpha_i = F.softmax(rel_logits, dim=1)

        attn_task_aff = torch.sum(alpha_i * cos01, dim=1)
        attn_novelty = torch.sum(alpha_i * novelty, dim=1)
        attn_redundancy = torch.sum(alpha_i * mean_sim_scaled, dim=1)
        attn_conflict = torch.sum(alpha_i * conflict, dim=1)
        attention_score = torch.clamp(
            0.5 * (attn_task_aff + attn_novelty)
            + 0.5 * ((1.0 - attn_redundancy) + (1.0 - attn_conflict)) * 0.5,
            0.0,
            1.0,
        )

        sim_each = torch.clamp((torch.matmul(s, q).squeeze(-1) + 1.0) / 2.0, 0.0, 1.0)
        sim_max = torch.max(sim_each, dim=1).values
        sim_mean = torch.mean(sim_each, dim=1)
        alpha = p["skill_task_alpha"]
        skill_task_score = torch.clamp(alpha * sim_max + (1.0 - alpha) * sim_mean, 0.0, 1.0)

        len_feat = torch.full((bsz,), float(min(m, 8) / 8.0), dtype=torch.float32, device=t.device)
        bundle_feat = torch.stack([sim_pool, sim_mean, sim_max, len_feat], dim=1)
        bundle_h = torch.tanh(self.bundle_task_proj(bundle_feat))
        bundle_task_score = torch.sigmoid(self.bundle_task_out(bundle_h)).squeeze(-1)

        sem_w = p["sem"]
        semantic_score = sem_w[0] * skill_task_score + sem_w[1] * attention_score + sem_w[2] * bundle_task_score
        fused = semantic_score
        length_penalty = torch.zeros_like(fused)
        if m > 1:
            gain_shortfall = torch.clamp(self.gain_threshold - fused, 0.0, 1.0)
            length_penalty = (m - 1) * self.growth * gain_shortfall
        final = fused - length_penalty - p["risk_lambda"] * risk_penalty_batch - p["token_lambda"] * token_penalty_batch
        return final

    def score_bundle_with_stats_batch_same_size(
        self,
        task_vec: torch.Tensor,
        skill_vecs_batch: torch.Tensor,
        risk_penalty_batch: torch.Tensor,
        token_penalty_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p = self._params_constrained()
        t = l2norm_vec(task_vec)
        s = l2norm_rows(skill_vecs_batch)
        bsz = s.shape[0]
        m = s.shape[1]
        tau_eff = torch.mean(p["tau"])

        bundle_mean = l2norm_rows(torch.mean(s, dim=1))
        sim_pool = torch.clamp((torch.sum(bundle_mean * t.unsqueeze(0), dim=1) + 1.0) / 2.0, 0.0, 1.0)

        self_logits = torch.matmul(s, s.transpose(1, 2)) / torch.clamp(tau_eff, min=1e-6)
        self_attn = F.softmax(self_logits, dim=2)
        contextual = l2norm_rows(torch.matmul(self_attn, s))

        q = t.view(1, -1, 1)
        qkv_logits = (torch.matmul(contextual, q).squeeze(-1)) / torch.clamp(tau_eff, min=1e-6)
        cos = torch.matmul(contextual, q).squeeze(-1)
        cos01 = torch.clamp((cos + 1.0) / 2.0, 0.0, 1.0)
        if m > 1:
            sim = torch.matmul(contextual, contextual.transpose(1, 2))
            eye = torch.eye(m, device=sim.device, dtype=sim.dtype).unsqueeze(0)
            sim = sim - sim * eye
            sim01 = torch.clamp((sim + 1.0) / 2.0, 0.0, 1.0)
            mean_sim = torch.mean(sim01, dim=2)
            max_sim = torch.max(sim01, dim=2).values
        else:
            mean_sim = torch.zeros_like(cos)
            max_sim = torch.zeros_like(cos)
        mean_sim_scaled = torch.clamp(mean_sim / torch.clamp(p["redundancy_threshold"], min=1e-6), 0.0, 1.0)
        task_low = torch.clamp(
            (torch.clamp(p["complement_low"], min=1e-6) - cos01) / torch.clamp(p["complement_low"], min=1e-6),
            0.0,
            1.0,
        )
        novelty = cos01 * (1.0 - mean_sim_scaled)
        conflict = task_low * mean_sim_scaled
        rel_feat = torch.stack([cos01, novelty, mean_sim_scaled, conflict], dim=2)
        rel_logits = self._relation_logits(rel_feat, qkv_logits)
        alpha_i = F.softmax(rel_logits, dim=1)

        attn_task_aff = torch.sum(alpha_i * cos01, dim=1)
        attn_novelty = torch.sum(alpha_i * novelty, dim=1)
        attn_redundancy = torch.sum(alpha_i * mean_sim_scaled, dim=1)
        attn_conflict = torch.sum(alpha_i * conflict, dim=1)
        attention_score = torch.clamp(
            0.5 * (attn_task_aff + attn_novelty)
            + 0.5 * ((1.0 - attn_redundancy) + (1.0 - attn_conflict)) * 0.5,
            0.0,
            1.0,
        )

        sim_each = torch.clamp((torch.matmul(s, q).squeeze(-1) + 1.0) / 2.0, 0.0, 1.0)
        sim_max = torch.max(sim_each, dim=1).values
        sim_mean = torch.mean(sim_each, dim=1)
        alpha = p["skill_task_alpha"]
        skill_task_score = torch.clamp(alpha * sim_max + (1.0 - alpha) * sim_mean, 0.0, 1.0)

        len_feat = torch.full((bsz,), float(min(m, 8) / 8.0), dtype=torch.float32, device=t.device)
        bundle_feat = torch.stack([sim_pool, sim_mean, sim_max, len_feat], dim=1)
        bundle_h = torch.tanh(self.bundle_task_proj(bundle_feat))
        bundle_task_score = torch.sigmoid(self.bundle_task_out(bundle_h)).squeeze(-1)

        sem_w = p["sem"]
        semantic_score = sem_w[0] * skill_task_score + sem_w[1] * attention_score + sem_w[2] * bundle_task_score
        fused = semantic_score
        length_penalty = torch.zeros_like(fused)
        if m > 1:
            gain_shortfall = torch.clamp(self.gain_threshold - fused, 0.0, 1.0)
            length_penalty = (m - 1) * self.growth * gain_shortfall
        final = fused - length_penalty - p["risk_lambda"] * risk_penalty_batch - p["token_lambda"] * token_penalty_batch
        stats = {
            "novelty_mean": torch.sum(alpha_i * novelty, dim=1),
            "redundancy_mean": torch.sum(alpha_i * mean_sim_scaled, dim=1),
            "conflict_mean": torch.sum(alpha_i * conflict, dim=1),
            "task_affinity_mean": torch.sum(alpha_i * cos01, dim=1),
            "attention_score": attention_score,
            "bundle_task_score": bundle_task_score,
        }
        return final, stats

    def score_bundle_with_stats(
        self,
        task_vec: torch.Tensor,
        skill_vecs: torch.Tensor,
        risk_penalty: float,
        token_penalty: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p = self._params_constrained()
        t = l2norm_vec(task_vec)
        s = l2norm_rows(skill_vecs)
        m = s.shape[0]
        tau_eff = torch.mean(p["tau"])

        bundle_mean = l2norm_vec(torch.mean(s, dim=0))
        sim_pool = torch.clamp((torch.dot(t, bundle_mean) + 1.0) / 2.0, 0.0, 1.0)

        self_logits = (s @ s.T) / torch.clamp(tau_eff, min=1e-6)
        self_attn = F.softmax(self_logits, dim=1)
        contextual = l2norm_rows(self_attn @ s)

        qkv_logits = (contextual @ t) / torch.clamp(tau_eff, min=1e-6)
        cos = contextual @ t
        cos01 = torch.clamp((cos + 1.0) / 2.0, 0.0, 1.0)
        if m > 1:
            sim = contextual @ contextual.T
            sim = sim - torch.diag_embed(torch.diag(sim))
            sim01 = torch.clamp((sim + 1.0) / 2.0, 0.0, 1.0)
            mean_sim = torch.mean(sim01, dim=1)
            max_sim = torch.max(sim01, dim=1).values
        else:
            mean_sim = torch.zeros_like(cos)
            max_sim = torch.zeros_like(cos)
        mean_sim_scaled = torch.clamp(mean_sim / torch.clamp(p["redundancy_threshold"], min=1e-6), 0.0, 1.0)
        task_low = torch.clamp(
            (torch.clamp(p["complement_low"], min=1e-6) - cos01) / torch.clamp(p["complement_low"], min=1e-6),
            0.0,
            1.0,
        )
        novelty = cos01 * (1.0 - mean_sim_scaled)
        conflict = task_low * mean_sim_scaled
        rel_feat = torch.stack([cos01, novelty, mean_sim_scaled, conflict], dim=1)
        rel_logits = self._relation_logits(rel_feat, qkv_logits)
        alpha_i = F.softmax(rel_logits, dim=0)

        attn_task_aff = torch.sum(alpha_i * cos01)
        attn_novelty = torch.sum(alpha_i * novelty)
        attn_redundancy = torch.sum(alpha_i * mean_sim_scaled)
        attn_conflict = torch.sum(alpha_i * conflict)
        attention_score = torch.clamp(
            0.5 * (attn_task_aff + attn_novelty)
            + 0.5 * ((1.0 - attn_redundancy) + (1.0 - attn_conflict)) * 0.5,
            0.0,
            1.0,
        )

        sim_each = torch.clamp(((s @ t) + 1.0) / 2.0, 0.0, 1.0)
        sim_max = torch.max(sim_each) if sim_each.numel() > 0 else torch.tensor(0.0, device=t.device)
        sim_mean = torch.mean(sim_each) if sim_each.numel() > 0 else torch.tensor(0.0, device=t.device)
        alpha = p["skill_task_alpha"]
        skill_task_score = torch.clamp(alpha * sim_max + (1.0 - alpha) * sim_mean, 0.0, 1.0)

        bundle_feat = torch.stack(
            [sim_pool, sim_mean, sim_max, torch.tensor(min(m, 8) / 8.0, dtype=torch.float32, device=t.device)]
        )
        bundle_h = torch.tanh(self.bundle_task_proj(bundle_feat))
        bundle_task_score = torch.sigmoid(self.bundle_task_out(bundle_h)).squeeze(-1)

        sem_w = p["sem"]
        semantic_score = sem_w[0] * skill_task_score + sem_w[1] * attention_score + sem_w[2] * bundle_task_score
        fused = semantic_score

        length_penalty = torch.tensor(0.0, device=t.device)
        if m > 1:
            gain_shortfall = torch.clamp(self.gain_threshold - fused, 0.0, 1.0)
            length_penalty = (m - 1) * self.growth * gain_shortfall

        final = fused - length_penalty - p["risk_lambda"] * float(risk_penalty) - p["token_lambda"] * float(token_penalty)
        stats = {
            "novelty_mean": torch.sum(alpha_i * novelty),
            "redundancy_mean": torch.sum(alpha_i * mean_sim_scaled),
            "conflict_mean": torch.sum(alpha_i * conflict),
            "task_affinity_mean": torch.sum(alpha_i * cos01),
            "attention_score": attention_score,
            "bundle_task_score": bundle_task_score,
        }
        return final, stats


def main() -> None:
    paper_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train internal attention score parameters with listwise+margin.")
    parser.add_argument("--python-note", default="Run with pytorch-gpu env python")
    parser.add_argument("--gold-manifest", default=str(paper_root / "data" / "benchmark_gold_manifest.json"))
    parser.add_argument("--split-json", default=str(paper_root / "data" / "benchmark_train_test_split.json"))
    parser.add_argument("--contrastive-scores-json", default=str(paper_root / "data" / "benchmark_bundle_contrastive_scores_rebalanced.json"))
    parser.add_argument("--task-emb", default=str(paper_root / "data" / "task_embeddings_bigmodel_512_f32_v2.npy"))
    parser.add_argument("--skill-emb", default=str(paper_root / "data" / "bundle_input" / "skills_embedding.npy"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-temp", type=float, default=0.20)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--min-epochs-before-early-stop",
        type=int,
        default=0,
        help="Do not trigger early stop before this epoch (0 means standard early stop).",
    )
    parser.add_argument(
        "--training-mode",
        choices=("joint", "staged"),
        default="joint",
        help="joint: train all scorer parameters from start; staged: use 3-stage schedule.",
    )
    parser.add_argument("--lambda-struct-contrast", type=float, default=0.2)
    parser.add_argument("--branch-alpha", type=float, default=0.5)
    parser.add_argument("--branch-margin", type=float, default=0.03)
    parser.add_argument("--keep-topk", type=int, default=4)
    parser.add_argument("--keep-margin", type=float, default=0.05)
    parser.add_argument("--view-balance-beta", type=float, default=0.10)
    parser.add_argument("--view-balance-tau", type=float, default=0.15)
    parser.add_argument("--stage1-frac", type=float, default=0.50)
    parser.add_argument("--stage2-frac", type=float, default=0.30)
    parser.add_argument("--stage3-lr-scale", type=float, default=0.25)
    parser.add_argument("--attention-pretrain-epochs", type=int, default=20)
    parser.add_argument("--attention-pretrain-lr", type=float, default=5e-3)
    parser.add_argument("--attention-pretrain-margin", type=float, default=0.08)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Linear LR warmup ratio over total epochs (0 disables warmup).",
    )
    parser.add_argument(
        "--min-warmup-epochs",
        type=int,
        default=3,
        help="Minimum warmup epochs when warmup is enabled.",
    )
    parser.add_argument("--attention-selection", action="store_true")
    parser.add_argument("--attn-corr-min", type=float, default=0.0)
    parser.add_argument("--attn-neg-spearman-max", type=float, default=0.40)
    parser.add_argument("--report-json", default=str(paper_root / "output" / "internal_attention_train_report.json"))
    parser.add_argument(
        "--export-scorer-config-json",
        default="",
        help="Optional path to export a stage05-compatible scorer_config json.",
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

    gold_manifest_path = _resolve_under_paper(args.gold_manifest, allow_missing=False)
    split_path = _resolve_under_paper(args.split_json, allow_missing=False)
    contrastive_scores_path = _resolve_under_paper(args.contrastive_scores_json, allow_missing=False)
    task_emb_path = _resolve_under_paper(args.task_emb, allow_missing=False)
    skill_emb_path = _resolve_under_paper(args.skill_emb, allow_missing=False)
    report_path = _resolve_under_paper(args.report_json, allow_missing=True)
    export_cfg_path = (
        _resolve_under_paper(args.export_scorer_config_json, allow_missing=True)
        if args.export_scorer_config_json
        else None
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    gold_manifest = json.loads(gold_manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    contrastive = json.loads(contrastive_scores_path.read_text(encoding="utf-8"))
    task_emb = torch.tensor(np.load(task_emb_path).astype(np.float32), device=device)
    skill_emb = torch.tensor(np.load(skill_emb_path).astype(np.float32), device=device)

    gold_lookup = {int(x["task_index"]): [int(v) for v in x["gold_merged_skill_indices"]] for x in gold_manifest}
    train_set = set(int(x) for x in split["train_task_indices"])
    test_set = set(int(x) for x in split["test_task_indices"])

    by_task_rows: dict[int, list[dict[str, Any]]] = {}
    for row in contrastive.get("rows", []):
        ti = int(row.get("task_index", -1))
        if ti < 0:
            continue
        idxs = [int(x) for x in row.get("indices", [])]
        if not idxs:
            continue
        by_task_rows.setdefault(ti, []).append(row)

    tasks = []
    for ti, rows in sorted(by_task_rows.items()):
        if ti not in gold_lookup:
            continue
        cands = [
            Candidate(
                indices=[int(x) for x in row.get("indices", [])],
                risk_penalty=float(row.get("risk_penalty", row.get("p_red", 0.0))),
                token_penalty=float(row.get("token_penalty", 0.0)),
                base_bundle_score=float(row.get("bundle_score", 0.0)),
            )
            for row in rows
            if row.get("indices")
        ]
        cands = _dedupe_candidates(cands)
        if len(cands) < 2:
            continue
        gold_set = set(gold_lookup[ti])
        sup_items = []
        for c in cands:
            q = _bundle_quality(c.indices, gold_set, len(gold_set))
            sup_items.append((q, c))
        sup_items.sort(key=lambda x: x[0], reverse=True)
        final_supervision_pool = [c for q, c in sup_items if q > 0.0][:12]
        if not final_supervision_pool:
            final_supervision_pool = [c for _q, c in sup_items[: min(4, len(sup_items))]]
        tasks.append(
            {
                "task_index": ti,
                "candidate_pool": cands,
                "final_supervision_pool": final_supervision_pool,
            }
        )

    for row in tasks:
        base_pool = list(row.get("candidate_pool", []))
        merged_pool = _dedupe_candidates(base_pool + list(row.get("final_supervision_pool", [])))
        ti = int(row["task_index"])
        gold_set = set(gold_lookup[ti])
        row["candidate_pool_groups"] = _prepare_candidate_groups(
            base_pool,
            skill_device=skill_emb.device,
            score_device=device,
        )
        row["merged_candidate_pool"] = merged_pool
        row["merged_candidate_groups"] = _prepare_candidate_groups(
            merged_pool,
            skill_device=skill_emb.device,
            score_device=device,
        )
        row["merged_candidate_util"] = [
            _bundle_preference_utility(
                c.indices,
                gold_set,
                len(gold_set),
                c.risk_penalty,
                c.token_penalty,
            )
            for c in merged_pool
        ]
        row["merged_candidate_control_costs"] = [
            max(len(c.indices) - 4, 0) / 4.0 + float(c.risk_penalty) + float(c.token_penalty)
            for c in merged_pool
        ]
        row["merged_candidate_target"] = torch.tensor(
            softmax_np(np.asarray(row["merged_candidate_util"], dtype=np.float64), temp=float(args.list_temp)),
            dtype=torch.float32,
            device=device,
        )
        row["merged_candidate_control_costs_t"] = torch.tensor(
            row["merged_candidate_control_costs"],
            dtype=torch.float32,
            device=device,
        )

    model = LearnableScore(
        dim=int(task_emb.shape[1]),
        num_heads=4,
    ).to(device)
    named_params = list(model.named_parameters())
    stage1_prefixes = (
        "tau_raw",
        "rel_proj",
        "rel_out",
        "bundle_task_proj",
        "bundle_task_out",
        "complement_low_raw",
        "redundancy_threshold_raw",
    )
    stage2_prefixes = ("semantic_logits",)
    stage3_prefixes = tuple(name for name, _ in named_params)
    stage1_epochs = max(1, int(round(args.epochs * args.stage1_frac)))
    stage2_epochs = max(1, int(round(args.epochs * args.stage2_frac)))

    def stage_spec(epoch_index: int) -> tuple[str, tuple[str, ...], float]:
        if args.training_mode == "joint":
            return ("joint", stage3_prefixes, float(args.lr))
        if epoch_index < stage1_epochs:
            return ("stage1_branch", stage1_prefixes, float(args.lr))
        if epoch_index < stage1_epochs + stage2_epochs:
            return ("stage2_fusion", stage2_prefixes, float(args.lr))
        return ("stage3_joint", stage3_prefixes, float(args.lr) * float(args.stage3_lr_scale))

    current_stage_name = ""
    current_stage_lr = None
    opt = None

    structural_params: list[torch.nn.Parameter] = []
    contrastive_pairs_all = _build_contrastive_pairs(contrastive.get("rows", []))
    # Strict protocol: only train-split contrastive pairs are allowed to
    # contribute to optimization. Test tasks are evaluation-only.
    contrastive_pairs_by_task = {
        int(ti): pairs
        for ti, pairs in contrastive_pairs_all.items()
        if int(ti) in train_set
    }
    contrastive_pair_count_train = int(sum(len(v) for v in contrastive_pairs_by_task.values()))
    contrastive_pair_count_test = int(
        sum(len(v) for ti, v in contrastive_pairs_all.items() if int(ti) in test_set)
    )

    train_tasks = [t for t in tasks if int(t["task_index"]) in train_set]
    for row in train_tasks:
        ti = int(row["task_index"])
        attention_pairs = _filter_attention_pretrain_pairs(contrastive_pairs_by_task.get(ti, []))
        row["attention_pretrain_groups"] = _prepare_contrastive_groups(
            attention_pairs,
            skill_device=skill_emb.device,
            score_device=device,
        )
        row["contrastive_groups"] = _prepare_contrastive_groups(
            contrastive_pairs_by_task.get(ti, []),
            skill_device=skill_emb.device,
            score_device=device,
        )
    weak_sample_count: dict[int, int] = {}
    best_epoch_loss = float("inf")
    best_state_dict = None
    stale_epochs = 0
    best_attention_diag: dict[str, float] | None = None

    if int(args.attention_pretrain_epochs) > 0:
        pretrain_prefixes = (
            "tau_raw",
            "rel_proj",
            "rel_out",
            "complement_low_raw",
            "redundancy_threshold_raw",
        )
        _set_requires_grad(named_params, pretrain_prefixes)
        opt = _build_optimizer(named_params, lr=float(args.attention_pretrain_lr), weight_decay=args.weight_decay)
        model.train()
        for epoch in range(int(args.attention_pretrain_epochs)):
            random.shuffle(train_tasks)
            total_pre = 0.0
            steps = 0
            for row in train_tasks:
                tvec = task_emb[int(row["task_index"])]
                pair_groups = row.get("attention_pretrain_groups", [])
                if not pair_groups:
                    continue
                loss_terms = []
                for group in pair_groups:
                    pos_vecs = skill_emb[group["pos_idx_mat"]]
                    neg_vecs = skill_emb[group["neg_idx_mat"]]
                    _pos_final, pos_stats = model.score_bundle_with_stats_batch_same_size(
                        tvec, pos_vecs, group["pos_risk_batch"], group["pos_token_batch"]
                    )
                    _neg_final, neg_stats = model.score_bundle_with_stats_batch_same_size(
                        tvec, neg_vecs, group["neg_risk_batch"], group["neg_token_batch"]
                    )
                    attn_margin = torch.full_like(group["margin_batch"], float(args.attention_pretrain_margin))
                    pair_loss = F.relu(attn_margin - pos_stats["attention_score"] + neg_stats["attention_score"])
                    loss_terms.extend((group["weight_batch"] * pair_loss).unbind())
                if not loss_terms:
                    continue
                loss = torch.mean(torch.stack(loss_terms))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_pre += float(loss.detach().cpu())
                steps += 1
            if (epoch + 1) % 5 == 0 or epoch + 1 == int(args.attention_pretrain_epochs):
                denom = max(steps, 1)
                print(f"attention_pretrain epoch={epoch+1} loss={total_pre / denom:.6f}")

    def _attention_diag(rows: list[dict[str, Any]]) -> dict[str, float]:
        attn_vals: list[float] = []
        util_vals: list[float] = []
        negative_sp = 0
        task_count = 0
        model.eval()
        with torch.no_grad():
            for task_row in rows:
                ti = int(task_row["task_index"])
                gset = set(gold_lookup[ti])
                gsize = len(gset)
                tvec_local = task_emb[ti]
                pool_local = task_row.get("candidate_pool", [])
                pool_groups = task_row.get("candidate_pool_groups", [])
                if len(pool_local) < 2:
                    continue
                task_count += 1
                task_attn: list[float] = []
                task_util: list[float] = []
                attn_scores: list[float | None] = [None] * len(pool_local)
                for group in pool_groups:
                    svecs_batch = skill_emb[group["idx_mat"]]
                    _score, stats = model.score_bundle_with_stats_batch_same_size(
                        tvec_local,
                        svecs_batch,
                        group["risk_batch"],
                        group["token_batch"],
                    )
                    for kk, pool_idx in enumerate(group["pool_indices"]):
                        attn_scores[pool_idx] = float(stats["attention_score"][kk].detach().cpu())
                for i, c in enumerate(pool_local):
                    attn = float(attn_scores[i]) if attn_scores[i] is not None else 0.0
                    util = _bundle_preference_utility(c.indices, gset, gsize, c.risk_penalty, c.token_penalty)
                    attn_vals.append(attn)
                    util_vals.append(float(util))
                    task_attn.append(attn)
                    task_util.append(float(util))
                if _spearman_corr(task_attn, task_util) < 0.0:
                    negative_sp += 1
        model.train()
        neg_ratio = float(negative_sp / max(task_count, 1))
        return {
            "attention_pearson_vs_utility": float(_pearson_corr(attn_vals, util_vals)),
            "attention_negative_spearman_task_ratio": neg_ratio,
            "attention_diag_task_count": float(task_count),
        }
    for epoch in range(args.epochs):
        stage_name, enabled_prefixes, stage_lr = stage_spec(epoch)
        if stage_name != current_stage_name or stage_lr != current_stage_lr:
            _set_requires_grad(named_params, enabled_prefixes)
            opt = _build_optimizer(named_params, lr=stage_lr, weight_decay=args.weight_decay)
            current_stage_name = stage_name
            current_stage_lr = stage_lr
            print(f"switch_stage epoch={epoch+1} stage={stage_name} lr={stage_lr:.6f}")
        warmup_epochs = 0
        if float(args.warmup_ratio) > 0.0:
            warmup_epochs = max(int(args.min_warmup_epochs), int(round(float(args.epochs) * float(args.warmup_ratio))))
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_scale = float(epoch + 1) / float(max(warmup_epochs, 1))
        else:
            warmup_scale = 1.0
        effective_lr = float(stage_lr) * float(warmup_scale)
        if opt is not None:
            for g in opt.param_groups:
                g["lr"] = effective_lr
        random.shuffle(train_tasks)
        total_loss = 0.0
        total_list_loss = 0.0
        total_contrastive_loss = 0.0
        total_branch_loss = 0.0
        total_control_loss = 0.0
        total_struct_contrast = 0.0
        total_keep_topk = 0.0
        total_view_balance = 0.0
        model.train()
        for row in train_tasks:
            ti = int(row["task_index"])
            tvec = task_emb[ti]
            gold_indices = gold_lookup[ti]

            candidate_pool = row.get("merged_candidate_pool", [])
            candidate_groups = row.get("merged_candidate_groups", [])
            list_loss = torch.tensor(0.0, device=tvec.device)
            contrastive_loss = torch.tensor(0.0, device=tvec.device)
            branch_loss = torch.tensor(0.0, device=tvec.device)
            control_loss = torch.tensor(0.0, device=tvec.device)
            logits_t: torch.Tensor | None = None
            if len(candidate_pool) >= 2:
                logits_t = _score_candidate_groups(model, tvec, skill_emb, candidate_pool, candidate_groups)
                target = row["merged_candidate_target"]
                log_probs = F.log_softmax(logits_t, dim=0)
                list_loss = F.kl_div(log_probs, target, reduction="batchmean", log_target=False)
                control_costs_t = row["merged_candidate_control_costs_t"]
                control_probs = F.softmax(logits_t, dim=0)
                control_loss = torch.sum(control_probs * control_costs_t)

                # Disabled by design for simplified objective: keep_topk loss removed.
                keep_topk_loss = torch.tensor(0.0, device=tvec.device)
            else:
                keep_topk_loss = torch.tensor(0.0, device=tvec.device)

            pair_terms = []
            branch_terms = []
            for group in row.get("contrastive_groups", []):
                pos_vecs = skill_emb[group["pos_idx_mat"]]
                neg_vecs = skill_emb[group["neg_idx_mat"]]
                pos_score, pos_stats = model.score_bundle_with_stats_batch_same_size(
                    tvec, pos_vecs, group["pos_risk_batch"], group["pos_token_batch"]
                )
                neg_score, neg_stats = model.score_bundle_with_stats_batch_same_size(
                    tvec, neg_vecs, group["neg_risk_batch"], group["neg_token_batch"]
                )
                pair_loss = F.relu(group["margin_batch"] - pos_score + neg_score)
                if group["use_struct_contrast"]:
                    struct_margin = 0.05
                    novelty_gap = F.relu(struct_margin - pos_stats["novelty_mean"] + neg_stats["novelty_mean"])
                    redundancy_gap = F.relu(struct_margin - neg_stats["redundancy_mean"] + pos_stats["redundancy_mean"])
                    conflict_gap = F.relu(struct_margin - neg_stats["conflict_mean"] + pos_stats["conflict_mean"])
                    pair_loss = pair_loss + float(args.lambda_struct_contrast) * (
                        novelty_gap + redundancy_gap + conflict_gap
                    )
                    total_struct_contrast += float(torch.mean(novelty_gap + redundancy_gap + conflict_gap).detach().cpu())
                pair_terms.extend((group["weight_batch"] * pair_loss).unbind())
                # Disabled by design for simplified objective: branch consistency loss removed.
            if pair_terms:
                contrastive_loss = torch.mean(torch.stack(pair_terms))
            if branch_terms:
                branch_loss = torch.mean(torch.stack(branch_terms))

            # Prevent three-view collapse: each semantic branch keeps at least tau share.
            p_now = model._params_constrained()
            sem_w = p_now["sem"]
            tau = float(args.view_balance_tau)
            view_balance_loss = (
                F.relu(torch.tensor(tau, device=sem_w.device) - sem_w[0])
                + F.relu(torch.tensor(tau, device=sem_w.device) - sem_w[1])
                + F.relu(torch.tensor(tau, device=sem_w.device) - sem_w[2])
            )
            loss = (
                list_loss
                + contrastive_loss
                + float(args.view_balance_beta) * view_balance_loss
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())
            total_list_loss += float(list_loss.detach().cpu())
            total_contrastive_loss += float(contrastive_loss.detach().cpu())
            total_branch_loss += float(branch_loss.detach().cpu())
            total_control_loss += float(control_loss.detach().cpu())
            total_view_balance += float(view_balance_loss.detach().cpu())

        if (epoch + 1) % 10 == 0:
            denom = max(len(train_tasks), 1)
            print(
                "epoch="
                f"{epoch+1} "
                f"stage={current_stage_name} "
                f"loss={total_loss / denom:.6f} "
                f"list={total_list_loss / denom:.6f} "
                f"contrast={total_contrastive_loss / denom:.6f} "
                f"branch={total_branch_loss / denom:.6f} "
                f"keep={total_keep_topk / denom:.6f} "
                f"control={total_control_loss / denom:.6f} "
                f"struct_contrast={total_struct_contrast / denom:.6f} "
                f"view_balance={total_view_balance / denom:.6f}"
            )

        denom = max(len(train_tasks), 1)
        epoch_loss = total_loss / denom
        attention_diag = _attention_diag(train_tasks) if args.attention_selection else None
        pass_attention_gate = True
        if attention_diag is not None:
            pass_attention_gate = (
                float(attention_diag["attention_pearson_vs_utility"]) > float(args.attn_corr_min)
                and float(attention_diag["attention_negative_spearman_task_ratio"]) <= float(args.attn_neg_spearman_max)
            )
        if pass_attention_gate and epoch_loss < best_epoch_loss - float(args.early_stop_min_delta):
            best_epoch_loss = epoch_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_attention_diag = attention_diag
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.early_stop_patience):
                print(
                    f"early_stop epoch={epoch+1} "
                    f"best_loss={best_epoch_loss:.6f} "
                    f"patience={args.early_stop_patience}"
                )
                break

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    def rerank_top1(split_indices: set[int], learned: bool) -> dict[str, float]:
        rows = []
        model.eval()
        with torch.no_grad():
            for row in tasks:
                ti = int(row["task_index"])
                if ti not in split_indices:
                    continue
                pool = row.get("candidate_pool", [])
                if not pool:
                    continue
                if learned:
                    tvec = task_emb[ti]
                    groups = row.get("candidate_pool_groups", [])
                    scores_t = _score_candidate_groups(model, tvec, skill_emb, pool, groups)
                    best_idx = int(torch.argmax(scores_t).item())
                else:
                    base = [c.base_bundle_score for c in pool]
                    best_idx = int(np.argmax(base))
                pred = pool[best_idx].indices
                rows.append(set_metrics(pred, gold_lookup[ti]))
        return avg(rows)

    train_before = rerank_top1(train_set, learned=False)
    train_after = rerank_top1(train_set, learned=True)
    test_before = rerank_top1(test_set, learned=False)
    test_after = rerank_top1(test_set, learned=True)

    with torch.no_grad():
        p = model._params_constrained()
        report = {
            "trainable_parameters": [
                "tau",
                "skill_task_alpha",
                "complement_low",
                "redundancy_threshold",
                "semantic_task_weight/semantic_attention_weight/semantic_bundle_weight",
                "relation_feature_mlp",
                "bundle_task_mlp",
            ],
            "config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "margin": args.margin,
                "list_temp": args.list_temp,
                "seed": args.seed,
                "device": str(device),
                "early_stop_patience": int(args.early_stop_patience),
                "early_stop_min_delta": float(args.early_stop_min_delta),
                "lambda_struct_contrast": float(args.lambda_struct_contrast),
                "branch_alpha": float(args.branch_alpha),
                "branch_margin": float(args.branch_margin),
                "keep_topk": int(args.keep_topk),
                "keep_margin": float(args.keep_margin),
                "view_balance_beta": float(args.view_balance_beta),
                "view_balance_tau": float(args.view_balance_tau),
                "stage1_frac": float(args.stage1_frac),
                "stage2_frac": float(args.stage2_frac),
                "stage3_lr_scale": float(args.stage3_lr_scale),
                "warmup_ratio": float(args.warmup_ratio),
                "min_warmup_epochs": int(args.min_warmup_epochs),
                "best_epoch_loss": float(best_epoch_loss),
                "loss_terms": [
                    "preference_listwise",
                    "topk_keep_surrogate",
                    "hard_negative_contrastive",
                    "branch_consistency(attention+bundle_task)",
                    "no_train_time_control",
                ],
            },
            "weak_supervision": {
                "weak_task_count": 0,
                "weak_sample_total": 0,
                "weak_sample_top10": [],
                "description": "No weak-beam data is used in the current paper pipeline.",
            },
            "contrastive_supervision": {
                "train_task_count": int(len(contrastive_pairs_by_task)),
                "train_pair_count": contrastive_pair_count_train,
                "test_pair_count_ignored": contrastive_pair_count_test,
                "policy": "train split pairs only; test pairs are ignored during optimization",
            },
            "learned_scalars": {
                "semantic_triplet": [float(x) for x in p["sem"].cpu().tolist()],
                "skill_task_alpha": float(p["skill_task_alpha"].cpu()),
                "complement_low": float(p["complement_low"].cpu()),
                "redundancy_threshold": float(p["redundancy_threshold"].cpu()),
                "risk_lambda": float(p["risk_lambda"].cpu()),
                "token_lambda": float(p["token_lambda"].cpu()),
                "gain_threshold": float(model.gain_threshold),
                "growth": float(model.growth),
                "tau": [float(x) for x in p["tau"].cpu().tolist()],
                "bundle_task_mlp": {
                    "w1": model.bundle_task_proj.weight.detach().cpu().tolist(),
                    "b1": model.bundle_task_proj.bias.detach().cpu().tolist(),
                    "w2": model.bundle_task_out.weight.detach().cpu().view(-1).tolist(),
                    "b2": float(model.bundle_task_out.bias.detach().cpu().view(-1)[0]),
                },
                "relation_feature_mlp": {
                    "w1": model.rel_proj.weight.detach().cpu().tolist(),
                    "b1": model.rel_proj.bias.detach().cpu().tolist(),
                    "w2": model.rel_out.weight.detach().cpu().view(-1).tolist(),
                    "b2": float(model.rel_out.bias.detach().cpu().view(-1)[0]),
                },
            },
            "metrics": {
                "train_before": train_before,
                "train_after": train_after,
                "test_before": test_before,
                "test_after": test_after,
            },
            "attention_selection": {
                "enabled": bool(args.attention_selection),
                "attn_corr_min": float(args.attn_corr_min),
                "attn_neg_spearman_max": float(args.attn_neg_spearman_max),
                "best_attention_diag": best_attention_diag,
            },
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if export_cfg_path is not None:
        sem = report["learned_scalars"]["semantic_triplet"]
        tau = report["learned_scalars"]["tau"]
        export_cfg = {
            "semantic_task_weight": float(sem[0]),
            "semantic_attention_weight": float(sem[1]),
            "semantic_bundle_weight": float(sem[2]),
            "skill_task_alpha": float(report["learned_scalars"]["skill_task_alpha"]),
            "complement_low": float(report["learned_scalars"]["complement_low"]),
            "redundancy_threshold": float(report["learned_scalars"]["redundancy_threshold"]),
            "risk_lambda": float(report["learned_scalars"]["risk_lambda"]),
            "token_lambda": float(report["learned_scalars"]["token_lambda"]),
            "gain_threshold": float(report["learned_scalars"]["gain_threshold"]),
            "growth": float(report["learned_scalars"]["growth"]),
            "bundle_task_mlp_w1": report["learned_scalars"]["bundle_task_mlp"]["w1"],
            "bundle_task_mlp_b1": report["learned_scalars"]["bundle_task_mlp"]["b1"],
            "bundle_task_mlp_w2": report["learned_scalars"]["bundle_task_mlp"]["w2"],
            "bundle_task_mlp_b2": float(report["learned_scalars"]["bundle_task_mlp"]["b2"]),
            "relation_feature_mlp_w1": report["learned_scalars"]["relation_feature_mlp"]["w1"],
            "relation_feature_mlp_b1": report["learned_scalars"]["relation_feature_mlp"]["b1"],
            "relation_feature_mlp_w2": report["learned_scalars"]["relation_feature_mlp"]["w2"],
            "relation_feature_mlp_b2": float(report["learned_scalars"]["relation_feature_mlp"]["b2"]),
            # stage05 supports scalar attention temperatures;
            # export mean of learned head-wise tau for compatibility.
            "semantic_attention_tau_match": float(sum(tau) / max(len(tau), 1)),
            "semantic_attention_tau_self": float(sum(tau) / max(len(tau), 1)),
        }
        export_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        export_cfg_path.write_text(json.dumps(export_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved stage05 scorer config: {export_cfg_path}")

        def rerank_top1_stage05(split_indices: set[int], scorer_cfg: dict[str, float], learned: bool) -> dict[str, float]:
            rows = []
            for row in tasks:
                ti = int(row["task_index"])
                if ti not in split_indices:
                    continue
                pool = row.get("candidate_pool", [])
                if not pool:
                    continue
                if learned:
                    scores = []
                    tvec = task_emb[ti].detach().cpu().numpy()
                    for c in pool:
                        svecs = skill_emb[torch.tensor(c.indices, dtype=torch.long)].detach().cpu().numpy()
                        result = stage05_score_bundle(
                            task_vec=tvec,
                            skill_vecs=svecs,
                            scorer_config=scorer_cfg,
                        )
                        scores.append(float(result.final_score))
                    best_idx = int(np.argmax(scores))
                else:
                    base = [c.base_bundle_score for c in pool]
                    best_idx = int(np.argmax(base))
                pred = pool[best_idx].indices
                rows.append(set_metrics(pred, gold_lookup[ti]))
            return avg(rows)

        report["stage05_export_metrics"] = {
            "train_before": rerank_top1_stage05(train_set, export_cfg, learned=False),
            "train_after": rerank_top1_stage05(train_set, export_cfg, learned=True),
            "test_before": rerank_top1_stage05(test_set, export_cfg, learned=False),
            "test_after": rerank_top1_stage05(test_set, export_cfg, learned=True),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
