#!/usr/bin/env python3
"""
Unified bundle scorer shared by beam search and contrastive training.

Current mainline scoring uses three semantic views plus penalties:
1. skill-task view: max/mean skill-to-task similarity
2. skill-bundle view: structure-aware attention plus PCA/cluster prior
3. bundle-task view: lightweight MLP over pooled bundle/task features
Then apply length / risk / token penalties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l2norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.clip(n, eps, None)
    return x / n


def l2norm_vec(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / max(n, eps)


def clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


@dataclass
class BundleScoreResult:
    semantic_vec: np.ndarray
    relation_vec: np.ndarray
    fused_vec: np.ndarray
    attention_weights: np.ndarray
    head_vectors: np.ndarray
    head_item_weights: np.ndarray
    head_gate_weights: np.ndarray
    head_scores: np.ndarray
    alpha: float
    p_syn: float
    p_red: float
    skill_task_score: float
    bundle_task_score: float
    task_bundle_score: float
    attention_score: float
    relation_align_score: float
    semantic_task_weight: float
    semantic_attention_weight: float
    semantic_bundle_weight: float
    fusion_semantic_weight: float
    fusion_structural_weight: float
    raw_fused_score: float
    length_penalty: float
    semantic_score: float
    structural_score: float
    risk_penalty: float
    token_penalty: float
    final_score: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.clip(np.sum(e), 1e-12, None)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _bundle_task_mlp_score(
    sim_pool_01: float,
    sim_absdiff_01: float,
    sim_prod_01: float,
    sim_l2_01: float,
    scorer_config: dict[str, float] | None,
) -> float:
    if not scorer_config:
        return float(sim_pool_01)
    w1 = scorer_config.get("bundle_task_mlp_w1")
    b1 = scorer_config.get("bundle_task_mlp_b1")
    w2 = scorer_config.get("bundle_task_mlp_w2")
    b2 = scorer_config.get("bundle_task_mlp_b2")
    if w1 is None or b1 is None or w2 is None or b2 is None:
        return float(sim_pool_01)
    x = np.asarray(
        [
            float(sim_pool_01),
            float(sim_absdiff_01),
            float(sim_prod_01),
            float(sim_l2_01),
        ],
        dtype=np.float32,
    )
    h = np.tanh(np.asarray(w1, dtype=np.float32) @ x + np.asarray(b1, dtype=np.float32))
    y = float(np.asarray(w2, dtype=np.float32) @ h + float(b2))
    return clip01(float(_sigmoid(y)))


def _attention_summary_mlp_score(
    attn_task_aff_01: float,
    attn_novelty_01: float,
    attn_redundancy_01: float,
    attn_conflict_01: float,
    scorer_config: dict[str, float] | None,
) -> float:
    if not scorer_config:
        return clip01(0.5 * (attn_task_aff_01 + attn_novelty_01) + 0.5 * ((1.0 - attn_redundancy_01) + (1.0 - attn_conflict_01)) * 0.5)
    w1 = scorer_config.get("attention_summary_mlp_w1")
    b1 = scorer_config.get("attention_summary_mlp_b1")
    w2 = scorer_config.get("attention_summary_mlp_w2")
    b2 = scorer_config.get("attention_summary_mlp_b2")
    if w1 is None or b1 is None or w2 is None or b2 is None:
        return clip01(0.5 * (attn_task_aff_01 + attn_novelty_01) + 0.5 * ((1.0 - attn_redundancy_01) + (1.0 - attn_conflict_01)) * 0.5)
    x = np.asarray(
        [
            float(attn_task_aff_01),
            float(attn_novelty_01),
            float(attn_redundancy_01),
            float(attn_conflict_01),
        ],
        dtype=np.float32,
    )
    h = np.tanh(np.asarray(w1, dtype=np.float32) @ x + np.asarray(b1, dtype=np.float32))
    y = float(np.asarray(w2, dtype=np.float32) @ h + float(b2))
    return clip01(float(_sigmoid(y)))


def _relation_logits_score(
    rel_feat: np.ndarray,
    scorer_config: dict[str, float] | None,
    fallback_logits: np.ndarray,
) -> np.ndarray:
    feat_mean = np.mean(rel_feat, axis=0, keepdims=True)
    feat_std = np.std(rel_feat, axis=0, keepdims=True)
    rel_feat_norm = (rel_feat - feat_mean) / np.clip(feat_std, 1e-4, None)
    fb_mean = np.mean(fallback_logits, keepdims=True)
    fb_std = np.std(fallback_logits, keepdims=True)
    fallback_norm = (fallback_logits - fb_mean) / np.clip(fb_std, 1e-4, None)
    if not scorer_config:
        return fallback_norm.astype(np.float32)
    w1 = scorer_config.get("relation_feature_mlp_w1")
    b1 = scorer_config.get("relation_feature_mlp_b1")
    w2 = scorer_config.get("relation_feature_mlp_w2")
    b2 = scorer_config.get("relation_feature_mlp_b2")
    if w1 is None or b1 is None or w2 is None or b2 is None:
        return fallback_norm.astype(np.float32)
    h = np.tanh(rel_feat_norm @ np.asarray(w1, dtype=np.float32).T + np.asarray(b1, dtype=np.float32))
    y = h @ np.asarray(w2, dtype=np.float32).reshape(-1, 1) + float(b2)
    return (y.reshape(-1) + 0.5 * fallback_norm).astype(np.float32)


# Cache of orthogonal projection matrices keyed by (d, num_heads).
_ORTHO_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _get_ortho_projections(d: int, num_heads: int) -> np.ndarray:
    """
    Return a [num_heads, d, d] array of deterministic orthogonal matrices.

    Built once per (d, num_heads) via QR decomposition of fixed random
    matrices (seed=42+h).  Each projection rotates the task vector into a
    genuinely different subspace so heads produce diverse attention patterns.
    """
    key = (d, num_heads)
    if key not in _ORTHO_CACHE:
        projs = []
        for h in range(num_heads):
            rng = np.random.default_rng(42 + h)
            A = rng.standard_normal((d, d)).astype(np.float32)
            Q, _ = np.linalg.qr(A)
            projs.append(Q.astype(np.float32))
        _ORTHO_CACHE[key] = np.stack(projs, axis=0)  # [h, d, d]
    return _ORTHO_CACHE[key]


def _build_head_queries(task_vec: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Create deterministic head-specific task queries via orthogonal projection.

    Each head applies a fixed random orthogonal matrix to task_vec so the
    resulting queries are genuinely diverse in embedding space 鈥?unlike
    cyclic roll which produces near-identical vectors in high dimensions.
    """
    d = task_vec.shape[0]
    projs = _get_ortho_projections(d, num_heads)  # [h, d, d]
    queries = []
    for h in range(num_heads):
        q = l2norm_vec(projs[h] @ task_vec)
        queries.append(q.astype(np.float32))
    return np.asarray(queries, dtype=np.float32)


def _multihead_semantic_bundle(
    task_vec: np.ndarray,
    skill_vecs: np.ndarray,
    num_heads: int = 4,
    tau_match: float = 0.20,
    tau_self: float = 0.25,
    complement_low: float = 0.35,
    redundancy_threshold: float = 0.85,
    scorer_config: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-head task-conditioned bundle encoder.
    Query starts from the task embedding, and each head uses a deterministic
    transformed view of the task so attention patterns can differ by head.
    Temperature scales with bundle size so attention stays meaningful as
    bundle grows 鈥?prevents collapse to argmax for large bundles.
    """
    m = skill_vecs.shape[0]
    tau_match_eff = max(0.12, float(tau_match))
    tau_self_eff = max(0.12, float(tau_self))
    head_queries = _build_head_queries(task_vec, num_heads=num_heads)  # [h,d]
    item_weights = []
    head_vectors = []
    head_scores = []

    for h in range(num_heads):
        q = head_queries[h]  # [d]
        # Skill self-attention (contextualize each skill by other skills).
        self_logits = (skill_vecs @ skill_vecs.T) / tau_self_eff  # [m,m]
        self_logits = self_logits - np.max(self_logits, axis=1, keepdims=True)
        self_exp = np.exp(self_logits)
        self_attn = self_exp / np.clip(np.sum(self_exp, axis=1, keepdims=True), 1e-12, None)  # [m,m]
        contextual = self_attn @ skill_vecs  # [m,d]
        contextual = l2norm_rows(contextual.astype(np.float32, copy=False))

        # Structure-aware task->skill attention:
        # base relevance + novelty bonus - redundancy penalty - conflict penalty.
        match_logits = (contextual @ q) / tau_match_eff  # [m]
        task_aff = np.clip(((contextual @ q) + 1.0) / 2.0, 0.0, 1.0)
        if m > 1:
            sim_ctx = contextual @ contextual.T
            sim_ctx_01 = np.clip((sim_ctx + 1.0) / 2.0, 0.0, 1.0)
            np.fill_diagonal(sim_ctx_01, 0.0)
            mean_sim = np.mean(sim_ctx_01, axis=1)
            max_sim = np.max(sim_ctx_01, axis=1)
        else:
            mean_sim = np.zeros((m,), dtype=np.float32)
            max_sim = np.zeros((m,), dtype=np.float32)
        red_thr = max(float(redundancy_threshold), 1e-6)
        mean_sim_scaled = np.clip(mean_sim / red_thr, 0.0, 1.0)
        comp_low = np.clip(float(complement_low), 1e-6, 1.0)
        task_low = np.clip((comp_low - task_aff) / comp_low, 0.0, 1.0)
        novelty = task_aff * (1.0 - mean_sim_scaled)
        conflict = task_low * mean_sim_scaled
        # Second-view fallback without hand-tuned scalar mixing:
        # if relation MLP weights are absent, use pure task-match logits.
        fallback_relation_logits = match_logits
        rel_feat = np.stack(
            [task_aff, novelty, mean_sim_scaled, conflict],
            axis=1,
        ).astype(np.float32)
        relation_logits = _relation_logits_score(rel_feat, scorer_config, fallback_relation_logits)
        w = _softmax(relation_logits).astype(np.float32)
        z = l2norm_vec((w[:, None] * contextual).sum(axis=0))
        # Second-view head score is defined by weighted relation statistics,
        # aligned with trainer-side attention definition.
        attn_task_aff_01 = float(np.sum(w * task_aff))
        attn_novelty_01 = float(np.sum(w * novelty))
        attn_redundancy_01 = float(np.sum(w * mean_sim_scaled))
        attn_conflict_01 = float(np.sum(w * conflict))
        s = _attention_summary_mlp_score(
            attn_task_aff_01=attn_task_aff_01,
            attn_novelty_01=attn_novelty_01,
            attn_redundancy_01=attn_redundancy_01,
            attn_conflict_01=attn_conflict_01,
            scorer_config=scorer_config,
        )
        item_weights.append(w)
        head_vectors.append(z.astype(np.float32))
        head_scores.append(float(s))

    head_item_weights = np.asarray(item_weights, dtype=np.float32)    # [h,m]
    head_vectors_arr = np.asarray(head_vectors, dtype=np.float32)     # [h,d]
    head_scores_arr = np.asarray(head_scores, dtype=np.float32)       # [h]

    # Task-conditioned gating over heads: prefer heads whose extracted view is
    # more aligned with the current task state.
    gate_logits = head_scores_arr.astype(np.float32)
    head_gate_weights = _softmax(gate_logits).astype(np.float32)      # [h]

    attention_weights = (head_gate_weights[:, None] * head_item_weights).sum(axis=0)
    attention_weights = attention_weights / np.clip(np.sum(attention_weights), 1e-12, None)
    # Diagnostic semantic vector: aggregate original skill embeddings with final
    # attention weights. This vector is no longer used to define attention score.
    semantic_vec = l2norm_vec((attention_weights[:, None] * skill_vecs).sum(axis=0))

    return (
        semantic_vec.astype(np.float32),
        attention_weights.astype(np.float32),
        head_vectors_arr.astype(np.float32),
        head_item_weights.astype(np.float32),
        head_gate_weights.astype(np.float32),
        (head_scores_arr * 100.0).astype(np.float32),
    )


def _length_penalty(
    bundle_size: int,
    raw_score_01: float,
    gain_threshold: float = 0.62,
    growth: float = 0.008,
) -> float:
    """Penalize only when score gain is insufficient.

    No hard target bundle size is used.
    """
    if bundle_size <= 1:
        return 0.0
    shortfall = max(0.0, gain_threshold - raw_score_01)
    return float((bundle_size - 1) * growth * shortfall)


def _normalize_fusion_weights(
    semantic_weight: float,
    structural_weight: float,
    min_share: float,
) -> tuple[float, float]:
    sem = max(float(semantic_weight), 0.0)
    st = max(float(structural_weight), 0.0)
    if sem <= 0.0 and st <= 0.0:
        return 1.0, 0.0
    total = max(sem + st, 1e-12)
    sem = sem / total
    st = st / total
    # Keep both branches observable when both are enabled.
    share_floor = clip01(float(min_share))
    share_floor = min(share_floor, 0.499)
    if share_floor > 0.0 and sem > 0.0 and st > 0.0:
        sem = max(sem, share_floor)
        st = max(st, share_floor)
        renorm = max(sem + st, 1e-12)
        sem /= renorm
        st /= renorm
    return float(sem), float(st)


def score_bundle(
    task_vec: np.ndarray,
    skill_vecs: np.ndarray,
    semantic_weight: float = 0.55,
    structural_weight: float = 0.45,
    complement_low: float = 0.35,
    redundancy_threshold: float = 0.85,
    token_cost_values: np.ndarray | None = None,
    permission_risk_values: np.ndarray | None = None,
    risk_lambda: float = 0.01,
    token_lambda: float = 0.005,
    structural_neutral: float = 0.50,
    structural_bonus_scale: float = 0.35,
    min_share: float = 0.15,
    num_heads: int = 4,
    scorer_config: dict[str, float] | None = None,
) -> BundleScoreResult:
    gain_threshold = 0.62
    growth = 0.008
    if scorer_config:
        if "complement_low" in scorer_config:
            complement_low = float(scorer_config["complement_low"])
        if "redundancy_threshold" in scorer_config:
            redundancy_threshold = float(scorer_config["redundancy_threshold"])
        if "risk_lambda" in scorer_config:
            risk_lambda = float(scorer_config["risk_lambda"])
        if "token_lambda" in scorer_config:
            token_lambda = float(scorer_config["token_lambda"])
        if "min_share" in scorer_config:
            min_share = float(scorer_config["min_share"])
        if "semantic_weight" in scorer_config:
            semantic_weight = float(scorer_config["semantic_weight"])
        if "structural_weight" in scorer_config:
            structural_weight = float(scorer_config["structural_weight"])
        if "structural_bonus_scale" in scorer_config:
            structural_bonus_scale = float(scorer_config["structural_bonus_scale"])
        if "structural_neutral" in scorer_config:
            structural_neutral = float(scorer_config["structural_neutral"])
        if "gain_threshold" in scorer_config:
            gain_threshold = float(scorer_config["gain_threshold"])
        if "growth" in scorer_config:
            growth = float(scorer_config["growth"])

    task_vec = l2norm_vec(task_vec.astype(np.float32, copy=False))
    skill_vecs = l2norm_rows(skill_vecs.astype(np.float32, copy=False))
    bundle_size = skill_vecs.shape[0]

    # Single-skill fast path:
    # Only use the first view (skill-task). Skip attention and bundle-task views
    # to make seed selection deterministic and cheaper.
    if bundle_size == 1:
        single_skill = skill_vecs[0]
        sim_01 = clip01((float(np.dot(single_skill, task_vec)) + 1.0) / 2.0)
        semantic_score_01 = sim_01
        structural_score_01 = clip01(float(structural_neutral))
        sem_fusion_w, struct_fusion_w = 1.0, 0.0
        raw_score_01 = semantic_score_01
        length_penalty = 0.0

        risk_penalty = 0.0
        token_penalty = 0.0
        if permission_risk_values is not None and permission_risk_values.size > 0:
            rv = permission_risk_values.astype(np.float32, copy=False)
            risk_penalty = float(0.60 * np.mean(rv) + 0.40 * np.max(rv))
        if token_cost_values is not None and token_cost_values.size > 0:
            tv = token_cost_values.astype(np.float32, copy=False)
            token_penalty = float(0.70 * np.mean(tv) + 0.20 * np.max(tv) + 0.10 * np.min(tv))

        # Keep seed score purely similarity-driven.
        final_score = 100.0 * clip01(raw_score_01)
        return BundleScoreResult(
            semantic_vec=single_skill.astype(np.float32),
            relation_vec=single_skill.astype(np.float32),
            fused_vec=single_skill.astype(np.float32),
            attention_weights=np.asarray([1.0], dtype=np.float32),
            head_vectors=single_skill.reshape(1, -1).astype(np.float32),
            head_item_weights=np.asarray([[1.0]], dtype=np.float32),
            head_gate_weights=np.asarray([1.0], dtype=np.float32),
            head_scores=np.asarray([0.0], dtype=np.float32),
            alpha=1.0,
            p_syn=0.0,
            p_red=0.0,
            skill_task_score=float(sim_01 * 100.0),
            bundle_task_score=0.0,
            task_bundle_score=0.0,
            attention_score=0.0,
            relation_align_score=float(sim_01 * 100.0),
            semantic_task_weight=1.0,
            semantic_attention_weight=0.0,
            semantic_bundle_weight=0.0,
            fusion_semantic_weight=float(sem_fusion_w),
            fusion_structural_weight=float(struct_fusion_w),
            raw_fused_score=float(raw_score_01 * 100.0),
            length_penalty=0.0,
            semantic_score=float(semantic_score_01 * 100.0),
            structural_score=float(structural_score_01 * 100.0),
            risk_penalty=float(risk_penalty * 100.0),
            token_penalty=float(token_penalty * 100.0),
            final_score=float(final_score),
        )

    bundle_mean = l2norm_vec(skill_vecs.mean(axis=0))

    attn_tau_match = (
        float(scorer_config.get("semantic_attention_tau_match", 0.20))
        if scorer_config
        else 0.20
    )
    attn_tau_self = (
        float(scorer_config.get("semantic_attention_tau_self", 0.25))
        if scorer_config
        else 0.25
    )
    (
        semantic_vec,
        attention_weights,
        head_vectors,
        head_item_weights,
        head_gate_weights,
        head_scores,
    ) = _multihead_semantic_bundle(
        task_vec,
        skill_vecs,
        num_heads=num_heads,
        tau_match=attn_tau_match,
        tau_self=attn_tau_self,
        complement_low=float(complement_low),
        redundancy_threshold=float(redundancy_threshold),
        scorer_config=scorer_config,
    )
    # Keep relation branch independent from attention semantic aggregation.
    # relation_vec now comes from pooled bundle representation instead of
    # reusing semantic_vec, so attention/relation do not collapse to one signal.
    relation_vec = bundle_mean

    # Split semantic signals into:
    # - skill_task_score: individual skill relevance to task
    # - bundle_task_score: pooled bundle representation to task
    # - attention_score: task-conditioned skill-bundle interaction
    sim_each_01 = np.clip(((skill_vecs @ task_vec) + 1.0) / 2.0, 0.0, 1.0)
    sim_max_01 = float(np.max(sim_each_01)) if sim_each_01.size else 0.0
    sim_mean_01 = float(np.mean(sim_each_01)) if sim_each_01.size else 0.0
    sim_pool_01 = clip01((float(np.dot(task_vec, bundle_mean)) + 1.0) / 2.0)
    sim_absdiff_01 = clip01(1.0 - float(np.mean(np.abs(bundle_mean - task_vec))) / 2.0)
    sim_prod_01 = clip01((float(np.mean(bundle_mean * task_vec)) + 1.0) / 2.0)
    sim_l2_01 = clip01(1.0 - float(np.linalg.norm(bundle_mean - task_vec)) / 2.0)
    # Fixed/default intra-view mix for skill-task branch:
    # alpha * max_sim + (1 - alpha) * mean_sim.
    # Default alpha preserves the previous max/mean ratio after pooled bundle
    # similarity was split into the explicit bundle-task branch.
    skill_task_alpha = (
        clip01(float(scorer_config.get("skill_task_alpha", 0.55 / 0.85)))
        if scorer_config
        else float(0.55 / 0.85)
    )
    skill_task_score_01 = clip01(skill_task_alpha * sim_max_01 + (1.0 - skill_task_alpha) * sim_mean_01)
    # Keep MLP result only as diagnostic; semantic fusion uses the packaged
    # attention-as-bundle-task view.
    _bundle_task_score_mlp_01 = _bundle_task_mlp_score(
        sim_pool_01=sim_pool_01,
        sim_absdiff_01=sim_absdiff_01,
        sim_prod_01=sim_prod_01,
        sim_l2_01=sim_l2_01,
        scorer_config=scorer_config,
    )
    attention_score_01 = clip01(float(np.sum(head_gate_weights * (head_scores / 100.0))))
    # Package attention branch as bundle-task view.
    bundle_task_score_01 = attention_score_01
    relation_align_01 = clip01((float(np.dot(task_vec, relation_vec)) + 1.0) / 2.0)

    sem_task_w = float(scorer_config.get("semantic_task_weight", 0.25)) if scorer_config else 0.25
    sem_attn_raw = float(scorer_config.get("semantic_attention_weight", 0.50)) if scorer_config else 0.50
    sem_bundle_raw = float(scorer_config.get("semantic_bundle_weight", scorer_config.get("semantic_coverage_weight", 0.25))) if scorer_config else 0.25
    # Merge old attention/bundle weights into one packaged bundle-task branch.
    sem_bundle_w = sem_attn_raw + sem_bundle_raw
    sem_total = max(sem_task_w + sem_bundle_w, 1e-6)
    sem_task_w /= sem_total
    sem_bundle_w /= sem_total
    sem_attn_w = 0.0
    structural_score_01 = clip01(
        structural_neutral + structural_bonus_scale * (relation_align_01 - structural_neutral)
    )
    semantic_score_01 = clip01(
        sem_task_w * skill_task_score_01
        + sem_bundle_w * bundle_task_score_01
    )
    sem_fusion_w, struct_fusion_w = _normalize_fusion_weights(
        semantic_weight=semantic_weight,
        structural_weight=structural_weight,
        min_share=min_share,
    )
    # Keep semantic/relation vectors as diagnostics only; scalar scoring uses
    # semantic_score_01 and structural_score_01 directly.
    alpha = sem_fusion_w
    fused_vec = relation_vec
    raw_score_01 = clip01(sem_fusion_w * semantic_score_01 + struct_fusion_w * structural_score_01)
    # Length growth is controlled by beam gain thresholds, not score penalty.
    length_penalty = 0.0

    risk_penalty = 0.0
    token_penalty = 0.0

    if permission_risk_values is not None and permission_risk_values.size > 0:
        rv = permission_risk_values.astype(np.float32, copy=False)
        risk_penalty = float(0.60 * np.mean(rv) + 0.40 * np.max(rv))

    if token_cost_values is not None and token_cost_values.size > 0:
        tv = token_cost_values.astype(np.float32, copy=False)
        token_penalty = float(0.70 * np.mean(tv) + 0.20 * np.max(tv) + 0.10 * np.min(tv))

    adjusted_score_01 = raw_score_01 - risk_lambda * risk_penalty - token_lambda * token_penalty
    final_score = 100.0 * clip01(adjusted_score_01)

    return BundleScoreResult(
        semantic_vec=semantic_vec.astype(np.float32),
        relation_vec=relation_vec.astype(np.float32),
        fused_vec=fused_vec.astype(np.float32),
        attention_weights=attention_weights.astype(np.float32),
        head_vectors=head_vectors.astype(np.float32),
        head_item_weights=head_item_weights.astype(np.float32),
        head_gate_weights=head_gate_weights.astype(np.float32),
        head_scores=head_scores.astype(np.float32),
        alpha=float(alpha),
        p_syn=0.0,
        p_red=0.0,
        skill_task_score=float(skill_task_score_01 * 100.0),
        bundle_task_score=float(bundle_task_score_01 * 100.0),
        task_bundle_score=float(bundle_task_score_01 * 100.0),
        attention_score=float(attention_score_01 * 100.0),
        relation_align_score=float(relation_align_01 * 100.0),
        semantic_task_weight=float(sem_task_w),
        semantic_attention_weight=float(sem_attn_w),
        semantic_bundle_weight=float(sem_bundle_w),
        fusion_semantic_weight=float(sem_fusion_w),
        fusion_structural_weight=float(struct_fusion_w),
        raw_fused_score=float(raw_score_01 * 100.0),
        length_penalty=float(length_penalty * 100.0),
        semantic_score=float(semantic_score_01 * 100.0),
        structural_score=float(structural_score_01 * 100.0),
        risk_penalty=float(risk_penalty * 100.0),
        token_penalty=float(token_penalty * 100.0),
        final_score=float(final_score),
    )

