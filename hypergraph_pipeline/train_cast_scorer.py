#!/usr/bin/env python3
"""Train the two-layer Typed LightGCN PET scorer on the train graph."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cast_relevance3_model import TaskPETRelevance3Scorer, load_pet_graph_tensors


NEG_TYPE_WEIGHT = {
    "missing": 3.0,
    "conflict": 1.5,
    "redundancy": 1.5,
    "zero_interaction": 0.5,
    "replace": 1.0,
    "multi_replace": 1.0,
    "higher_risk": 1.0,
    "higher_cost": 1.0,
}


def expert_index_for_neg_type(neg_type: str) -> int:
    if neg_type in {"replace", "multi_replace", "zero_interaction"}:
        return 0  # relevance expert
    if neg_type in {"missing", "redundancy", "conflict"}:
        return 1  # structure expert
    if neg_type in {"higher_risk", "higher_cost"}:
        return 2  # safety-cost expert
    return -1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def build_training_edges(split_dir: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    task_nodes = json.loads((split_dir / "task_nodes.json").read_text(encoding="utf-8"))
    bundle_nodes = json.loads((split_dir / "bundle_nodes.json").read_text(encoding="utf-8"))
    task_map = {int(x["task_index"]): i for i, x in enumerate(task_nodes)}
    bundle_map = {int(x["bundle_id"]): i for i, x in enumerate(bundle_nodes)}

    task_ids = []
    bundle_ids = []
    targets = []
    negative_types = []
    for row in read_jsonl(split_dir / "task_bundle_interactions.jsonl"):
        gti = int(row["task_index"])
        gbid = int(row["bundle_id"])
        if gti not in task_map or gbid not in bundle_map:
            continue
        task_ids.append(task_map[gti])
        bundle_ids.append(bundle_map[gbid])
        targets.append(float(row["interaction"]))
        negative_types.append(str(row.get("negative_type", "positive" if float(row["interaction"]) == 1.0 else "other_negative")))
    return (
        torch.tensor(task_ids, dtype=torch.long, device=device),
        torch.tensor(bundle_ids, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
        negative_types,
    )


def build_bpr_pairs(
    task_idx: torch.Tensor,
    bundle_idx: torch.Tensor,
    targets: torch.Tensor,
    negative_types: list[str],
    *,
    hard_per_positive: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    rng = random.Random(seed)
    by_task: dict[int, dict[str, list[int]]] = {}
    for row_id, (ti, y) in enumerate(zip(task_idx.detach().cpu().tolist(), targets.detach().cpu().tolist())):
        group = by_task.setdefault(int(ti), {"pos": [], "soft": [], "hard": []})
        if float(y) == 1.0:
            group["pos"].append(row_id)
        elif float(y) > 0.0:
            group["soft"].append(row_id)
        else:
            group["hard"].append(row_id)

    pos_rows: list[int] = []
    neg_rows: list[int] = []
    margins: list[float] = []
    weights: list[float] = []
    neg_types: list[str] = []
    for group in by_task.values():
        if not group["pos"]:
            continue
        pos_id = group["pos"][0]
        soft_ids = list(group["soft"])
        hard_ids = list(group["hard"])
        rng.shuffle(soft_ids)
        rng.shuffle(hard_ids)
        # gold > soft
        for neg_id in soft_ids[: max(1, min(len(soft_ids), 4))]:
            pos_rows.append(pos_id)
            neg_rows.append(neg_id)
            margins.append(0.05)
            weights.append(0.7 * float(NEG_TYPE_WEIGHT.get(negative_types[neg_id], 1.0)))
            neg_types.append(negative_types[neg_id])
        # gold > hard
        for neg_id in hard_ids[: min(len(hard_ids), int(hard_per_positive))]:
            pos_rows.append(pos_id)
            neg_rows.append(neg_id)
            margins.append(0.15)
            weights.append(1.0 * float(NEG_TYPE_WEIGHT.get(negative_types[neg_id], 1.0)))
            neg_types.append(negative_types[neg_id])

    order = list(range(len(pos_rows)))
    rng.shuffle(order)
    return (
        torch.tensor([pos_rows[i] for i in order], dtype=torch.long, device=targets.device),
        torch.tensor([neg_rows[i] for i in order], dtype=torch.long, device=targets.device),
        torch.tensor([margins[i] for i in order], dtype=torch.float32, device=targets.device),
        torch.tensor([weights[i] for i in order], dtype=torch.float32, device=targets.device),
        [neg_types[i] for i in order],
    )


def inter_cl_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2, max_nodes: int = 512) -> torch.Tensor:
    n = min(int(z1.shape[0]), int(z2.shape[0]), int(max_nodes))
    if n <= 1:
        return torch.zeros((), dtype=z1.dtype, device=z1.device)
    a = F.normalize(z1[:n], dim=1)
    b = F.normalize(z2[:n], dim=1)
    logits = (a @ b.T) / max(float(tau), 1e-6)
    labels = torch.arange(n, dtype=torch.long, device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def evaluate_split(
    model: TaskPETRelevance3Scorer,
    graph,
    task_idx: torch.Tensor,
    bundle_idx: torch.Tensor,
    targets: torch.Tensor,
    negative_types: list[str],
    *,
    hard_per_positive: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        enc = model.encode(graph)
        pred, _stats = model.score_bundle_ids(graph, enc, task_idx, bundle_idx)
        mse = float(F.mse_loss(pred, targets).detach().cpu())
        pos_rows, neg_rows, _margins, weights, _pair_neg_types = build_bpr_pairs(
            task_idx,
            bundle_idx,
            targets,
            negative_types,
            hard_per_positive=hard_per_positive,
            seed=seed,
        )
        if pos_rows.numel() == 0:
            bpr = 0.0
        else:
            pos_score, _ = model.score_bundle_ids(graph, enc, task_idx[pos_rows], bundle_idx[pos_rows])
            neg_score, _ = model.score_bundle_ids(graph, enc, task_idx[neg_rows], bundle_idx[neg_rows])
            bpr = float(torch.mean(weights * F.softplus(-(pos_score - neg_score))).detach().cpu())
    return {"mse": mse, "bpr": bpr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Typed LightGCN PET scorer.")
    parser.add_argument("--split-dir", default="../data/splits/train")
    parser.add_argument("--val-split-dir", default="../data/splits/val")
    parser.add_argument("--paper-data", default="../../data")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--structure-mode", choices=("soft", "learned_red"), default="soft")
    parser.add_argument("--view-mode", choices=("full", "task_skill_only"), default="full")
    parser.add_argument("--score-view-mode", choices=("full", "tb_only", "ts_only", "bs_only"), default="full")
    parser.add_argument("--expert-subset", choices=("full", "rel_only", "rel_struct", "rel_safe"), default="full")
    parser.add_argument("--task-prompt-mode", choices=("none", "prompt", "retrieved"), default="none")
    parser.add_argument("--prompt-count", type=int, default=4)
    parser.add_argument("--retrieved-prompt-topm", type=int, default=4)
    parser.add_argument("--graph-path-dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-metric", choices=("bpr", "mse"), default="bpr")
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hard-per-positive", type=int, default=8)
    parser.add_argument("--lambda-inter", type=float, default=0.01)
    parser.add_argument("--lambda-expert", type=float, default=0.10)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--lambda-gate-entropy", type=float, default=0.001)
    parser.add_argument("--lambda-gate-floor-struct", type=float, default=0.01)
    parser.add_argument("--gate-floor-struct", type=float, default=0.12)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--inter-tau", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-model", default="../output/typed_lightgcn_pet_l2.pt")
    parser.add_argument("--output-report", default="../output/typed_lightgcn_pet_l2_report.json")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    split_dir = Path(args.split_dir).resolve()
    paper_data = Path(args.paper_data).resolve()

    graph = load_pet_graph_tensors(
        pet_split_dir=split_dir,
        paper_data_dir=paper_data,
        device=device,
        retrieved_prompt_topm=int(args.retrieved_prompt_topm),
    )
    task_idx, bundle_idx, targets, negative_types = build_training_edges(split_dir, device)
    val_split_dir = Path(args.val_split_dir).resolve()
    val_graph = load_pet_graph_tensors(
        pet_split_dir=val_split_dir,
        paper_data_dir=paper_data,
        device=device,
        retrieved_prompt_topm=int(args.retrieved_prompt_topm),
    )
    val_task_idx, val_bundle_idx, val_targets, val_negative_types = build_training_edges(val_split_dir, device)

    model = TaskPETRelevance3Scorer(
        dim=graph.task_emb.shape[1],
        num_layers=int(args.num_layers),
        gate_temperature=float(args.gate_temperature),
        structure_mode=str(args.structure_mode),
        view_mode=str(args.view_mode),
        score_view_mode=str(args.score_view_mode),
        expert_subset=str(args.expert_subset),
        task_prompt_mode=str(args.task_prompt_mode),
        prompt_count=int(args.prompt_count),
        graph_path_dropout=float(args.graph_path_dropout),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=1e-5)

    history = []
    best_state = None
    best_epoch = 0
    best_val = float("inf")
    bad_epochs = 0
    for epoch in range(int(args.epochs)):
        model.train()
        pos_rows, neg_rows, margins, weights, pair_neg_types = build_bpr_pairs(
            task_idx,
            bundle_idx,
            targets,
            negative_types,
            hard_per_positive=int(args.hard_per_positive),
            seed=int(args.seed) + epoch,
        )
        enc = model.encode(graph)
        if str(args.view_mode) == "full":
            loss_inter = inter_cl_loss(enc["task_tb"], enc["task_ts"], tau=float(args.inter_tau))
            loss_inter = loss_inter + inter_cl_loss(enc["bundle_tb"], enc["bundle_bs"], tau=float(args.inter_tau))
        else:
            loss_inter = torch.zeros((), dtype=targets.dtype, device=device)
        pos_score, pos_stats = model.score_bundle_ids(graph, enc, task_idx[pos_rows], bundle_idx[pos_rows])
        neg_score, neg_stats = model.score_bundle_ids(graph, enc, task_idx[neg_rows], bundle_idx[neg_rows])
        # PET-style BPR loss. Margins are intentionally not used in the first model.
        pair_loss = F.softplus(-(pos_score - neg_score))
        loss_bpr = torch.mean(weights * pair_loss)
        loss_expert = torch.zeros((), dtype=loss_bpr.dtype, device=device)
        pos_experts = pos_stats.get("experts")
        neg_experts = neg_stats.get("experts")
        if pos_experts is not None and neg_experts is not None:
            expert_losses = []
            for i, nt in enumerate(pair_neg_types):
                ei = expert_index_for_neg_type(nt)
                if ei < 0:
                    continue
                expert_losses.append(F.softplus(-(pos_experts[i, ei] - neg_experts[i, ei])))
            if expert_losses:
                loss_expert = torch.mean(torch.stack(expert_losses))
        gate = neg_stats["view_gate"]
        gate_entropy = -torch.mean(torch.sum(gate * torch.log(torch.clamp(gate, min=1e-8)), dim=1))
        # Encourage sparse routing: penalize overly high-entropy gates.
        entropy_ceiling = 0.45 * math.log(float(gate.shape[1]))
        gate_entropy_penalty = torch.relu(gate_entropy - torch.tensor(entropy_ceiling, dtype=gate_entropy.dtype, device=device)).pow(2)
        gate_mean = torch.mean(gate, dim=0)
        gate_floor_penalty = torch.relu(
            torch.tensor(float(args.gate_floor_struct), dtype=gate.dtype, device=device) - gate_mean[1]
        ).pow(2)
        l2_reg = torch.zeros((), dtype=loss_bpr.dtype, device=device)
        for p in model.parameters():
            l2_reg = l2_reg + torch.sum(p * p)
        loss_reg = (
            float(args.lambda_gate_entropy) * gate_entropy_penalty
            + float(args.lambda_gate_floor_struct) * gate_floor_penalty
            + float(args.lambda_reg) * l2_reg
        )
        loss = loss_bpr + float(args.lambda_inter) * loss_inter + float(args.lambda_expert) * loss_expert + loss_reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_count = int(pos_rows.numel())
        avg_loss = float(loss.detach().cpu())
        avg_bpr = float(loss_bpr.detach().cpu())
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_bpr": avg_bpr,
            "inter_cl": float(loss_inter.detach().cpu()),
            "expert_loss": float(loss_expert.detach().cpu()),
            "reg_loss": float(loss_reg.detach().cpu()),
            "gate_entropy": float(gate_entropy.detach().cpu()),
            "gate_entropy_penalty": float(gate_entropy_penalty.detach().cpu()),
            "gate_floor_penalty": float(gate_floor_penalty.detach().cpu()),
            "gate_mean_rel": float(gate_mean[0].detach().cpu()),
            "gate_mean_struct": float(gate_mean[1].detach().cpu()),
            "gate_mean_safe": float(gate_mean[2].detach().cpu()),
            "pair_count": total_count,
        })
        val_metrics = evaluate_split(
            model,
            val_graph,
            val_task_idx,
            val_bundle_idx,
            val_targets,
            val_negative_types,
            hard_per_positive=int(args.hard_per_positive),
            seed=int(args.seed) + 2000 + epoch,
        )
        history[-1]["val_mse"] = float(val_metrics["mse"])
        history[-1]["val_bpr"] = float(val_metrics["bpr"])
        monitor = float(val_metrics[str(args.early_stop_metric)])
        improved = monitor < (best_val - float(args.early_stop_min_delta))
        if improved:
            best_val = monitor
            best_epoch = epoch + 1
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        print(
            f"epoch={epoch+1} train_loss={avg_loss:.6f} bpr={avg_bpr:.6f} "
            f"inter={float(loss_inter.detach().cpu()):.6f} expert={float(loss_expert.detach().cpu()):.6f} "
            f"val_bpr={val_metrics['bpr']:.6f} val_mse={val_metrics['mse']:.6f} "
            f"pairs={total_count}"
        )
        if (not args.disable_early_stop) and bad_epochs >= int(args.early_stop_patience):
            print(
                f"early_stop epoch={epoch+1} best_epoch={best_epoch} "
                f"best_{args.early_stop_metric}={best_val:.6f}"
            )
            break

    if (not args.disable_early_stop) and best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model.eval()
    with torch.no_grad():
        enc = model.encode(graph)
        pred, stats = model.score_bundle_ids(graph, enc, task_idx, bundle_idx)
        final_mse = float(F.mse_loss(pred, targets).detach().cpu())
        pos_rows, neg_rows, margins, weights, _pair_neg_types = build_bpr_pairs(
            task_idx,
            bundle_idx,
            targets,
            negative_types,
            hard_per_positive=int(args.hard_per_positive),
            seed=int(args.seed) + 999,
        )
        pos_score, _ = model.score_bundle_ids(graph, enc, task_idx[pos_rows], bundle_idx[pos_rows])
        neg_score, _ = model.score_bundle_ids(graph, enc, task_idx[neg_rows], bundle_idx[neg_rows])
        final_bpr = float(torch.mean(weights * F.softplus(-(pos_score - neg_score))).detach().cpu())
        gates = stats["view_gate"].detach().cpu()
        report = {
            "device": str(device),
            "num_layers": int(args.num_layers),
            "structure_mode": str(args.structure_mode),
            "view_mode": str(args.view_mode),
            "score_view_mode": str(args.score_view_mode),
            "expert_subset": str(args.expert_subset),
            "task_prompt_mode": str(args.task_prompt_mode),
            "prompt_count": int(args.prompt_count),
            "retrieved_prompt_topm": int(args.retrieved_prompt_topm),
            "graph_path_dropout": float(args.graph_path_dropout),
            "train_edge_count": int(targets.numel()),
            "target_counts": {
                "positive_1": int((targets == 1.0).sum().detach().cpu()),
                "soft_0_7": int(((targets > 0.0) & (targets < 1.0)).sum().detach().cpu()),
                "hard_0": int((targets == 0.0).sum().detach().cpu()),
            },
            "loss_config": {
                "loss": "L_BPR + lambda_inter * L_InterCL + lambda_expert * L_expert + regularization",
                "lambda_inter": float(args.lambda_inter),
                "lambda_expert": float(args.lambda_expert),
                "lambda_reg": float(args.lambda_reg),
                "lambda_gate_entropy": float(args.lambda_gate_entropy),
                "lambda_gate_floor_struct": float(args.lambda_gate_floor_struct),
                "gate_floor_struct": float(args.gate_floor_struct),
                "hard_per_positive": int(args.hard_per_positive),
                "gate_temperature": float(args.gate_temperature),
                "inter_tau": float(args.inter_tau),
                "negative_type_weight": NEG_TYPE_WEIGHT,
            },
            "early_stop": {
                "metric": str(args.early_stop_metric),
                "patience": int(args.early_stop_patience),
                "min_delta": float(args.early_stop_min_delta),
                "best_epoch": int(best_epoch),
                "best_metric_value": float(best_val) if best_epoch > 0 else None,
                "disabled": bool(args.disable_early_stop),
            },
            "final_train_mse_all_edges": final_mse,
            "final_train_bpr_sampled_pairs": final_bpr,
            "mean_gate": [float(x) for x in gates.mean(dim=0).tolist()],
            "history": history,
        }

    out_model = Path(args.output_model).resolve()
    out_report = Path(args.output_report).resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "report": report}, out_model)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model : {out_model}")
    print(f"Saved report: {out_report}")


if __name__ == "__main__":
    main()
