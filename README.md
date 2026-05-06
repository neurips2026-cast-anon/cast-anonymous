# Supplementary Material for Bundle Recommendation for Agentic Skills

This directory combines the reviewer-facing supplementary `code/` and `data/` folders into a single package candidate.

Contents:

- `code/`
  Compact supplementary code release.
- `data/`
  Benchmark, graph, embedding, and split data used by the released code.

## Current Size Status

The current combined directory is substantially larger than the NeurIPS supplementary upload limit of 100 MB.

Main size contributors include:

- `data/benchmark_data/bundle_input/skills_embedding.npy`
- `data/benchmark_data/benchmark_merged_skill_embeddings.npy`
- `data/cast_graph_data/skill_nodes.json`
- `data/benchmark_data/benchmark_merged_skills.json`
- `data/benchmark_data/bundle_input/skills_with_risk_cost.json`
- `data/cast_graph_data/bundle_skill_interactions.jsonl`

In its current form, this directory is suitable for local organization and auditing, but it is too large for direct conference submission without further pruning.

## Practical Submission Note

If a strict 100 MB limit must be satisfied, the most likely pruning targets are:

1. duplicated embedding files
2. duplicated full-library JSON exports
3. train/val/test graph expansions that can be regenerated from smaller canonical files
4. files preserved only for convenience rather than minimal reproducibility

The `code/README.md` and `data/README.md` files describe the contents of each subfolder in detail.
