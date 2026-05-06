# CAST Anonymous Release

This repository provides anonymized supplementary code and data for the NeurIPS 2026 submission **Bundle Recommendation for Agentic Skills**.

## Contents

- `mainline_pipeline/`: code for bundle construction, scoring, and evaluation.
- `hypergraph_pipeline/`: code for task-local hypergraph reranking.
- `artifacts/cast_main/`: released model checkpoints and reranker configuration.
- `data/benchmark_data/`: benchmark annotations, train/test splits, skill library, embeddings, and evaluation utilities.
- `data/cast_graph_data/`: graph-structured data used by the CAST hypergraph pipeline.

Large embedding files are stored with Git LFS. Please run:

```bash
git lfs install
git lfs pull
