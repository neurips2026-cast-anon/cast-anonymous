# CAST Anonymous Release

This repository provides anonymized supplementary code and data for the NeurIPS 2026 submission **Bundle Recommendation for Agentic Skills**.

## Contents

- `mainline_pipeline/`: code for bundle construction, scoring, and evaluation.
- `hypergraph_pipeline/`: code for task-local hypergraph reranking.
- `artifacts/cast_main/`: released model checkpoints and reranker configuration.
- `data/benchmark_data/`: benchmark annotations, train/test splits, skill library, embeddings, and evaluation utilities.
- `data/cast_graph_data/`: graph-structured data used by the CAST hypergraph pipeline.

Large embedding files are stored with Git LFS. After cloning the repository, run:

```bash
git lfs install
git lfs pull
```

## Environment

Install dependencies from the repository root:

```bash
pip install -r requirements_cast_main_reproduction.txt
```

## Reproduce Main Offline Results

On Windows PowerShell, run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_cast_main_reproduction.ps1
```

The script performs candidate search, hypergraph reranking, and final evaluation. Reproduced outputs are written to the configured output directory.

## Data

The benchmark data are provided under `data/benchmark_data/`. The graph-structured data used by the hypergraph pipeline are provided under `data/cast_graph_data/`.

Large embedding files are managed by Git LFS and must be downloaded with `git lfs pull` before running the reproduction script.

## Anonymity

This repository is anonymized for double-blind review. Author names, affiliations, institution-specific paths, and non-anonymous project identifiers are omitted.
