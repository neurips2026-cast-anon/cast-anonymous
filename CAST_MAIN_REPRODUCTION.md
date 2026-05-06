# Bundle Recommendation for Agentic Skills: CAST Main Reproduction

This note documents the reviewer-facing main reproduction entry for the CAST line in *Bundle Recommendation for Agentic Skills*.

## Scope

The packaged reproduction target is the main CAST test result produced by:

- scorer checkpoint: `artifacts/cast_main/cast_main_scorer.pt`
- hypergraph reranker config: `artifacts/cast_main/cast_main_hypergraph_config.json`
- beam-search setting: `beam_size = 5`

The reproduction script runs the search, hypergraph reranking, and final evaluation stages with explicit supplementary-relative paths.

## One-command reproduction

Run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File ".\paper\Supplementary Material\code\run_cast_main_reproduction.ps1"
```

The script writes outputs to:

- `paper\Supplementary Material\code\reproduction_outputs\cast_main\`

and prints the final summary metrics to the console.

## Environment

Verified local environment:

- Python `3.12.4`
- `numpy==1.26.4`
- `torch==2.11.0+cpu`

The minimal package list is recorded in:

- `requirements_cast_main_reproduction.txt`

## Alignment

The source hypergraph scripts under:

- `paper/pet+hypergraph/code/`

and the supplementary copies under:

- `paper/Supplementary Material/code/hypergraph_pipeline/`

are byte-identical for the files used in this experiment:

- `train_cast_scorer.py`
- `search_candidate_bundles.py`
- `rerank_hypergraph_candidates.py`
- `evaluate_bundle_predictions.py`
- `train_hypergraph_reranker.py`
- `cast_relevance3_model.py`

The core graph tables under `paper/pet+hypergraph/data/` and `paper/Supplementary Material/data/cast_graph_data/` are also byte-identical for the tables used by the packaged reproduction script.
