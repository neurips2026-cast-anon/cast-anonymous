# Supplementary Data for Bundle Recommendation for Agentic Skills

This folder packages the data needed to understand and reproduce the code released in the supplementary `code/` folder.

It is organized into two parts:

1. `benchmark_data/`
   Data used by the current `paper/code/` mainline bundle-construction and reranking pipeline.
2. `cast_graph_data/`
   Data used by the CAST hypergraph pipeline under `paper/pet+hypergraph/code/`.

## 1. Included Mainline Data

`benchmark_data/` contains the benchmark assets needed by the current mainline pipeline:

- `benchmark_gold_manifest.json`
  Gold bundle annotations for benchmark tasks.
- `benchmark_negative_bundle_index.json`
  Constructed negative bundles used for bundle-level supervision.
- `benchmark_negative_bundle_index_summary.json`
  Summary statistics for the negative-bundle index.
- `benchmark_train_test_split.json`
  Main train/validation/test split.
- `benchmark_train_test_split_frozen.json`
  Frozen copy of the main split.
- `benchmark_train_valtest_swapped_split.json`
  Validation/test swapped split used for robustness checks.
- `tasks_all_metadata_for_embedding_v2.json`
  Structured task metadata used to build task embeddings.
- `task_embeddings_bigmodel_512_f32_v2.npy`
  Task embedding matrix.
- `task_embeddings_bigmodel_512_f32_v2.meta.json`
  Metadata describing task embedding alignment.
- `benchmark_merged_skills.json`
  Full merged skill library with descriptions and risk/cost annotations.
- `benchmark_merged_skill_embeddings.npy`
  Skill embedding matrix aligned with `benchmark_merged_skills.json`.
- `default_bundle_scorer_config.json`
  Default mainline scorer configuration.
- `benchmark_gold_aware_top4_rerank_config_with_risk_cost.json`
  Learned reranker configuration used by the mainline reranking stage. The renamed copy makes explicit that the reranker configuration uses bundle-level structural features together with risk/cost-aware inputs.
- `eval_bundle_metrics.py`
  Evaluation utility for bundle predictions.

The subdirectory `benchmark_data/bundle_input/` contains the retrieval-side files used directly by `paper/code/run_pipeline.ps1`:

- `skills_with_risk_cost.json`
- `skills_raw.json`
- `skills_embedding.npy`

Although some original project filenames used ``risk''-only wording, the packaged supplementary copy makes the joint role of risk and cost explicit. In particular:

- `benchmark_merged_skills.json` contains both `permission_risk_*` and `token_cost_*` fields for each skill.
- `benchmark_data/bundle_input/skills_with_risk_cost.json` contains the same joint risk/cost annotations used by the bundle scorer and reranker.
- the hypergraph skill node files also preserve both risk- and cost-related annotations.

## 2. Included CAST Hypergraph Data

`cast_graph_data/` contains the graph-structured and split-structured data needed by the CAST hypergraph pipeline:

- `pet_bundle_catalog.json`
- `pet_task_bundle_edges.jsonl`
- `pet_task_skill_edges.jsonl`
- `pet_bundle_skill_edges.jsonl`
- `pet_graph_summary.json`
- `task_bundle_interactions.jsonl`
- `task_bundle_catalog.json`
- `bundle_skill_interactions.jsonl`
- `task_skill_interactions.jsonl`
- `task_nodes.json`
- `bundle_nodes.json`
- `skill_nodes.json`
- `pet_node_summary.json`

The subdirectory `cast_graph_data/splits/` contains split-specific train / val / test graph data:

- `splits/train/`
- `splits/val/`
- `splits/test/`

Each split folder includes:

- `task_nodes.json`
- `bundle_nodes.json`
- `skill_nodes.json`
- `task_bundle_interactions.jsonl`
- `task_skill_interactions.jsonl`
- `bundle_skill_interactions.jsonl`
- `split_summary.json`

## 3. What Was Intentionally Excluded

This package intentionally excludes files that are not required for supplementary review or that may create avoidable confusion:

- backup files such as `*.bak*`
- archived directories such as `archive_generated_*`
- older / deprecated files such as `*.old61*`
- task-new asset summaries unrelated to the paper method
- diagnostic and development-only data products not required to run the released methods
- files whose main value is intermediate bookkeeping rather than reproducibility

In particular, we excluded several local cache / bookkeeping files that embed workstation-specific paths or development-only references and are not required to reproduce the method.

## 4. Sanity Check

We manually filtered the packaged data to avoid:

- absolute local paths such as `E:\...` or `C:\Users\...`
- local username references
- private S3-style URIs
- obvious backup and temporary artifacts

No such workstation-specific references were found in the final packaged data folder.

## 5. Relation to the Supplementary Code Folder

- Use `benchmark_data/` together with `code/mainline_pipeline/` for the current mainline pipeline.
- Use `cast_graph_data/` together with `code/hypergraph_pipeline/` for the CAST hypergraph pipeline.

The top-level paper draft may discuss both lines in different sections or historical comparisons; this data package keeps both available to avoid omissions during review.

## 6. Note on Regenerable Intermediate Artifacts

One historical orchestration script in the code package, `run_pipeline.ps1`, references an intermediate contrastive-score cache named `benchmark_bundle_contrastive_scores_rebalanced.json`. That file is not included here because it is a derived training artifact rather than a canonical benchmark asset. It can be regenerated from the released code and benchmark bundle data if needed. The packaged data folder therefore focuses on benchmark assets, graph data, embeddings, and released reranker/scorer inputs rather than every transient intermediate produced during development.

## 7. Naming Adjustments for Clarity

To reduce reviewer confusion, we renamed a small number of copied supplementary files relative to their original working-directory names:

- `skills_with_risk.json` -> `skills_with_risk_cost.json`
- `benchmark_gold_aware_top4_rerank_config_bs9_pca_blend06_exact.json` -> `benchmark_gold_aware_top4_rerank_config_with_risk_cost.json`

These are filename-only changes inside the supplementary package. Their contents are unchanged except for packaging context.

When running scripts directly without overriding command-line arguments, some code files may still reference the original working-directory names. In such cases, either:

- pass the packaged filenames explicitly via command-line arguments, or
- create local filename aliases matching the original names.

The renamed files are semantically identical to their original counterparts.
