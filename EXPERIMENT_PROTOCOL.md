# CAST Experiment Protocol for Bundle Recommendation for Agentic Skills

## Frozen Benchmark Split

Use the fixed split:

- `benchmark_train_test_split_frozen.json`

Counts:

- Train: `49`
- Test: `12`

This split is frozen and should not be changed across experiments.

## Roles of Each Dataset

### 61 labeled benchmark tasks

- Train split:
  - train beam scorer
  - construct positive/negative bundles
  - train reranker
- Test split:
  - final evaluation only
  - do not use for parameter or model selection

### 89 unlabeled real tasks

These tasks are **not** part of the benchmark train/test protocol.

They are only used for:

- auxiliary analysis
- downstream agent execution
- risk/cost sanity checks

They are not used as a labeled validation set.

## Current Mainline System

The current mainline pipeline is:

1. top20 retrieval
2. risk/cost labeling
3. three-view beam scorer
4. beam search candidate generation
5. hypergraph reranking
6. final benchmark evaluation

## Current Best Development Result

The current best development result is stored in:

- `benchmark_main_eval.json`

This result was obtained during development and should be treated as a development-best reference, not as a clean final test result under a never-touched holdout protocol.

## Training Separation

### Beam scorer training

- use labeled benchmark train tasks
- use relative supervision from positive/negative bundles

### Hypergraph reranker training

- use labeled benchmark train tasks
- train separately from beam scorer

### Unlabeled tasks

- no gold-relative loss
- no direct participation in benchmark model selection

## Hyperparameter Tuning Plan (Post-Refactor)

For the full-skill-library beam pipeline (no external top-20 file), tune on train split only:

- `max-candidates` (recommended search: `48, 64, 80`)
- `beam-size` (`8, 10, 12`)
- `final-candidate-count` (`4, 6, 8`)
- `stop-threshold` (`0.20, 0.25, 0.30`)
- `min-bundle-size` (`2` fixed by default)
- `min-balanced-gain` (`-0.90, -0.75, -0.60`)

For scorer penalty controls:

- `risk_lambda`
- `token_lambda`
- length penalty params in scorer (`gain_threshold`, `growth`)

Principle:

- Learn semantic representation parameters.
- Keep policy-style controls as explicit hyperparameters and report sensitivity.
