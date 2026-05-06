$ErrorActionPreference = 'Stop'

# Run from any directory.
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$MainCodeDir = Join-Path $RepoRoot "mainline_pipeline"
$HyperCodeDir = Join-Path $RepoRoot "hypergraph_pipeline"
$DataDir = Join-Path $RepoRoot "data\benchmark_data"
$OutputDir = Join-Path $RepoRoot "reproduction_outputs\cast_main"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Repository root: $RepoRoot"
Write-Host "Mainline code:   $MainCodeDir"
Write-Host "Hypergraph code: $HyperCodeDir"
Write-Host "Data directory:  $DataDir"
Write-Host "Output directory:$OutputDir"

# 1. Train the mainline bundle scorer.
Set-Location $MainCodeDir

python train_mainline_scorer.py `
  --gold-manifest (Join-Path $DataDir "benchmark_gold_manifest.json") `
  --split-json (Join-Path $DataDir "benchmark_train_test_split.json") `
  --contrastive-scores-json (Join-Path $DataDir "benchmark_bundle_contrastive_scores_rebalanced.json") `
  --task-emb (Join-Path $DataDir "task_embeddings_bigmodel_512_f32_v2.npy") `
  --skill-emb (Join-Path $DataDir "bundle_input\skills_embedding.npy") `
  --report-json (Join-Path $OutputDir "internal_attention_train_report.json") `
  --export-scorer-config-json (Join-Path $OutputDir "trained_scorer_config.json")

# 2. Search candidate skill bundles.
python search_skill_bundles.py `
  --skills-json (Join-Path $DataDir "bundle_input\skills_with_risk.json") `
  --task-json (Join-Path $DataDir "tasks_all_metadata_for_embedding_v2.json") `
  --task-emb (Join-Path $DataDir "task_embeddings_bigmodel_512_f32_v2.npy") `
  --skill-emb (Join-Path $DataDir "bundle_input\skills_embedding.npy") `
  --max-candidates 64 `
  --beam-size 12 `
  --stop-threshold 0.25 `
  --min-bundle-size 2 `
  --min-balanced-gain -0.75 `
  --final-candidate-count 6 `
  --final-selection-mode diverse_topk `
  --scorer-config (Join-Path $OutputDir "trained_scorer_config.json") `
  --output (Join-Path $OutputDir "beam_search.json")

# 3. Build hypergraph candidates.
Set-Location $HyperCodeDir

python build_hypergraph_candidates.py `
  --beam-json (Join-Path $OutputDir "beam_search.json") `
  --skills-json (Join-Path $DataDir "bundle_input\skills_with_risk.json") `
  --output-json (Join-Path $OutputDir "hypergraph_candidates.json")

# 4. Train the hypergraph reranker.
python train_bundle_reranker.py `
  --beam-json (Join-Path $OutputDir "beam_search.json") `
  --hypergraph-json (Join-Path $OutputDir "hypergraph_candidates.json") `
  --gold-manifest (Join-Path $DataDir "benchmark_gold_manifest.json") `
  --split-json (Join-Path $DataDir "benchmark_train_test_split.json") `
  --output-json (Join-Path $OutputDir "benchmark_gold_aware_top4_rerank_config_bs9_pca_blend06_exact.json") `
  --rerank-candidate-count 6

# 5. Apply the hypergraph reranker.
python apply_bundle_reranker.py `
  --beam-json (Join-Path $OutputDir "beam_search.json") `
  --hypergraph-json (Join-Path $OutputDir "hypergraph_candidates.json") `
  --config-json (Join-Path $OutputDir "benchmark_gold_aware_top4_rerank_config_bs9_pca_blend06_exact.json") `
  --output-json (Join-Path $OutputDir "reranked.json") `
  --rerank-candidate-count 6

# 6. Evaluate final predictions.
Set-Location $MainCodeDir

python evaluate_mainline_predictions.py `
  --beam-json (Join-Path $OutputDir "reranked.json") `
  --gold-manifest (Join-Path $DataDir "benchmark_gold_manifest.json") `
  --split-json (Join-Path $DataDir "benchmark_train_test_split.json") `
  --use-rerank `
  --output-json (Join-Path $OutputDir "eval.json")

Write-Host "Reproduction complete. Results saved to $OutputDir"
