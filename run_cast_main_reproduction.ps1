$ErrorActionPreference = 'Stop'

$CodeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $CodeRoot
$HyperCode = Join-Path $CodeRoot 'hypergraph_pipeline'
$DataRoot = Join-Path $RepoRoot 'data'
$MainData = Join-Path $DataRoot 'benchmark_data'
$PetData = Join-Path $DataRoot 'cast_graph_data'
$ArtifactDir = Join-Path $CodeRoot 'artifacts\cast_main'
$OutDir = Join-Path $CodeRoot 'reproduction_outputs\cast_main'

$ModelPath = Join-Path $ArtifactDir 'cast_main_scorer.pt'
$HyperConfig = Join-Path $ArtifactDir 'cast_main_hypergraph_config.json'

foreach($required in @($ModelPath, $HyperConfig)) {
  if (!(Test-Path -LiteralPath $required)) {
    throw "Missing required artifact: $required"
  }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$BeamJson = Join-Path $OutDir 'cast_main_test_beam.json'
$HyperJson = Join-Path $OutDir 'cast_main_test_hyper.json'
$EvalJson = Join-Path $OutDir 'cast_main_test_eval.json'

Set-Location $HyperCode

python .\search_candidate_bundles.py `
  --task-json (Join-Path $MainData 'tasks_all_metadata_for_embedding_v2.json') `
  --task-emb (Join-Path $MainData 'task_embeddings_bigmodel_512_f32_v2.npy') `
  --skill-json (Join-Path $MainData 'benchmark_merged_skills.json') `
  --skill-emb (Join-Path $MainData 'benchmark_merged_skill_embeddings.npy') `
  --train-split-dir (Join-Path $PetData 'splits\train') `
  --paper-data $MainData `
  --model-path $ModelPath `
  --structure-mode learned_red `
  --view-mode full `
  --score-view-mode full `
  --expert-subset full `
  --task-prompt-mode prompt `
  --prompt-count 4 `
  --retrieved-prompt-topm 4 `
  --split-json (Join-Path $MainData 'benchmark_train_test_split.json') `
  --split test `
  --max-candidates 30 `
  --beam-size 5 `
  --max-bundle-size 5 `
  --min-coverage-gain 0.0 `
  --min-bundle-size 2 `
  --final-candidate-count 10 `
  --final-selection-mode length_diverse `
  --no-direct-fallback `
  --device cpu `
  --output-json $BeamJson

python .\rerank_hypergraph_candidates.py `
  --beam-json $BeamJson `
  --skill-json (Join-Path $MainData 'benchmark_merged_skills.json') `
  --skill-emb (Join-Path $MainData 'benchmark_merged_skill_embeddings.npy') `
  --task-emb (Join-Path $MainData 'task_embeddings_bigmodel_512_f32_v2.npy') `
  --config-json $HyperConfig `
  --output-json $HyperJson

python .\evaluate_bundle_predictions.py `
  --prediction-json $HyperJson `
  --gold-json (Join-Path $MainData 'benchmark_gold_manifest.json') `
  --split-json (Join-Path $MainData 'benchmark_train_test_split.json') `
  --skill-json (Join-Path $MainData 'benchmark_merged_skills.json') `
  --skill-emb (Join-Path $MainData 'benchmark_merged_skill_embeddings.npy') `
  --split test `
  --output-json $EvalJson

$eval = Get-Content -LiteralPath $EvalJson -Raw | ConvertFrom-Json
$m = $eval.summary.metrics
Write-Output ("F1={0:N4} Precision={1:N4} Recall={2:N4} Jaccard={3:N4} NDCG@3={4:N4} Size={5:N4} Risk={6:N4} Cost={7:N4}" -f `
  $m.f1, $m.precision, $m.recall, $m.jaccard, $m.ndcg_at_k, $m.pred_size, $m.risk_mean, $m.cost_mean)
