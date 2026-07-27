# Local evaluation outputs

Evaluation outputs should be written under `evals/results/`.

This directory is ignored by Git because it may contain private applicant information.

## Candidate-level baseline outputs

Default command:

```bash
python evals/run_candidate_baseline.py --cases evals/private/candidates.v0.local.jsonl
```

Default output layout:

```text
evals/results/
├── candidate_baseline_results.local.jsonl
└── candidate_outputs/
    ├── *_combined_result.txt
    └── *_report.html
```

`candidate_baseline_results.local.jsonl` is the machine-readable benchmark result.

`candidate_outputs/` contains the same combined text files and HTML reports produced by the application workflow.

## Manual server outputs

If the model only runs on the server Gradio app, download the generated files manually into:

```text
evals/results/manual_candidate_outputs/
├── candidate_001/
│   ├── *_combined_result.txt
│   └── *_report.html
└── candidate_002/
    ├── *_combined_result.txt
    └── *_report.html
```

Then convert those files into benchmark JSONL:

```bash
python evals/import_manual_candidate_outputs.py \
  --cases evals/private/candidates.v0.local.jsonl
```

This writes:

```text
evals/results/manual_candidate_results.local.jsonl
```

Run metrics against the imported manual results:

```bash
python evals/candidate_metrics.py \
  --cases evals/private/candidates.v0.local.jsonl \
  --results evals/results/manual_candidate_results.local.jsonl
```

## Metrics

After running the baseline:

```bash
python evals/candidate_metrics.py --cases evals/private/candidates.v0.local.jsonl
```
