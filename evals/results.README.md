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

## Metrics

After running the baseline:

```bash
python evals/candidate_metrics.py --cases evals/private/candidates.v0.local.jsonl
```

