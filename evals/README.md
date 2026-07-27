# Evaluations

This folder is for de-identified evaluation cases and metrics.

Do not commit real applicant documents, raw OCR from private documents, or generated reports containing private information.

The next milestone is to compare:

```text
OCR -> extraction
OCR -> extraction -> validation
OCR -> extraction -> validation -> targeted retry
```

## PDF-level local benchmark flow

Build a private manifest from local applicant folders:

```bash
python scripts/build_private_eval_manifest.py /path/to/documents-export-2026-01-30
```

Fill `evals/private/cases.local.jsonl` with ground-truth fields for a small subset first.

Run the current baseline:

```bash
python evals/run_baseline.py --limit 5
```

Compute metrics:

```bash
python evals/metrics.py
```

Everything under `evals/private/` and `evals/results/` is ignored by Git.

## Candidate-level benchmark flow

This is the preferred benchmark for the current application because the real workflow processes one applicant folder and produces one combined result.

Build a private candidate manifest:

```bash
python scripts/build_private_candidate_manifest.py /path/to/documents-export-2026-01-30
```

Run the current folder-level baseline:

```bash
python evals/run_candidate_baseline.py --limit 2
```

By default, outputs are saved locally under:

```text
evals/results/
├── candidate_baseline_results.local.jsonl
└── candidate_outputs/
    ├── *_combined_result.txt
    └── *_report.html
```

Compute candidate-level metrics:

```bash
python evals/candidate_metrics.py
```

If results are downloaded manually from the server Gradio UI, put them under `evals/results/manual_candidate_outputs/` and import them with:

```bash
python evals/import_manual_candidate_outputs.py --cases evals/private/candidates.v0.local.jsonl
```
