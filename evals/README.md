# Evaluations

This folder is for de-identified evaluation cases and metrics.

Do not commit real applicant documents, raw OCR from private documents, or generated reports containing private information.

The next milestone is to compare:

```text
OCR -> extraction
OCR -> extraction -> validation
OCR -> extraction -> validation -> targeted retry
```

## Local benchmark flow

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
