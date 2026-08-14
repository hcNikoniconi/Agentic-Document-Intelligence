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
├── UG000000_Example Applicant/
│   ├── *_combined_result.txt
│   └── *_report.html
└── UG000001_Another Applicant/
    ├── *_combined_result.txt
    └── *_report.html
```

Use the same folder names as the original applicant folders so manual downloads are easy to align.

For pipeline comparison, use versioned local folders:

```text
evals/results/manual_candidate_outputs_v0_ocr/
evals/results/manual_candidate_outputs_v1_vlm/
evals/results/manual_candidate_outputs_v2_vlm_text_agent/
```

Do not rename applicants to `candidate_001`; use the original applicant folder
name so results line up with ground truth.

Then convert those files into benchmark JSONL:

```bash
python evals/import_manual_candidate_outputs.py \
  --cases evals/private/candidates.v0.local.jsonl \
  --input-dir evals/results/manual_candidate_outputs_v0_ocr \
  --output evals/results/manual_candidate_results.v0_ocr.local.jsonl
```

This writes:

```text
evals/results/manual_candidate_results.local.jsonl
```

Run metrics against the imported manual results:

```bash
python evals/candidate_metrics.py \
  --cases evals/private/candidates.v0.local.jsonl \
  --results evals/results/manual_candidate_results.v0_ocr.local.jsonl
```

The metric script applies conservative normalization before comparing fields, so
format-only differences are not counted as extraction errors. Examples:

- `20 SEP 2008`, `20 September 2008`, and `2008-09-20` are treated as the same date.
- `F` and `Female` are treated as the same gender.
- `PASSPORT` and `P` are treated as the same passport type.
- Extra spaces, casing, and punctuation differences are ignored for plain text.

Ground-truth values marked as `unknown` are scored as real expected values. This
means `unknown` is counted as correct only when the PDF does not contain that
information and the model also outputs `unknown`; if the PDF has a known value
but the model outputs `unknown`, it is counted as an extraction error.

Ground-truth values marked as `not_scored` are excluded from numeric accuracy.
Use this only when the benchmark rule itself is not settled yet, not when the
model simply misses a field.

## Metrics

After running the baseline:

```bash
python evals/candidate_metrics.py --cases evals/private/candidates.v0.local.jsonl
```
