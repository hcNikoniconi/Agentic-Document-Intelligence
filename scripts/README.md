# Scripts

This folder contains repeatable local workflows for the v2 MVP.

Private candidate documents and generated outputs are intentionally ignored by Git. Use these scripts against local folders only.

## Run one candidate

```bash
python scripts/run_v2_candidate.py "candidate folder name or path"
```

The script loads local configuration from `.env` if present.

It writes outputs to:

```text
output/v2_vlm_text_agent/<candidate_name>/
```

## Build a token-safe batch plan

```bash
python scripts/run_v2_batch.py
```

Default behavior does **not** call model APIs. It only creates a local batch plan so you can inspect candidate folders, routing decisions, and existing outputs without spending tokens.

To actually run API/model calls:

```bash
python scripts/run_v2_batch.py --run
```

Use `--run` only when you are ready to spend tokens.

Useful options:

```bash
python scripts/run_v2_batch.py --candidate Michelle
python scripts/run_v2_batch.py --max-candidates 2
python scripts/run_v2_batch.py --run --skip-existing
python scripts/run_v2_batch.py --run --deterministic-aggregate
```

## Rebuild a summary report

```bash
python scripts/build_summary_report.py /path/to/output/v2_vlm_text_agent/<candidate_name>
```

This is useful after changing verification/report formatting. It does not call model APIs.

## Inspect no-OCR routing decisions

```bash
python scripts/route_documents_no_ocr.py
```

This creates a local report showing which files/pages would use PDF text layer, VLM page reading, or human review.

## Private manifests

The manifest-building scripts are for local benchmarking with private files:

```text
build_private_candidate_manifest.py
build_private_eval_manifest.py
```

Their outputs should stay local and should not be committed.
