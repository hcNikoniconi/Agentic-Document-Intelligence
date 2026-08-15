# Agentic Document Intelligence

Agentic Document Intelligence is a candidate-level document understanding system for admissions-style application files. It reads a folder of applicant documents, extracts field-level evidence, aggregates the evidence into a structured candidate profile, and verifies missing fields, conflicts, and source consistency.

This repository now focuses on the **v2 MVP**: a VLM + text-model agent workflow. Earlier OCR and direct-VLM experiments are treated as local baselines, not the main product path.

## What the system does

```text
Candidate folder
  -> scan all files
  -> route each file/page to the right reader
  -> extract field-level evidence with source and page references
  -> aggregate evidence into one candidate result
  -> verify conflicts, missing fields, weak evidence, and source consistency
  -> generate machine-readable outputs and a human-readable report
```

The project is designed around a practical problem: real application packets contain passports, application forms, transcripts, certificates, recommendation letters, personal statements, and miscellaneous PDFs. A single one-shot prompt is fragile, expensive, and hard to debug. This system keeps intermediate evidence and verification traces so failures can be located and improved.

## Core idea

Instead of asking a model to directly produce the final answer from all documents, the pipeline separates the work:

| Stage | Purpose |
|---|---|
| Document routing | Decide which files/pages are likely useful |
| Evidence extraction | Read documents and emit field-level evidence |
| Aggregation | Merge evidence across files into one candidate result |
| Verification | Detect conflicts, missing values, unsupported fields, and source mismatch |
| Reporting | Produce JSON/JSONL traces plus a readable summary |

This makes the system easier to evaluate and iterate than a plain OCR/VLM demo.

## Current MVP status

Implemented:

- candidate-folder processing
- PDF/image rendering for VLM calls
- text-layer reading for text-heavy PDFs
- VLM page reader
- field-level evidence records
- text-model aggregation
- deterministic fallback aggregation
- schema repair for model output drift
- verification policy for conflicts, unknown values, weak support, and source consistency
- token-safe batch planner
- human-readable summary report

Not yet the focus:

- public web deployment
- polished frontend/backend rewrite
- large-scale batch runs
- cost-heavy benchmark runs
- user account/authentication system

## Repository structure

```text
.
├── app.py                         # Gradio entry point
├── app_v2_agent.py                # v2 MVP UI/runner entry point
├── src/agentic_doc_intel/
│   ├── pipelines/
│   │   ├── vlm_direct.py          # direct VLM pipeline utilities
│   │   └── vlm_text_agent.py      # active v2 agent pipeline
│   ├── routing/
│   │   └── reading_policy.py      # no-OCR reader/page routing policy
│   ├── verification/
│   │   └── policy.py              # conflict/unknown/source checks
│   ├── schemas/                   # shared candidate result schema
│   ├── document_rendering.py      # PDF/image rendering helpers
│   ├── model_client.py            # OpenAI-compatible model client
│   └── template_registry.py       # document schema loading
├── templates/                     # extraction schemas for supported document types
├── scripts/                       # repeatable local workflows
├── evals/                         # evaluation helpers and sanitized examples
├── docs/                          # design notes
├── requirements.txt
└── .env.example
```

## Private-data policy

Only code, templates, scripts, docs, and sanitized examples belong in GitHub.

Do not commit:

- API keys or `.env`
- candidate documents
- generated extraction results
- uploads
- local benchmark outputs
- model weights
- logs or caches

The `.gitignore` is configured to keep these local.

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create local configuration:

```bash
cp .env.example .env
```

Then fill in your local model endpoint:

```dotenv
MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_API_KEY=your-local-key
MODEL_NAME=qwen3.7-flash
```

The model service only needs to be OpenAI-compatible. It can be:

- a local/self-hosted vLLM server;
- a remote server accessed through SSH tunnel;
- an external provider such as DashScope compatible mode.

## Run one candidate

```bash
python scripts/run_v2_candidate.py "candidate folder name or path"
```

Outputs are written locally under:

```text
output/v2_vlm_text_agent/<candidate_name>/
```

Typical output files:

```text
document_manifest.json
document_evidence.jsonl
candidate_result.json
verification_report.json
summary_report.md
trace.json
raw_model_outputs.json
aggregator_raw_output.json
```

## Token-safe batch planning

The batch runner is safe by default:

```bash
python scripts/run_v2_batch.py
```

Without `--run`, it only creates a local plan and does not call model APIs.

To actually run model calls:

```bash
python scripts/run_v2_batch.py --run
```

Use this only when you are ready to spend API tokens.

## Project direction

The next milestones are:

1. improve README/demo packaging with a sanitized example;
2. add a small end-to-end evaluation set;
3. compare full-document VLM vs selective reading on cost and accuracy;
4. add a FastAPI backend;
5. build a cleaner frontend for upload, model selection, and report review.

## Resume framing

> Built an Agentic Document Intelligence MVP that processes candidate application packets end-to-end: document routing, VLM/text evidence extraction, cross-document aggregation, conflict verification, source consistency checks, and token-safe batch planning.
