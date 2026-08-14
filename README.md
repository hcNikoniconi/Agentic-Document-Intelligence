# Agentic Document Intelligence

Agentic Document Intelligence is a multi-stage document extraction and
verification project for admissions-related applicant files. The project starts
with an OCR + text LLM baseline, then evolves toward VLM direct extraction and a
VLM + text-model verifier/aggregator.

The goal is not just to extract fields from PDFs. The goal is to compare
different document intelligence pipelines under the same candidate-level
benchmark.

## Pipeline versions

### v0: OCR + text LLM baseline

```text
Applicant folder
  -> OCR and document classification
  -> text LLM field extraction
  -> cross-document validation
  -> combined candidate result
  -> benchmark accuracy
```

This is the current working baseline. It is still useful because it gives the
project a measurable starting point.

### v1: VLM direct extraction

```text
Applicant folder
  -> original PDF pages / images
  -> multimodal model extraction
  -> same combined candidate result format
  -> benchmark accuracy
```

This removes OCR as the first bottleneck and tests whether a VLM can better
understand layout-heavy documents such as transcripts and diplomas.

### v2: VLM + text verifier/aggregator

```text
Applicant folder
  -> VLM page/document evidence extraction
  -> text model aggregation and normalization
  -> verifier for conflicts, missing fields, and unknown values
  -> final candidate-level result
  -> benchmark accuracy
```

The model runs separately from this repository. The same application supports two model access modes through `MODEL_BASE_URL`:

- Local or self-hosted model service, such as vLLM running on the same machine or a private server.
- External API service, as long as it exposes an OpenAI-compatible chat completions endpoint.

This keeps the code path shared while allowing local deployment and API-based usage to evolve in parallel.

## Local setup

1. Create and activate a Python environment.
2. Install `requirements.txt`.
3. Copy `.env.example` to `.env` and provide local values without committing the file.
4. Export the variables from `.env`, then run `python app.py`.

The application defaults to `127.0.0.1` so the UI is not exposed publicly by accident. Use an SSH tunnel when the application or model is running on a remote server.

## Project structure

```text
.
├── app.py                         # Gradio UI
├── extract_v_0_5.py               # Working v0 OCR + text LLM baseline
├── validator.py                   # Cross-document validation and reports
├── src/agentic_doc_intel/
│   ├── pipelines/                 # v0/v1/v2 pipeline entry points
│   ├── schemas/                   # Shared result shapes
│   ├── evaluators/                # Future shared evaluation helpers
│   └── model_client.py            # OpenAI-compatible model access
├── templates/                     # Document extraction schemas
├── evals/                         # De-identified evaluation cases
├── docs/                          # Architecture notes
└── scripts/                       # Repeatable command-line workflows
```

See `docs/project-structure.md` for the reasoning behind this layout.
See `docs/roadmap.md` for the v0 -> v1 -> v2 project plan.
See `docs/model-providers.md` for the model-provider strategy.

## Model configuration examples

Self-hosted vLLM on the same machine:

```dotenv
MODEL_BASE_URL=http://127.0.0.1:8000/v1
MODEL_API_KEY=replace-me
MODEL_NAME=qwen3-8b
MODEL_TIMEOUT_SECONDS=120
```

Remote server accessed through an SSH tunnel:

```dotenv
MODEL_BASE_URL=http://127.0.0.1:8000/v1
MODEL_API_KEY=replace-me
MODEL_NAME=/path/to/server/model
MODEL_TIMEOUT_SECONDS=120
```

External OpenAI-compatible API:

```dotenv
MODEL_BASE_URL=https://example.com/v1
MODEL_API_KEY=replace-me
MODEL_NAME=provider-model-name
MODEL_TIMEOUT_SECONDS=120
```

## Repository policy

Source code and redacted templates belong in Git. Model weights, uploaded documents, OCR output, reports, logs, credentials, and real applicant data do not. The included `.gitignore` enforces these boundaries for new files.

## Next milestone

Freeze v0 as the OCR + text LLM baseline, then implement v1 VLM direct
extraction while keeping the same final output format and benchmark.
