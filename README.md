# Agentic Document Intelligence

Agentic Document Intelligence is a document-processing prototype for extracting structured information from admissions-related PDF files. It combines PaddleOCR-based parsing, OpenAI-compatible LLM field extraction, multi-document validation, batch processing, and a Gradio interface.

## Current pipeline

```text
PDF upload
  -> OCR and document classification
  -> template-based field extraction
  -> cross-document validation
  -> combined result and validation report
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
├── extract_v_0_5.py               # Current extraction pipeline
├── validator.py                   # Cross-document validation and reports
├── src/agentic_doc_intel/         # Shared reusable application code
├── templates/                     # Document extraction schemas
├── evals/                         # De-identified evaluation cases
├── docs/                          # Architecture notes
└── scripts/                       # Repeatable command-line workflows
```

See `docs/project-structure.md` for the reasoning behind this layout.

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

Create a small, de-identified evaluation set and compare the current extraction baseline with evidence-based validation and targeted retry.
