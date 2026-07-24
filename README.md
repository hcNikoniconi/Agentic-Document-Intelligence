# Agentic Document Intelligence

This repository contains the current document-intelligence prototype: PaddleOCR-based document parsing, OpenAI-compatible LLM field extraction, multi-document validation, batch processing, and a Gradio interface.

## Current pipeline

```text
PDF upload
  -> OCR and document classification
  -> template-based field extraction
  -> cross-document validation
  -> combined result and validation report
```

The model runs separately from this repository. In development, the application calls an OpenAI-compatible endpoint such as a vLLM server through `MODEL_BASE_URL`.

## Local setup

1. Create and activate a Python environment.
2. Install `requirements.txt`.
3. Copy `.env.example` to `.env` and provide local values without committing the file.
4. Export the variables from `.env`, then run `python app.py`.

The application defaults to `127.0.0.1` so the UI is not exposed publicly by accident. Use an SSH tunnel when the application or model is running on a remote server.

## Repository policy

Source code and redacted templates belong in Git. Model weights, uploaded documents, OCR output, reports, logs, credentials, and real applicant data do not. The included `.gitignore` enforces these boundaries for new files.

## Next milestone

Create a small, de-identified evaluation set and compare the current extraction baseline with evidence-based validation and targeted retry.
