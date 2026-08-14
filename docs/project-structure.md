# Project structure

Agentic Document Intelligence is organized around pipeline comparison. Each
layer owns one responsibility, so OCR, VLM, Agent verification, evaluation, API,
and deployment work can evolve without turning the project into one giant
script.

## Current layout

```text
.
├── app.py
├── extract_v_0_5.py
├── validator.py
├── src/
│   └── agentic_doc_intel/
│       ├── pipelines/
│       │   ├── ocr_text_baseline.py
│       │   ├── vlm_direct.py
│       │   └── vlm_text_agent.py
│       ├── schemas/
│       │   └── candidate_result.py
│       ├── evaluators/
│       └── model_client.py
├── templates/
├── evals/
├── docs/
└── scripts/
```

## Why this structure exists

`app.py` is the user interface layer. It should handle uploads, button clicks,
preview text, and downloadable files. It should not know model provider details.

`extract_v_0_5.py` is the current working v0 baseline. It owns OCR, template
matching, prompt construction, parsing, and saving results. It is intentionally
kept working instead of being rewritten immediately.

`src/agentic_doc_intel/pipelines/` contains versioned pipeline entry points:

- `ocr_text_baseline.py`: wraps the current v0 OCR + text LLM workflow.
- `vlm_direct.py`: planned v1 pipeline for direct multimodal extraction.
- `vlm_text_agent.py`: planned v2 pipeline for VLM extraction plus text-model
  verification and aggregation.

This makes the project readable as an experiment:

```text
v0 OCR baseline
vs
v1 VLM direct extraction
vs
v2 VLM + text verifier
```

`validator.py` owns cross-document checks and report generation. Validation
should stay separate because it will later become the verifier layer of the
Agent workflow.

`src/agentic_doc_intel/model_client.py` owns all OpenAI-compatible model access.
Local vLLM, remote vLLM, and external APIs should all be switched through
environment variables rather than business logic changes.

`src/agentic_doc_intel/schemas/` contains shared result shapes. The key idea is
that every pipeline should eventually output the same candidate-level prediction
format, so the benchmark can compare them fairly.

`src/agentic_doc_intel/evaluators/` is reserved for reusable evaluation logic
that does not belong to one specific benchmark script.

`templates/` contains document schemas. These are product data, not model code.

`evals/` contains benchmark cases and metrics. Real applicant documents,
private ground truth, and generated reports should not be committed.

`docs/` explains architecture and decisions. If a future reader cannot
understand why the project is shaped this way, the project will become hard to
maintain.

`scripts/` is reserved for repeatable command-line tasks such as running
pipelines, importing manual results, creating reports, or launching local
services.

## Rule of thumb

When adding a new feature, ask: "Which layer owns this?"

- UI behavior goes in `app.py`.
- Extraction logic goes in the corresponding versioned pipeline.
- Model provider setup goes in `model_client.py`.
- Shared output contracts go in `schemas/`.
- Validation and consistency checks go in `validator.py`.
- Test cases and metrics go in `evals/`.
- Human-readable design notes go in `docs/`.

## Why this matters for the resume

A weaker project structure says:

```text
I wrote an OCR extraction script.
```

A stronger project structure says:

```text
I built a benchmarked document intelligence system and compared multiple
pipelines: OCR+LLM, VLM direct extraction, and VLM+LLM verification.
```

