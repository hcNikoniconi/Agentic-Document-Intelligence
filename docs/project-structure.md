# Project structure

Agentic Document Intelligence is being organized in layers. Each layer owns one kind of responsibility, so future Agent, evaluation, API, and serving work can be added without rewriting the working prototype.

## Current layout

```text
.
├── app.py
├── extract_v_0_5.py
├── validator.py
├── src/
│   └── agentic_doc_intel/
│       └── model_client.py
├── templates/
├── evals/
├── docs/
└── scripts/
```

## Why this structure exists

`app.py` is the user interface layer. It should handle uploads, button clicks, preview text, and downloadable files. It should not know model provider details.

`extract_v_0_5.py` is the current extraction pipeline. It owns OCR, template matching, prompt construction, parsing, and saving results.

`validator.py` owns cross-document checks and report generation. Validation should stay separate because it will later become the verifier layer of the Agent workflow.

`src/agentic_doc_intel/model_client.py` owns all OpenAI-compatible model access. Local vLLM, remote vLLM, and external APIs should all be switched through environment variables rather than business logic changes.

`templates/` contains document schemas. These are product data, not model code.

`evals/` will contain de-identified test cases, expected outputs, and metrics. Real applicant documents should not be committed.

`docs/` explains architecture and decisions. If a future reader cannot understand why the project is shaped this way, the project will become hard to maintain.

`scripts/` is reserved for repeatable command-line tasks such as running batches, creating reports, or launching local services.

## Rule of thumb

When adding a new feature, ask: "Which layer owns this?"

- UI behavior goes in `app.py`.
- Extraction logic goes in the extraction pipeline.
- Model provider setup goes in `model_client.py`.
- Validation and consistency checks go in `validator.py`.
- Test cases and metrics go in `evals/`.
- Human-readable design notes go in `docs/`.

