# Project Structure

This repository is intentionally centered on the current v2 MVP instead of exposing every historical experiment.

The active product kernel is:

```text
candidate documents
  -> document routing
  -> evidence extraction
  -> evidence aggregation
  -> verification
  -> candidate report
```

## Layout

```text
.
├── app.py
├── app_v2_agent.py
├── src/
│   └── agentic_doc_intel/
│       ├── pipelines/
│       │   ├── vlm_direct.py
│       │   └── vlm_text_agent.py
│       ├── routing/
│       │   └── reading_policy.py
│       ├── verification/
│       │   └── policy.py
│       ├── schemas/
│       │   └── candidate_result.py
│       ├── document_rendering.py
│       ├── model_client.py
│       └── template_registry.py
├── templates/
├── scripts/
├── evals/
├── docs/
└── requirements.txt
```

## Responsibilities

### `app.py` and `app_v2_agent.py`

UI and local demo entry points. These files should stay thin. They can collect inputs, display status, and expose downloadable outputs, but core extraction logic should live under `src/`.

### `src/agentic_doc_intel/pipelines/vlm_text_agent.py`

The active v2 pipeline. It owns the candidate-level flow:

```text
scan folder
  -> build document manifest
  -> run readers
  -> collect evidence
  -> aggregate result
  -> verify result
  -> write outputs
```

This is the main file to improve when adding product intelligence.

### `src/agentic_doc_intel/routing/reading_policy.py`

Chooses how each file/page should be read. The current policy avoids traditional OCR as the main path and uses:

- PDF text layer for text-heavy documents;
- VLM page reader for visual/layout-heavy documents;
- human review for unsupported or low-confidence cases.

### `src/agentic_doc_intel/verification/policy.py`

Encodes deterministic verification checks that should not depend entirely on model judgment:

- hard conflicts;
- soft conflicts;
- acceptable unknowns;
- review-needed unknowns;
- weakly supported fields;
- certificate identity consistency;
- source consistency between complementary files.

The point is not to replace the model with rules. The point is to make model outputs auditable.

### `src/agentic_doc_intel/model_client.py`

The model access layer. It should hide whether the model is local vLLM, remote vLLM, or an external OpenAI-compatible API.

### `templates/`

Document schemas. These define what fields the system expects for passports, application forms, transcripts, certificates, and English-language documents.

### `scripts/`

Repeatable local workflows:

- run one candidate;
- build summary reports;
- create token-safe batch plans;
- inspect routing decisions;
- build private manifests without committing private data.

### `evals/`

Evaluation helpers and sanitized examples. Real candidate files, manual outputs, and private benchmark results should stay local and ignored by Git.

## Design rule

When adding a feature, ask which layer owns it:

| Feature | Location |
|---|---|
| UI behavior | `app.py` / `app_v2_agent.py` |
| candidate pipeline | `pipelines/vlm_text_agent.py` |
| page/file choice | `routing/reading_policy.py` |
| model call | `model_client.py` |
| output contract | `schemas/` |
| conflict/missing checks | `verification/policy.py` |
| repeatable command | `scripts/` |
| explanation/design note | `docs/` |

This keeps the project from collapsing into one giant demo script.
