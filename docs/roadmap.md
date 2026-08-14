# Roadmap: from OCR baseline to multimodal document intelligence

This project should not stay as an OCR-only extraction demo. The OCR pipeline is
useful, but mainly as a baseline. The higher-value direction is to compare
several document intelligence pipelines under one benchmark.

## Version plan

### v0: OCR + text LLM baseline

```text
Applicant folder
  -> OCR / text extraction
  -> text LLM field extraction
  -> combined candidate result
  -> benchmark accuracy
```

Purpose:

- Establish the first working end-to-end baseline.
- Produce a measurable field-level accuracy number.
- Keep the old workflow available for comparison.

Current status:

- Implemented by `extract_v_0_5.py`.
- Wrapped by `src/agentic_doc_intel/pipelines/ocr_text_baseline.py`.
- Current manual benchmark baseline: `0.761` field accuracy on 7 candidates.

### v1: VLM direct extraction

```text
Applicant folder
  -> render PDF/pages or load images
  -> multimodal model reads document pages directly
  -> template-based JSON extraction
  -> same combined candidate result format
  -> benchmark accuracy
```

Purpose:

- Remove OCR as the first bottleneck.
- Test whether a VLM can better understand layout-heavy documents such as
  transcripts, diplomas, certificates, and forms.
- Compare directly against v0 using the same ground truth and metrics.

Implementation target:

- `src/agentic_doc_intel/pipelines/vlm_direct.py`
- Add a multimodal model client.
- Add PDF/image rendering utilities.
- Reuse `templates/` as the extraction schema.

First runnable command:

```bash
python scripts/run_pipeline.py \
  v1_vlm_direct \
  /path/to/one-applicant-folder \
  --output-root evals/results/manual_candidate_outputs_v1_vlm
```

Required model settings:

```dotenv
MODEL_BASE_URL=http://your-vlm-endpoint/v1
MODEL_API_KEY=replace-me
MODEL_NAME=your-vlm-model-name
VLM_MAX_PAGES_PER_FILE=4
VLM_MAX_TOKENS=4096
VLM_MIN_PIXELS=
VLM_MAX_PIXELS=
```

Qwen / Alibaba Cloud Model Studio example:

```dotenv
MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_API_KEY=your-dashscope-api-key
MODEL_NAME=qwen-vl-max
MODEL_TIMEOUT_SECONDS=180
VLM_MAX_PAGES_PER_FILE=4
VLM_MAX_TOKENS=4096
```

Start with one applicant folder before running all seven. The first goal is to
check whether the VLM endpoint accepts PDF-rendered page images and returns
valid JSON in the same schema as v0.

### v2: VLM + text verifier/aggregator

```text
Applicant folder
  -> VLM extracts page/document-level evidence
  -> text model aggregates all documents for one candidate
  -> verifier checks missing fields, conflicts, unknown values, and evidence
  -> final candidate-level result
  -> benchmark accuracy
```

Purpose:

- Make the project agentic rather than just one-shot extraction.
- Handle multi-document consistency:
  - passport name vs application name
  - transcript school vs diploma school
  - English test name vs passport name
  - missing or conflicting graduation information
- Improve long-document handling by not throwing hundreds of pages into one
  prompt without structure.

Implementation target:

- `src/agentic_doc_intel/pipelines/vlm_text_agent.py`
- Add field-level evidence.
- Add conflict resolution and verification.
- Add targeted retry only for failed or uncertain fields.

### v3: Long-document selective reading

```text
Large document set
  -> page/document routing
  -> only relevant pages go to expensive extraction
  -> VLM/text extraction
  -> verifier
  -> final benchmark
```

Purpose:

- Avoid sending hundreds of pages into one model call.
- Improve speed and cost.
- Make the system closer to real production document processing.

## Benchmark rule

Every version must produce the same final output shape:

```text
#passport
name:
passport number:

#transcript
student name:
math score:
english score:

#diploma_certificate
school name:
graduation year:
```

This makes the benchmark fair:

```text
v0 OCR + text LLM
vs
v1 VLM direct extraction
vs
v2 VLM + text verifier
```

## Resume framing

The goal is not "OCR extraction." The stronger framing is:

> Built a multi-stage Agentic Document Intelligence system for applicant
> verification, comparing OCR+LLM, VLM direct extraction, and VLM+LLM
> verification pipelines under a schema-based end-to-end benchmark.
