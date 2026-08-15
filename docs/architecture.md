# Architecture

Agentic Document Intelligence is organized as a candidate-level pipeline rather than a one-file extraction script.

## High-level flow

```text
Candidate folder
  |
  v
Document manifest
  |
  v
Reading policy
  |-------------------------|
  v                         v
PDF text-layer reader       VLM page reader
  |                         |
  |-------------------------|
  v
Field-level evidence
  |
  v
Text-model aggregator
  |
  v
Candidate result
  |
  v
Verification policy
  |
  v
Summary report + machine-readable artifacts
```

## Why evidence comes before the final result

A direct extraction system usually asks the model for the final JSON immediately.

This project instead asks readers to produce evidence first:

```json
{
  "document_type": "transcript",
  "source_file": "transcript.pdf",
  "page": 2,
  "field": "math score",
  "value": "Grade 11: 92; Grade 12: 94",
  "evidence": "Mathematics: 92, 94",
  "reader": "vlm_page_reader"
}
```

That makes the system easier to debug:

- if a value is wrong, inspect the evidence;
- if evidence is missing, inspect routing/page selection;
- if two files disagree, inspect source consistency;
- if the final answer is `unknown`, check whether evidence existed but aggregation failed.

## Reader strategy

The MVP avoids making OCR the default path.

Current readers:

| Reader | Use case |
|---|---|
| PDF text layer | Text-heavy forms or documents with reliable embedded text |
| VLM page reader | Visual documents, tables, stamps, signatures, passports, transcripts, certificates |
| Human review | Unsupported, missing, or ambiguous cases |

This keeps token usage lower than sending every page to a VLM, while still allowing VLMs to handle layout-heavy pages.

## Verification strategy

Verification combines deterministic rules with model-generated evidence.

The verifier checks:

- missing fields;
- unknown values;
- hard conflicts;
- soft/explainable conflicts;
- weakly supported fields;
- certificate identity consistency;
- source consistency across complementary documents.

This is intentionally not a fully autonomous multi-agent system yet. The current MVP is a controlled agentic workflow: model calls are used for reading and aggregation, while deterministic checks make the result auditable.

## Output artifacts

Each run produces:

| File | Purpose |
|---|---|
| `document_manifest.json` | Files discovered and planned reading decisions |
| `document_evidence.jsonl` | Field-level evidence extracted from files |
| `candidate_result.json` | Final structured candidate result |
| `combined_result.txt` | Human-friendly text version of the final result |
| `verification_report.json` | Machine-readable verification output |
| `summary_report.md` | Human-readable report |
| `trace.json` | Execution trace for debugging |
| `raw_model_outputs.json` | Raw reader outputs for debugging |
| `aggregator_raw_output.json` | Raw aggregation output for debugging |

## Next architectural step

The next major improvement is selective reading:

```text
cheap page scan
  -> identify candidate pages
  -> send only relevant pages to VLM
  -> compare accuracy/cost against full VLM reading
```

This is the bridge from MVP to a production-like document intelligence system.
