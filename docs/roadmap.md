# Roadmap

The project has reached a v2 MVP: one candidate folder can be processed end-to-end through document routing, evidence extraction, aggregation, verification, and report generation.

The next goal is not to keep adding random features. The next goal is to make the MVP easy to understand, evaluate, and eventually productize.

## Current: v2 MVP

```text
Candidate folder
  -> scan all files
  -> choose reader per file/page
  -> extract field-level evidence
  -> aggregate candidate result
  -> verify conflicts and source consistency
  -> write report
```

Implemented outputs:

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

## Milestone 1: Project packaging

Purpose: make the repository readable as a serious project.

Tasks:

- rewrite README around the v2 MVP;
- add architecture documentation;
- add sanitized example input/output;
- document what is private and never committed;
- document model-provider setup.

Success condition:

> A reader can open GitHub and understand the project in under one minute.

## Milestone 2: Small evaluation set

Purpose: make improvement measurable without spending too many tokens.

Tasks:

- create a small sanitized or local-only eval manifest;
- run a few representative candidates manually;
- compare final candidate-level fields;
- track unknowns, conflicts, and human-review cases;
- keep private ground truth out of Git.

Success condition:

> The project can say which fields improved or regressed after a change.

## Milestone 3: Selective reading experiment

Purpose: reduce cost for long documents.

Compare:

```text
full document/pages -> VLM
vs
text/page routing -> selected pages -> VLM
```

Measure:

- final field accuracy;
- number of pages sent to VLM;
- estimated token/image cost;
- latency;
- missed-field rate;
- conflict rate.

Success condition:

> The project can show whether selective reading saves cost without hurting accuracy too much.

## Milestone 4: Backend/API

Purpose: separate product logic from the local Gradio demo.

Planned API shape:

```text
POST /candidates
POST /candidates/{id}/files
POST /candidates/{id}/runs
GET  /runs/{id}
GET  /runs/{id}/report
```

The backend should support two model access modes:

- user-provided external API key;
- self-hosted/local model endpoint.

Success condition:

> The extraction pipeline can be triggered without the Gradio UI.

## Milestone 5: Frontend

Purpose: turn the MVP into a usable product demo.

Planned UI:

- upload one candidate folder;
- select model provider;
- provide API key locally/session-only;
- show routing decisions;
- show extracted evidence;
- show verification report;
- download final JSON/report.

Success condition:

> A non-technical user can upload files, run extraction, and inspect the report.

## Deferred

Do not prioritize these until the core loop is stable:

- public deployment;
- authentication;
- database persistence;
- multi-user workspace;
- large-scale benchmark runs;
- agent memory;
- complex multi-agent orchestration.
