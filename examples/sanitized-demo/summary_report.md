# Sanitized Demo Candidate Report

This is a hand-written, fully synthetic example. It is not generated from real applicant documents.

## Overview

- Candidate: `DEMO_APPLICANT`
- Files scanned: `3`
- Evidence items: `11`
- Needs retry: `false`
- Needs human review: `false`

## Files Processed

| File | Routed type | Reader | Notes |
|---|---|---|---|
| `passport_demo.pdf` | passport | VLM page reader | Visual identity document |
| `application_form_demo.pdf` | application_form | PDF text layer | Text-heavy form |
| `transcript_demo.pdf` | transcript | VLM page reader | Layout/table-heavy academic record |

## Final Result Highlights

| Field | Final value | Evidence |
|---|---|---|
| passport.name | `ALEX CHEN` | `Name / Nom: ALEX CHEN` |
| passport.passport number | `P00012345` | `Passport No: P00012345` |
| application_form.application id | `APP-DEMO-001` | `Application ID: APP-DEMO-001` |
| transcript.institution name | `Demo International High School` | `Demo International High School Official Transcript` |
| transcript.math score | `Grade 11: 92; Grade 12: 94` | `Mathematics: 92, 94` |
| transcript.passing grade | `60` | `Minimum passing mark: 60` |

## Verification

- Unknown fields: `2`
- Acceptable unknowns: `1`
- Review unknowns: `0`
- Weakly supported fields: `1`
- Unsupported fields needing review: `0`
- Hard conflicts: `0`
- Soft conflicts: `0`
- Certificate identity mismatches: `0`
- Needs human review: `false`

### Acceptable Unknowns

- `english_language.ielts/toefl/pte score`: No English test document was provided.

### Weakly Supported Fields

- `application_form.Have you ever been convicted of a crime?` = `No`. Boolean value is accepted but would be stronger with a direct quote in production evidence.

### Source Consistency Checks

| Document type | Left source | Right source | Compatible | Shared signals | Reason |
|---|---|---|---|---|---|
| transcript | `transcript_demo.pdf` | `application_form_demo.pdf` | `true` | name: Alex Chen ≈ Alex Chen | The transcript and application form refer to the same applicant name. |

## Recommended Next Action

- Result is ready for benchmark-style comparison under the demo policy.
