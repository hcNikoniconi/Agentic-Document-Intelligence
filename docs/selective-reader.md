# Selective reader design note

This is a future optimization direction, not the current baseline.

The current pipeline processes a candidate folder by running OCR and LLM extraction on each recognized PDF, then writing one combined result and one validation report.

For short documents, full OCR is acceptable. For long PDFs, such as documents with dozens or hundreds of pages, full OCR plus sending all extracted text to the LLM can become slow, expensive, noisy, and less accurate.

The selective reader idea is:

```text
PDF
  -> page-level text or low-cost OCR index
  -> candidate page selection
  -> high-quality OCR only for relevant pages
  -> LLM extraction from focused evidence
```

The benchmark should still evaluate the final end-to-end result. Selective reading is an alternative implementation strategy to compare against the full-OCR baseline later.

Future comparison:

```text
Full baseline:
candidate folder -> full OCR -> LLM -> combined result

Selective baseline:
candidate folder -> page routing -> focused OCR -> LLM -> combined result
```

Compare:

- field accuracy
- missing-field rate
- total latency
- OCR pages processed
- LLM input length
- failure cases

