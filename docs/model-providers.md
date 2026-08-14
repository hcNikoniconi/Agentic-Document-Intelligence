# Model provider strategy

The project should not depend on one fixed model provider. The product value is
in the document workflow:

```text
document routing
-> schema-guided extraction
-> evidence-aware verification
-> benchmarked accuracy
```

The model should be pluggable.

## Default model choice

For the first VLM direct extraction experiment, use:

```text
qwen3.6-flash
```

Reason:

- It is cheaper and faster than larger flagship models.
- The current goal is not to prove that the strongest model can extract fields.
- The current goal is to test whether the workflow helps a reasonable VLM
  produce stable structured outputs.

Use a stronger model only as an upper-bound comparison:

```text
qwen3.7-plus
```

Use OCR-specialized models as a separate baseline:

```text
qwen-vl-ocr
```

## UI design

The Gradio app supports runtime model configuration:

```text
Provider
Base URL
Model name
API Key
Pipeline
```

The API key is only used at runtime. It should not be written to `.env`, logs,
Git, or generated reports.

## Provider examples

### Qwen / DashScope

```dotenv
MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen3.6-flash
```

### Local vLLM

```dotenv
MODEL_BASE_URL=http://127.0.0.1:8000/v1
MODEL_NAME=qwen3-8b
```

### Custom OpenAI-compatible endpoint

```dotenv
MODEL_BASE_URL=https://provider.example.com/v1
MODEL_NAME=provider-model-name
```

## Benchmark principle

Do not compare models by vibes. Compare them through the same candidate-level
benchmark:

```text
v0 OCR + text LLM
v1 VLM direct extraction
v2 VLM + text verifier
```

The final output schema must stay the same across providers and models.

