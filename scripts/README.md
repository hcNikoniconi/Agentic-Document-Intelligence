# Scripts

Put repeatable command-line workflows here, such as batch extraction, evaluation runs, report generation, and local service launch helpers.

Keep one-off experiments out of this folder unless they become reusable.

## Pipeline runner

Use `run_pipeline.py` as the unified entry point for versioned extraction
pipelines:

```bash
python scripts/run_pipeline.py v0_ocr_text_baseline /path/to/applicant-folder
```

Planned future commands:

```bash
python scripts/run_pipeline.py v1_vlm_direct /path/to/applicant-folder
python scripts/run_pipeline.py v2_vlm_text_agent /path/to/applicant-folder
```

The point is to keep v0, v1, and v2 comparable instead of creating unrelated
scripts for every experiment.

For v1 VLM direct extraction, configure a multimodal OpenAI-compatible endpoint
first:

```dotenv
MODEL_BASE_URL=http://your-vlm-endpoint/v1
MODEL_API_KEY=replace-me
MODEL_NAME=your-vlm-model-name
VLM_MAX_PAGES_PER_FILE=4
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

Then run one applicant first:

```bash
python scripts/run_pipeline.py \
  v1_vlm_direct \
  /path/to/one-applicant-folder \
  --output-root evals/results/manual_candidate_outputs_v1_vlm
```
