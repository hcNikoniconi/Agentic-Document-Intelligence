"""v2 placeholder app.

v2 is reserved for the VLM + text verifier/agent workflow. It intentionally
does not include the OCR baseline.
"""

from __future__ import annotations

import gradio as gr


with gr.Blocks(title="Agentic Document Intelligence - v2 Agent") as demo:
    gr.Markdown("## Agentic Document Intelligence - v2 VLM + Text Agent")
    gr.Markdown(
        "v2 is the next version: VLM extracts page-level evidence, then a text "
        "model verifies, normalizes, aggregates, and retries uncertain fields."
    )
    gr.Markdown(
        "This folder is intentionally separated from v0 OCR and v1 VLM-direct. "
        "Implementation starts after v1 benchmark results are available."
    )


if __name__ == "__main__":
    demo.launch()

