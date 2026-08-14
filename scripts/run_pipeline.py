"""Run the v2 VLM + text agent pipeline for one applicant folder.

Examples:
    python scripts/run_pipeline.py v2_vlm_text_agent /path/to/applicant
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PIPELINES = {
    "v2_vlm_text_agent": "agentic_doc_intel.pipelines.vlm_text_agent",
}


def load_pipeline(name: str):
    import importlib

    module_path = PIPELINES[name]
    module = importlib.import_module(module_path)
    return module.run_candidate_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", choices=sorted(PIPELINES))
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-base-name", default=None)
    args = parser.parse_args()

    run_candidate_folder = load_pipeline(args.pipeline)
    result = run_candidate_folder(
        input_dir=args.input_dir,
        output_root=args.output_root,
        output_base_name=args.output_base_name,
    )

    printable = {
        "pipeline": result.get("pipeline"),
        "combined_output_file": result.get("combined_output_file"),
        "report_file": result.get("report_file"),
        "logs": result.get("logs", []),
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
