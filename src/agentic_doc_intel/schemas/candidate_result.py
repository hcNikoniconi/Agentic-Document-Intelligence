"""Common candidate-level result shape.

The benchmark should not care whether a result came from OCR, a VLM, or a
two-stage Agent pipeline. It only needs a stable final prediction dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateResult:
    candidate_id: str
    pipeline: str
    prediction: dict[str, str] = field(default_factory=dict)
    combined_output_file: str | None = None
    report_file: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.candidate_id,
            "pipeline": self.pipeline,
            "prediction": self.prediction,
            "combined_output_file": self.combined_output_file,
            "report_file": self.report_file,
            "error": self.error,
            "metadata": self.metadata,
        }

