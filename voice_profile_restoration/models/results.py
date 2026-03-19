from __future__ import annotations

from datetime import datetime, UTC
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    info = "info"
    warn = "warn"
    critical = "critical"


class IssueFlag(BaseModel):
    analyzer: str
    severity: Severity
    message: str
    start_sec: float | None = None
    end_sec: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class CorrectionMove(BaseModel):
    filter_type: str
    freq_hz: float | None = None
    gain_db: float | None = None
    q: float | None = None
    freq_range_hz: tuple[float, float] | None = None
    intensity: str | None = None
    reason: str


class AnalysisResult(BaseModel):
    analyzer: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    flags: list[IssueFlag] = Field(default_factory=list)
    recommendations: list[CorrectionMove] = Field(default_factory=list)


class AudioData(BaseModel):
    samples: Any
    sample_rate: int = Field(..., gt=0)
    duration_sec: float = Field(..., ge=0)


class SpeakerProfile(BaseModel):
    speaker_id: str
    source_files: list[str] = Field(default_factory=list)

    avg_band_ratios: dict[str, float] = Field(default_factory=dict)
    spectral_tilt: float | None = None

    sibilance_reference: dict[str, Any] = Field(default_factory=dict)
    plosive_reference: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FileReport(BaseModel):
    file_path: str
    sample_rate: int
    duration_sec: float

    profile_match_score_before: float | None = None
    profile_match_score_after: float | None = None

    results: list[AnalysisResult] = Field(default_factory=list)
    score: float = 100.0

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))