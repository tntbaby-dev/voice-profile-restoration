from __future__ import annotations

from abc import ABC, abstractmethod

from voice_profile_restoration.models.results import AnalysisResult


class BaseAnalyzer(ABC):
    name = "base"

    @abstractmethod
    def analyze(self, audio, speaker_profile=None) -> AnalysisResult:
        """
        Analyze an audio file and return a structured result.
        """
        raise NotImplementedError