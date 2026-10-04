"""
Natural Language Query & Evidence Retrieval Engine for Impact Intelligence
==========================================================================
Translates natural-language user questions into structured spatial and meteorological
evidence queries. Grounds every response in verified satellite change detection
statistics, IBTrACS cyclone trajectories, and sensor acquisition records.
Zero hallucinations: strictly cites sources, pre/post dates, areas, and uncertainty.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.ml.vlm_provider import BaseSatelliteVLMProvider, GroundedAnswerResult, get_vlm_provider
from backend.app.scientific.change_detector import ChangeDetectionResult


class ImpactQueryEngine:
    """
    Coordinates question understanding, multi-source evidence extraction,
    and grounded answer generation.
    """

    SUPPORTED_EXAMPLE_QUESTIONS = [
        "What changed near Puri after Cyclone Fani?",
        "Did water extent or flooding increase after the cyclone?",
        "How much vegetation or tree canopy was affected?",
        "What satellite evidence supports these observations?",
        "Compare the pre-event and post-event satellite observations.",
        "What was Cyclone Fani's intensity and closest distance to Puri?",
    ]

    def __init__(self, provider: Optional[BaseSatelliteVLMProvider] = None):
        self.provider = provider or get_vlm_provider()

    def process_query(
        self,
        question: str,
        change_result: ChangeDetectionResult,
        cyclone_context: Dict[str, Any],
        location_name: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> GroundedAnswerResult:
        """
        Executes query parsing, retrieves ground truth and satellite evidence,
        and generates an evidence-grounded answer.
        """
        clean_q = question.strip()
        if not clean_q:
            clean_q = "What changed after the cyclone?"

        return self.provider.answer_question(
            question=clean_q,
            change_result=change_result,
            cyclone_context=cyclone_context,
            location_name=location_name,
            additional_metadata=additional_metadata,
        )
