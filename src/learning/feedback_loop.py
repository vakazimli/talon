"""Process user feedback and integrate it into source scoring."""

import logging

from src.learning.source_evaluator import SourceEvaluator

logger = logging.getLogger(__name__)


class FeedbackLoop:
    def __init__(self):
        self.evaluator = SourceEvaluator()

    def process_feedback(self, alert_id: int, feedback_type: str) -> None:
        """Apply feedback-based adjustments to source reliability scores."""
        logger.info("Processing feedback '%s' for alert #%d", feedback_type, alert_id)
        self.evaluator.apply_feedback_multiplier(alert_id, feedback_type)
