"""
Streaming Sequential Evidence Accumulator for CallGuard.
Implements online fusion of risk evidence over time:
- Uses sequential Bayesian log-odds updating with recency weighting
- Smooths instantaneous spikes so an isolated token does not cause abrupt 0%->100% swings
- Maintains a real, non-flat sequential risk trajectory over conversation time
"""

import numpy as np
from typing import List, Dict, Any, Optional


class StreamingEvidenceAccumulator:
    """
    Accumulates evidence over elapsed call seconds.
    Maintains a rolling belief state using log-odds updating:
    L_t = alpha * L_{t-1} + (1 - alpha) * L_evidence
    where L = log(p / (1 - p))
    """

    def __init__(self, prior_prob: float = 0.15, alpha: float = 0.72):
        self.prior_prob = prior_prob
        self.alpha = alpha  # Memory factor: 0.72 retains historical context while accommodating new evidence
        self.current_log_odds = self._to_log_odds(prior_prob)
        self.trajectory: List[Dict[str, Any]] = []

    def _to_log_odds(self, p: float) -> float:
        p_c = np.clip(p, 1e-4, 1.0 - 1e-4)
        return float(np.log(p_c / (1.0 - p_c)))

    def _from_log_odds(self, lo: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(lo, -12.0, 12.0))))

    def reset(self):
        self.current_log_odds = self._to_log_odds(self.prior_prob)
        self.trajectory = []

    def update(
        self,
        current_second: int,
        instantaneous_text_risk: float,
        instantaneous_audio_risk: float,
        semantic_intent_score: float,
        is_anti_warning: bool = False
    ) -> Dict[str, Any]:
        """
        Updates cumulative belief given current turn evidence and returns trajectory point.
        """
        if is_anti_warning:
            # Protective notice strongly drives belief toward safe prior
            evidence_p = 0.05
        else:
            # Weighted multimodal turn evidence
            evidence_p = (
                instantaneous_text_risk * 0.40 +
                instantaneous_audio_risk * 0.35 +
                semantic_intent_score * 0.25
            )

        evidence_log_odds = self._to_log_odds(evidence_p)

        # Exponentially smoothed Bayesian update
        self.current_log_odds = (
            self.alpha * self.current_log_odds +
            (1.0 - self.alpha) * evidence_log_odds
        )

        fused_risk = self._from_log_odds(self.current_log_odds)
        fused_risk = round(fused_risk, 3)

        point = {
            "t": current_second,
            "risk": round(fused_risk * 100),
            "raw_risk": fused_risk,
            "text_risk": round(instantaneous_text_risk, 3),
            "audio_risk": round(instantaneous_audio_risk, 3),
            "intent_score": round(semantic_intent_score, 3)
        }
        self.trajectory.append(point)

        return {
            "current_fused_risk": fused_risk,
            "trajectory": self.trajectory,
            "latest_point": point
        }
