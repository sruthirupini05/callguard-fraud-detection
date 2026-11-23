"""
Calibration and Uncertainty Estimation Module for CallGuard.
Implements:
1. Temperature Scaling Logit Calibration:
   calibrated_p = sigmoid(logit / T)
2. Epistemic & Aleatoric Uncertainty Estimation:
   Produces confidence intervals [lower_bound, upper_bound]
3. Multi-Tier Verdict Classification with explicit 'UNCERTAIN' range (40%-60%):
   - 'LOW RISK': < 0.40
   - 'UNCERTAIN / INCONCLUSIVE': 0.40 <= p <= 0.60
   - 'HIGH RISK': > 0.60
"""

import numpy as np
from typing import Dict, Tuple, Any


class ProbabilityCalibrator:
    """
    Calibrates raw neural network probabilities using Temperature Scaling
    and computes dynamic confidence uncertainty bands.
    """

    def __init__(self, temperature: float = 1.25):
        self.temperature = max(0.1, temperature)

    def _prob_to_logit(self, p: float) -> float:
        p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        return float(np.log(p_clipped / (1.0 - p_clipped)))

    def _sigmoid(self, x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0))))

    def calibrate(self, raw_prob: float) -> float:
        """Applies temperature scaling to raw classification probability."""
        logit = self._prob_to_logit(raw_prob)
        calibrated_logit = logit / self.temperature
        return self._sigmoid(calibrated_logit)

    def compute_uncertainty(self, p: float, n_tokens: int = 40) -> Dict[str, Any]:
        """
        Calculates confidence intervals and categorical verdict.
        Variance is highest around p ~ 0.5 (maximum binary entropy)
        and decreases with sequence length n_tokens.
        """
        # Binary entropy: H(p) = -p log(p) - (1-p) log(1-p)
        eps = 1e-9
        entropy = -(p * np.log(p + eps) + (1.0 - p) * np.log(1.0 - p + eps))
        
        # Standard error proxy
        margin = float(entropy * 0.08 + (1.0 / np.sqrt(max(n_tokens, 10))) * 0.04)
        margin = min(0.12, max(0.02, margin))

        lower = round(max(0.0, p - margin), 3)
        upper = round(min(1.0, p + margin), 3)

        # 3-Tier Verdict Range (prevents forcing binary results on ambiguous calls)
        if p >= 0.60:
            verdict = "HIGH RISK"
            color = "#B42318"  # Red
        elif p >= 0.40:
            verdict = "UNCERTAIN"
            color = "#F5A524"  # Amber
        else:
            verdict = "LOW RISK"
            color = "#16a34a"  # Green

        return {
            "calibrated_prob": round(p, 3),
            "lower_bound": lower,
            "upper_bound": upper,
            "confidence_margin": round(margin, 3),
            "verdict": verdict,
            "verdict_color": color,
            "is_uncertain": verdict == "UNCERTAIN"
        }
