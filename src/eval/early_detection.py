"""
Early-Stage Fraud Detection Engine for CallGuard.
Evaluates incoming conversations in progressive 5-second prefix increments.
Enforces multi-window persistence (k=2) to trigger early warning alerts while minimizing false positives.
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.eval.metrics import compute_early_detection_metrics


class EarlyStageDetectionEngine:
    """Simulates real-time streaming prefixes of calls to evaluate time-to-detection."""

    def __init__(self, window_seconds: float = 5.0, fraud_threshold: float = 0.75, consecutive_k: int = 2):
        self.window_seconds = window_seconds
        self.fraud_threshold = fraud_threshold
        self.consecutive_k = consecutive_k

    def slice_call_prefixes(self, call_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Slices full call conversation turns into progressive 5s prefixes.
        For each prefix, accumulates turns whose timestamps fall within the window.
        """
        turns = call_data.get("turns", [])
        total_duration = float(call_data.get("duration_seconds", 30.0))
        prefixes = []
        
        current_t = self.window_seconds
        while current_t <= total_duration + self.window_seconds:
            # Collect turns heard up to current_t
            prefix_turns = [t for t in turns if t["start_time"] < current_t]
            prefix_text = " \n".join([f"{t['speaker'].upper()}: {t['text']}" for t in prefix_turns])
            
            # Check suspicious phrase presence in prefix
            suspicious_in_prefix = [
                p for p in call_data.get("suspicious_phrases", [])
                if any(p.lower() in t["text"].lower() for t in prefix_turns)
            ]
            
            prefixes.append({
                "seconds": round(current_t, 1),
                "transcript": prefix_text,
                "num_turns": len(prefix_turns),
                "flagged_phrases": suspicious_in_prefix
            })
            current_t += self.window_seconds
            
        return prefixes

    def evaluate_test_calls(
        self,
        test_calls: List[Dict[str, Any]],
        predict_fn
    ) -> Dict[str, Any]:
        """
        Runs prefix evaluation across test calls and calculates detection rate curves.
        """
        call_eval_records = []
        time_bins = [5, 10, 15, 20, 25, 30]
        detections_at_time = {t: 0 for t in time_bins}
        total_fraud_calls = 0

        for call in test_calls:
            label = int(call["label"])
            if label == 1:
                total_fraud_calls += 1
                
            prefixes = self.slice_call_prefixes(call)
            timestamps = []
            probs = []

            for p in prefixes:
                sec = p["seconds"]
                transcript = p["transcript"]
                # Score prefix
                prob = predict_fn(transcript)
                timestamps.append(sec)
                probs.append(prob)

            call_eval_records.append({
                "call_id": call["call_id"],
                "label": label,
                "timestamps": timestamps,
                "probabilities": probs
            })

            # Check if flagged by specific timestamp bins
            if label == 1:
                consecutive = 0
                first_flag_t = None
                for t, p in zip(timestamps, probs):
                    if p >= self.fraud_threshold:
                        consecutive += 1
                        if consecutive >= self.consecutive_k:
                            first_flag_t = t
                            break
                    else:
                        consecutive = 0

                if first_flag_t is not None:
                    for tb in time_bins:
                        if first_flag_t <= tb:
                            detections_at_time[tb] += 1

        metrics = compute_early_detection_metrics(
            call_eval_records,
            threshold=self.fraud_threshold,
            consecutive_k=self.consecutive_k
        )

        # Detection rate vs time curve
        curve = {
            f"{tb}s": round(detections_at_time[tb] / max(total_fraud_calls, 1), 4)
            for tb in time_bins
        }
        metrics["detection_rate_vs_time"] = curve
        return metrics
