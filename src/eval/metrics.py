"""
Comprehensive Evaluation Metrics for CallGuard
Calculates Precision, Recall, F1, AUROC, Confusion Matrix, Word Error Rate (WER),
and early-stage Time-to-Detection metrics.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Computes Precision, Recall, F1, AUROC, and Confusion Matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # In case a single class is present in y_true, handle AUROC safely
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = float("nan")

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4) if not np.isnan(auroc) else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "threshold": threshold,
        "support": int(len(y_true))
    }


def compute_word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Computes exact Word Error Rate (WER) using dynamic programming Levenshtein distance.
    WER = (Substitutions + Deletions + Insertions) / Total Reference Words
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float(len(hyp_words))

    # DP matrix: (len(ref) + 1) x (len(hyp) + 1)
    dp = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1].lower() == hyp_words[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                dlt = dp[i - 1][j] + 1
                dp[i][j] = min(sub, ins, dlt)

    total_errors = dp[len(ref_words)][len(hyp_words)]
    return float(total_errors / len(ref_words))


def compute_early_detection_metrics(
    call_prefix_results: List[Dict[str, Any]],
    threshold: float = 0.75,
    consecutive_k: int = 2
) -> Dict[str, Any]:
    """
    Calculates time-to-detection and early warning performance.
    Rule: A fraud call is considered detected at the earliest timestamp t where
    at least consecutive_k evaluations have remained >= threshold.
    Undetected calls are NOT counted as zero-second detections.
    """
    time_to_detections = []
    false_alerts = 0
    total_fraud = 0
    total_legit = 0
    detected_fraud = 0

    for call in call_prefix_results:
        is_fraud = bool(call["label"] == 1)
        timestamps = call["timestamps"]  # e.g. [5, 10, 15, 20...]
        probs = call["probabilities"]

        if is_fraud:
            total_fraud += 1
        else:
            total_legit += 1

        # Check consecutive alerts
        first_alert_time = None
        consecutive_count = 0
        for t, p in zip(timestamps, probs):
            if p >= threshold:
                consecutive_count += 1
                if consecutive_count >= consecutive_k:
                    first_alert_time = t
                    break
            else:
                consecutive_count = 0

        if is_fraud:
            if first_alert_time is not None:
                detected_fraud += 1
                time_to_detections.append(first_alert_time)
        else:
            if first_alert_time is not None:
                false_alerts += 1

    median_ttd = float(np.median(time_to_detections)) if time_to_detections else None
    mean_ttd = float(np.mean(time_to_detections)) if time_to_detections else None
    detection_rate = float(detected_fraud / max(total_fraud, 1))
    false_alert_rate = float(false_alerts / max(total_legit, 1))
    missed_fraud_rate = float((total_fraud - detected_fraud) / max(total_fraud, 1))

    return {
        "detection_rate": round(detection_rate, 4),
        "false_alert_rate": round(false_alert_rate, 4),
        "missed_fraud_rate": round(missed_fraud_rate, 4),
        "median_time_to_detection_sec": round(median_ttd, 2) if median_ttd is not None else None,
        "mean_time_to_detection_sec": round(mean_ttd, 2) if mean_ttd is not None else None,
        "total_fraud_calls": total_fraud,
        "detected_fraud_calls": detected_fraud,
        "total_legit_calls": total_legit,
        "false_alert_calls": false_alerts
    }
