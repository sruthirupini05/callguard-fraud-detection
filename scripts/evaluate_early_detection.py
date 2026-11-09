"""
Phase 6 — Early-Stage Fraud Detection Evaluation.
Tests detection rate at increasing conversation prefix lengths (5s, 10s, 15s, 20s, 30s).
"""

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.bilstm import PureBiLSTMTextModel
from src.data.tokenizer import CallGuardTokenizer
from src.data.normalizer import TextNormalizer
from src.eval.metrics import compute_classification_metrics


WORDS_PER_SECOND = 2.5   # average telephone speech rate


def truncate_to_prefix(transcript: str, seconds: float) -> str:
    """Return the first `seconds` seconds worth of words."""
    words = transcript.split()
    max_words = max(1, int(seconds * WORDS_PER_SECOND))
    return " ".join(words[:max_words])


def main():
    SEED = 42
    THRESHOLDS = [5, 10, 15, 20, 30, None]  # None = full transcript
    FRAUD_THRESHOLD = 0.5

    print("=" * 55)
    print("  CallGuard -- Early-Stage Detection Evaluation")
    print("=" * 55)

    # Load model
    tokenizer = CallGuardTokenizer.load(f"data/processed/tokenizer_seed_{SEED}.json")
    model = PureBiLSTMTextModel.load(f"checkpoints/bilstm_seed_{SEED}.npz")
    print(f"  BiLSTM loaded (seed {SEED}).")

    # Load test set
    test_df = pd.read_csv("data/processed/test.csv")
    normalizer = TextNormalizer()
    if "cleaned_transcript" not in test_df.columns:
        test_df["cleaned_transcript"] = test_df["raw_transcript"].apply(
            lambda t: normalizer.normalize(str(t), lang="en")
        )
    y_test = test_df["label"].to_numpy().astype(int)
    n_fraud = int(y_test.sum())
    print(f"  Test set: {len(test_df)} samples  ({n_fraud} fraud, {len(y_test)-n_fraud} legit)")

    results = []

    for t in THRESHOLDS:
        label = f"{t}s" if t is not None else "Full"

        if t is None:
            texts = test_df["cleaned_transcript"].tolist()
        else:
            texts = [truncate_to_prefix(tx, t) for tx in test_df["cleaned_transcript"]]

        # Batch inference
        all_probs = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            ids_arr = tokenizer.batch_encode_and_pad(batch_texts)
            probs, _, _ = model.forward(ids_arr)
            all_probs.extend(probs.tolist())

        probs_arr = np.array(all_probs)
        preds = (probs_arr >= FRAUD_THRESHOLD).astype(int)

        # Metrics
        metrics = compute_classification_metrics(y_test, probs_arr)

        # Detection rate = fraction of fraud calls correctly detected
        fraud_mask = (y_test == 1)
        detection_rate = float(preds[fraud_mask].mean()) if fraud_mask.sum() > 0 else 0.0
        false_alarm_rate = float(preds[~fraud_mask].mean()) if (~fraud_mask).sum() > 0 else 0.0

        print(
            f"  Prefix={label:6s} | F1={metrics['f1']:.4f} | "
            f"DetRate={detection_rate:.3f} | FAR={false_alarm_rate:.3f} | "
            f"AUROC={metrics['auroc']:.4f}"
        )

        results.append({
            "Prefix (s)":    t if t is not None else "Full",
            "Avg Words":     int(WORDS_PER_SECOND * t) if t is not None else int(test_df["cleaned_transcript"].apply(lambda x: len(x.split())).mean()),
            "Precision":     round(metrics["precision"], 4),
            "Recall":        round(metrics["recall"], 4),
            "F1-Score":      round(metrics["f1"], 4),
            "AUROC":         round(metrics["auroc"], 4),
            "Detection Rate": round(detection_rate, 4),
            "False Alarm Rate": round(false_alarm_rate, 4),
        })

    # Save
    os.makedirs("docs/tables", exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv("docs/tables/early_detection_results.csv", index=False)
    print("\n  Saved -> docs/tables/early_detection_results.csv")

    # Markdown table
    md_lines = [
        "| Prefix (s) | Avg Words | Precision | Recall | F1-Score | AUROC | Detection Rate | False Alarm Rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        md_lines.append(
            f"| {r['Prefix (s)']} | {r['Avg Words']} | {r['Precision']} | {r['Recall']} | "
            f"{r['F1-Score']} | {r['AUROC']} | {r['Detection Rate']} | {r['False Alarm Rate']} |"
        )
    with open("docs/tables/early_detection_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  Saved -> docs/tables/early_detection_results.md")

    print("\n  Done. Early-stage detection evaluation complete.")


if __name__ == "__main__":
    main()
