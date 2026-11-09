"""
ASR Robustness Evaluation Script for CallGuard.
Runs degradation matrix across codecs / SNR / packet-loss conditions,
measures WER and resulting BiLSTM F1 degradation, and saves tables.
"""

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.tokenizer import CallGuardTokenizer
from src.models.bilstm import PureBiLSTMTextModel
from src.asr.asr_pipeline import CallGuardASRPipeline
from src.eval.metrics import compute_classification_metrics


def run_robustness_evaluation():
    test_df = pd.read_csv("data/processed/test.csv")
    tokenizer = CallGuardTokenizer.load("data/processed/tokenizer_seed_42.json")
    model = PureBiLSTMTextModel.load("checkpoints/bilstm_seed_42.npz")
    asr = CallGuardASRPipeline()

    # Clean transcript baseline
    X_clean = tokenizer.batch_encode_and_pad(test_df["cleaned_transcript"].tolist())
    y_test = test_df["label"].to_numpy().astype(int)
    clean_probs, _, _ = model.forward(X_clean)
    clean_metrics = compute_classification_metrics(y_test, clean_probs)

    print("=== Clean Transcript Baseline ===")
    print(f"  F1={clean_metrics['f1']:.4f}  AUROC={clean_metrics['auroc']:.4f}")

    degradation_matrix = [
        {"codec": "none",        "snr_db": 20, "packet_loss": 0.00},
        {"codec": "g711_mulaw",  "snr_db": 20, "packet_loss": 0.00},
        {"codec": "g711_mulaw",  "snr_db": 10, "packet_loss": 0.03},
        {"codec": "g711_alaw",   "snr_db": 10, "packet_loss": 0.03},
        {"codec": "amr_nb",      "snr_db": 20, "packet_loss": 0.00},
        {"codec": "amr_nb",      "snr_db": 10, "packet_loss": 0.03},
        {"codec": "amr_nb",      "snr_db": 5,  "packet_loss": 0.03},
        {"codec": "amr_nb",      "snr_db": 0,  "packet_loss": 0.10},
    ]

    results = []
    for deg in degradation_matrix:
        codec, snr_db, packet_loss = deg["codec"], deg["snr_db"], deg["packet_loss"]
        wers, hyps = [], []
        for idx, row in test_df.iterrows():
            lang = row["language"]
            ref = row["cleaned_transcript"]
            hyp, wer = asr.simulate_degraded_transcription(
                ref, language=lang, codec=codec,
                snr_db=snr_db, packet_loss=packet_loss,
                seed=42 + int(idx)
            )
            wers.append(wer)
            hyps.append(hyp)

        X_asr = tokenizer.batch_encode_and_pad(hyps)
        probs, _, _ = model.forward(X_asr)
        m = compute_classification_metrics(y_test, probs)
        mean_wer = float(np.mean(wers))
        f1_drop = round(clean_metrics["f1"] - m["f1"], 4)

        row_result = {
            "Codec": codec,
            "SNR_dB": snr_db,
            "Packet_Loss_%": int(packet_loss * 100),
            "Mean_WER": round(mean_wer, 4),
            "F1-Score": m["f1"],
            "F1_Drop": f1_drop,
            "AUROC": m["auroc"],
            "Precision": m["precision"],
            "Recall": m["recall"],
        }
        results.append(row_result)
        print(f"  [{codec:12s} | SNR={snr_db:2d}dB | PL={int(packet_loss*100)}%] "
              f"WER={mean_wer:.3f}  F1={m['f1']:.4f}  Drop={f1_drop:+.4f}", flush=True)

    df = pd.DataFrame(results)
    os.makedirs("docs/tables", exist_ok=True)
    df.to_csv("docs/tables/asr_robustness_matrix.csv", index=False)

    # Write markdown manually (no tabulate dependency)
    headers = list(df.columns)
    md = ["# CallGuard: ASR Robustness Evaluation Matrix\n",
          "| " + " | ".join(headers) + " |",
          "| " + " | ".join("---" for _ in headers) + " |"]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(str(v) for v in r) + " |")
    with open("docs/tables/asr_robustness_matrix.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nSaved: docs/tables/asr_robustness_matrix.csv")


if __name__ == "__main__":
    run_robustness_evaluation()
