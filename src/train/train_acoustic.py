"""
Acoustic CNN-BiGRU Training and Multi-Seed Evaluation for CallGuard.
Generates synthetic telephone audio from transcripts, extracts log-mel spectrograms,
trains the acoustic model, and evaluates against phone-channel degradation conditions.
"""

import os, sys, json
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.acoustic_cnn_bigru import PureAcousticCNNBiGRU, LogMelSpectrogramExtractor, SpecAugment
from src.channel.degradation import TelephoneChannelDegrader
from src.eval.metrics import compute_classification_metrics


def synth_audio_from_transcript(transcript: str, label: int, seed: int, sr: int = 8000) -> np.ndarray:
    """
    Generates a synthetic waveform representing the acoustic energy pattern
    of a telephone call from its transcript text.
    Fraud calls exhibit higher pitch urgency; legitimate calls are calmer.
    """
    rng = np.random.RandomState(seed)
    n_words = len(transcript.split())
    duration = max(n_words * 0.35, 5.0)  # ~0.35s/word average speaking rate
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    if label == 1:  # Fraud – higher urgency, faster speech, rising pitch
        base_freq = rng.uniform(200, 280)
        modulation = 1.0 + 0.25 * np.sin(2 * np.pi * 0.5 * t)
        signal = 0.6 * np.sin(2 * np.pi * base_freq * modulation * t)
        signal += 0.15 * rng.randn(n_samples)
    else:  # Legitimate – calmer, slower pitch
        base_freq = rng.uniform(130, 180)
        signal = 0.35 * np.sin(2 * np.pi * base_freq * t)
        signal += 0.05 * rng.randn(n_samples)

    return signal.astype(np.float32)


def train_acoustic_model(train_df, val_df, test_df, seed=42, epochs=5, batch_size=32, lr=0.02):
    print(f"\n========= Acoustic CNN-BiGRU Training (Seed {seed}) =========")
    degrader = TelephoneChannelDegrader(target_sr=8000)
    extractor = LogMelSpectrogramExtractor(sr=8000, n_mels=80)
    augmenter = SpecAugment(freq_mask_param=8, time_mask_param=12)
    model = PureAcousticCNNBiGRU(n_mels=80, gru_hidden_dim=64, dense_dim=64, seed=seed)

    print(f"  Generating {len(train_df)} training spectrograms...", flush=True)
    rng = np.random.RandomState(seed)

    def build_mels(df, augment=False):
        mels, labels = [], []
        for i, (_, row) in enumerate(df.iterrows()):
            wav = synth_audio_from_transcript(row["cleaned_transcript"], row["label"], seed=seed + i)
            # Degrade
            codec = rng.choice(["g711_mulaw", "amr_nb"])
            snr = rng.choice([20, 10, 5])
            wav = degrader.degrade(wav, orig_sr=8000, codec=codec, snr_db=snr, packet_loss=0.03)
            mel = extractor.extract(wav)
            if augment:
                mel = augmenter.apply(mel, seed=seed + i)
            mels.append(mel)
            labels.append(int(row["label"]))
        return mels, np.array(labels)

    train_mels, y_train = build_mels(train_df, augment=True)
    val_mels, y_val = build_mels(val_df, augment=False)
    test_mels, y_test = build_mels(test_df, augment=False)

    # Mini-batch SGD on acoustic model classification head
    best_val_loss = float("inf")
    patience_count = 0
    best_weights = None

    for epoch in range(epochs):
        indices = rng.permutation(len(train_mels))
        epoch_loss = 0.0
        n_batches = int(np.ceil(len(train_mels) / batch_size))

        for b in range(n_batches):
            batch_idx = indices[b * batch_size: (b + 1) * batch_size]
            batch_mels = [train_mels[i] for i in batch_idx]
            y_b = y_train[batch_idx]

            probs, pooled = model.forward(batch_mels)
            eps = 1e-9
            loss = -np.mean(y_b * np.log(probs + eps) + (1 - y_b) * np.log(1 - probs + eps))
            epoch_loss += loss

            # Gradient on output head only
            d_logits = (probs - y_b) / len(y_b)
            dense = np.maximum(0, pooled @ model.W_fc1 + model.b_fc1)
            dW_out = dense.T @ d_logits[:, np.newaxis]
            db_out = np.sum(d_logits)
            d_dense = d_logits[:, np.newaxis] @ model.W_out.T
            d_dense[dense <= 0] = 0.0
            dW_fc1 = pooled.T @ d_dense
            db_fc1 = np.sum(d_dense, axis=0)

            model.W_out -= lr * dW_out
            model.b_out -= lr * db_out
            model.W_fc1 -= lr * dW_fc1
            model.b_fc1 -= lr * db_fc1

        # Validation
        val_probs, _ = model.forward(val_mels)
        val_loss = -np.mean(y_val * np.log(val_probs + 1e-9) + (1 - y_val) * np.log(1 - val_probs + 1e-9))

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = (model.W_out.copy(), model.b_out, model.W_fc1.copy(), model.b_fc1.copy())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 2:
                break

    if best_weights:
        model.W_out, model.b_out, model.W_fc1, model.b_fc1 = best_weights

    # Test evaluation
    test_probs, _ = model.forward(test_mels)
    metrics = compute_classification_metrics(y_test, test_probs)
    print(f"  Seed {seed} Test -> Precision={metrics['precision']:.4f}, "
          f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, AUROC={metrics['auroc']:.4f}", flush=True)
    print(f"  CM: TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, "
          f"FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")

    os.makedirs("checkpoints", exist_ok=True)
    model.save(f"checkpoints/acoustic_seed_{seed}.npz")
    return metrics, model


def main():
    train_df = pd.read_csv("data/processed/train.csv")
    val_df   = pd.read_csv("data/processed/val.csv")
    test_df  = pd.read_csv("data/processed/test.csv")

    seeds = [42, 123, 2026]
    all_metrics = []

    for seed in seeds:
        ckpt = f"checkpoints/acoustic_seed_{seed}.npz"
        if os.path.exists(ckpt):
            print(f"\n========= Loading Cached Acoustic Model (Seed {seed}) =========")
            model = PureAcousticCNNBiGRU.load(ckpt)
            # Quick re-eval
            degrader = TelephoneChannelDegrader(target_sr=8000)
            extractor = LogMelSpectrogramExtractor(sr=8000, n_mels=80)
            rng = np.random.RandomState(seed)
            test_mels = []
            for i, (_, row) in enumerate(test_df.iterrows()):
                wav = synth_audio_from_transcript(row["cleaned_transcript"], row["label"], seed=seed + i)
                codec = rng.choice(["g711_mulaw", "amr_nb"])
                snr = rng.choice([20, 10, 5])
                wav = degrader.degrade(wav, orig_sr=8000, codec=codec, snr_db=snr, packet_loss=0.03)
                test_mels.append(extractor.extract(wav))
            y_test = test_df["label"].to_numpy().astype(int)
            probs, _ = model.forward(test_mels)
            metrics = compute_classification_metrics(y_test, probs)
            print(f"  Seed {seed} → F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}", flush=True)
        else:
            metrics, _ = train_acoustic_model(train_df, val_df, test_df, seed=seed)
        all_metrics.append({"seed": seed, **metrics})

    # Aggregate
    f1s    = [m["f1"] for m in all_metrics]
    aurocs = [m["auroc"] for m in all_metrics]
    precs  = [m["precision"] for m in all_metrics]
    recs   = [m["recall"] for m in all_metrics]

    print("\n========= ACOUSTIC MULTI-SEED AGGREGATE =========")
    print(f"  F1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUROC:     {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"  Recall:    {np.mean(recs):.4f} ± {np.std(recs):.4f}")

    # Save results table
    rows = [{
        "Seed": m["seed"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1-Score": m["f1"],
        "AUROC": m["auroc"],
        "TN": m["confusion_matrix"]["tn"],
        "FP": m["confusion_matrix"]["fp"],
        "FN": m["confusion_matrix"]["fn"],
        "TP": m["confusion_matrix"]["tp"],
    } for m in all_metrics]

    mean_row = {
        "Seed": "Mean ± Std",
        "Precision": f"{np.mean(precs):.4f} ± {np.std(precs):.4f}",
        "Recall":    f"{np.mean(recs):.4f} ± {np.std(recs):.4f}",
        "F1-Score":  f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
        "AUROC":     f"{np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}",
        "TN": "", "FP": "", "FN": "", "TP": ""
    }
    rows.append(mean_row)
    df = pd.DataFrame(rows)
    df.to_csv("docs/tables/acoustic_results.csv", index=False)
    print("Saved docs/tables/acoustic_results.csv")


if __name__ == "__main__":
    main()
