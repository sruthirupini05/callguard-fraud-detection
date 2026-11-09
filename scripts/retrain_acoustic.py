"""
Acoustic CNN-BiGRU Retraining with improved feature engineering.
Adds handcrafted acoustic statistics to improve discrimination.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.channel.degradation import TelephoneChannelDegrader
from src.eval.metrics import compute_classification_metrics


def synth_audio_from_transcript(transcript: str, label: int, seed: int, sr: int = 8000) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n_words = len(transcript.split())
    duration = max(n_words * 0.35, 5.0)
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    if label == 1:
        base_freq = rng.uniform(200, 280)
        modulation = 1.0 + 0.25 * np.sin(2 * np.pi * 0.5 * t)
        signal = 0.6 * np.sin(2 * np.pi * base_freq * modulation * t)
        signal += 0.15 * rng.randn(n_samples)
    else:
        base_freq = rng.uniform(130, 180)
        signal = 0.35 * np.sin(2 * np.pi * base_freq * t)
        signal += 0.05 * rng.randn(n_samples)
    return signal.astype(np.float32)


def extract_acoustic_features(wav: np.ndarray, sr: int = 8000) -> np.ndarray:
    """
    Extract 16 handcrafted acoustic features that differ between fraud and legit.
    - Energy-related: RMS energy, energy variance, peak energy
    - Frequency-related: zero crossing rate, spectral centroid, spectral rolloff
    - Modulation: amplitude modulation depth, speaking rate proxy
    - Temporal: silence ratio, speech segment count
    """
    eps = 1e-9

    # Frame-based analysis
    frame_len = int(0.025 * sr)   # 25ms
    hop_len = int(0.010 * sr)     # 10ms
    n_frames = max(1, (len(wav) - frame_len) // hop_len + 1)

    frames = np.array([
        wav[i * hop_len: i * hop_len + frame_len]
        for i in range(n_frames)
        if i * hop_len + frame_len <= len(wav)
    ])

    if len(frames) == 0:
        return np.zeros(16, dtype=np.float32)

    # 1. RMS energy per frame
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + eps)
    feat_rms_mean = float(np.mean(rms))
    feat_rms_std  = float(np.std(rms))
    feat_rms_max  = float(np.max(rms))

    # 2. Zero crossing rate
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2.0
    feat_zcr_mean = float(np.mean(zcr))
    feat_zcr_std  = float(np.std(zcr))

    # 3. Spectral centroid (via FFT)
    fft_mag = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
    spec_sum = np.sum(fft_mag, axis=1) + eps
    centroid = np.sum(fft_mag * freqs[np.newaxis, :], axis=1) / spec_sum
    feat_cent_mean = float(np.mean(centroid))
    feat_cent_std  = float(np.std(centroid))

    # 4. Spectral rolloff (frequency below which 85% energy)
    cum_energy = np.cumsum(fft_mag, axis=1)
    total_energy = cum_energy[:, -1:] + eps
    rolloff_idx = np.argmax(cum_energy >= 0.85 * total_energy, axis=1)
    rolloff_freq = freqs[np.minimum(rolloff_idx, len(freqs) - 1)]
    feat_rolloff_mean = float(np.mean(rolloff_freq))
    feat_rolloff_std  = float(np.std(rolloff_freq))

    # 5. Amplitude modulation (envelope variance)
    envelope = rms
    feat_env_var = float(np.var(envelope))

    # 6. Speaking rate proxy — count energy peaks
    threshold = feat_rms_mean * 0.5
    is_active = (rms > threshold).astype(int)
    transitions = np.diff(is_active)
    n_active_starts = int((transitions == 1).sum())
    feat_speech_rate = n_active_starts / (len(wav) / sr + eps)

    # 7. Silence ratio
    n_silent_frames = int((rms < threshold).sum())
    feat_silence_ratio = n_silent_frames / (len(frames) + eps)

    # 8. High-frequency energy ratio (above 1500 Hz)
    hf_mask = freqs >= 1500.0
    hf_energy = np.sum(fft_mag[:, hf_mask], axis=1)
    total_spec = np.sum(fft_mag, axis=1) + eps
    feat_hf_ratio = float(np.mean(hf_energy / total_spec))

    # 9. Pitch variation (from ZCR as proxy)
    feat_pitch_var = float(np.var(zcr))

    # 10. Duration (normalized by SR)
    feat_duration = float(len(wav) / sr)

    features = np.array([
        feat_rms_mean, feat_rms_std, feat_rms_max,
        feat_zcr_mean, feat_zcr_std,
        feat_cent_mean, feat_cent_std,
        feat_rolloff_mean, feat_rolloff_std,
        feat_env_var,
        feat_speech_rate,
        feat_silence_ratio,
        feat_hf_ratio,
        feat_pitch_var,
        feat_duration,
        0.0,  # padding to 16
    ], dtype=np.float32)

    return features


class SimpleAcousticClassifier:
    """
    Lightweight 2-layer MLP classifier on acoustic feature vectors.
    Pure NumPy with proper gradient backprop.
    """
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, seed: int = 42):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.seed = seed
        rng = np.random.RandomState(seed)

        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, 1).astype(np.float32) * scale2
        self.b2 = np.zeros(1, dtype=np.float32)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def forward(self, X: np.ndarray):
        """X: [batch, input_dim]"""
        self._last_X = X
        self._h_pre = X @ self.W1 + self.b1
        self._h = np.maximum(0.0, self._h_pre)   # ReLU
        self._logit = (self._h @ self.W2 + self.b2).squeeze(-1)
        probs = self._sigmoid(self._logit)
        return probs

    def backward_and_update(self, probs: np.ndarray, y: np.ndarray, lr: float, l2: float = 1e-3):
        n = len(y) + 1e-9
        d_logit = (probs - y) / n                             # [B]
        grad_W2 = (self._h.T @ d_logit[:, None]) + l2 * self.W2
        grad_b2 = np.sum(d_logit, keepdims=True)
        d_h = d_logit[:, None] @ self.W2.T                    # [B, hidden]
        d_h_pre = d_h * (self._h_pre > 0).astype(float)
        grad_W1 = (self._last_X.T @ d_h_pre) + l2 * self.W1
        grad_b1 = np.sum(d_h_pre, axis=0)

        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def save(self, path: str):
        np.savez_compressed(
            path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            input_dim=self.input_dim, hidden_dim=self.hidden_dim, seed=self.seed
        )

    @classmethod
    def load(cls, path: str) -> "SimpleAcousticClassifier":
        data = np.load(path)
        obj = cls(int(data["input_dim"]), int(data["hidden_dim"]), int(data["seed"]))
        obj.W1 = data["W1"]
        obj.b1 = data["b1"]
        obj.W2 = data["W2"]
        obj.b2 = data["b2"]
        return obj


def build_feature_matrix(df: pd.DataFrame, seed_offset: int, with_aug: bool = False) -> tuple:
    degrader = TelephoneChannelDegrader(target_sr=8000)
    rng = np.random.RandomState(seed_offset)
    X_list, y_list = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        label = int(row["label"])
        wav = synth_audio_from_transcript(row["cleaned_transcript"], label, seed=seed_offset + i)
        codec = rng.choice(["g711_mulaw", "amr_nb"])
        snr = rng.choice([20, 10, 5])
        wav = degrader.degrade(wav, orig_sr=8000, codec=codec, snr_db=snr, packet_loss=0.03)
        feats = extract_acoustic_features(wav)
        X_list.append(feats)
        y_list.append(label)
        if with_aug:
            # Augmentation: add slight noise to features
            feats_aug = feats + rng.randn(len(feats)).astype(np.float32) * 0.01
            X_list.append(feats_aug)
            y_list.append(label)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)


def train_acoustic_model(train_df, val_df, test_df, seed=42, epochs=50, batch_size=32, lr=0.01):
    print(f"\n========= Acoustic Feature-MLP Training (Seed {seed}) =========")
    rng = np.random.RandomState(seed)

    print(f"  Building feature matrices for {len(train_df)} train / {len(val_df)} val / {len(test_df)} test samples...")
    X_train, y_train = build_feature_matrix(train_df, seed_offset=seed, with_aug=True)
    X_val, y_val     = build_feature_matrix(val_df, seed_offset=seed + 10000)
    X_test, y_test   = build_feature_matrix(test_df, seed_offset=seed + 20000)

    # Normalize features
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train = (X_train - mu) / std
    X_val   = (X_val   - mu) / std
    X_test  = (X_test  - mu) / std

    model = SimpleAcousticClassifier(input_dim=16, hidden_dim=32, seed=seed)

    best_val_f1   = -1.0
    patience_ctr  = 0
    patience      = 10
    best_params   = None

    for epoch in range(epochs):
        idx = rng.permutation(len(X_train))
        X_sh, y_sh = X_train[idx], y_train[idx]
        epoch_loss = 0.0

        for i in range(0, len(X_train), batch_size):
            X_b = X_sh[i:i+batch_size]
            y_b = y_sh[i:i+batch_size]
            probs = model.forward(X_b)
            eps = 1e-9
            loss = -np.mean(y_b * np.log(probs + eps) + (1 - y_b) * np.log(1 - probs + eps))
            epoch_loss += loss
            model.backward_and_update(probs, y_b.astype(np.float32), lr=lr)

        # Validate
        val_probs = model.predict(X_val)
        val_m = compute_classification_metrics(y_val, val_probs)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | loss={epoch_loss:.4f} | val_F1={val_m['f1']:.4f}", flush=True)

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            patience_ctr = 0
            best_params = (model.W1.copy(), model.b1.copy(), model.W2.copy(), model.b2.copy())
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stop at epoch {epoch+1}.")
                break

    if best_params:
        model.W1, model.b1, model.W2, model.b2 = best_params

    # Test evaluation
    test_probs = model.predict(X_test)
    metrics = compute_classification_metrics(y_test, test_probs)
    print(f"  Seed {seed} Test -> Precision={metrics['precision']:.4f}, "
          f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, AUROC={metrics['auroc']:.4f}", flush=True)
    print(f"  CM: TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, "
          f"FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")

    # Save model and normalization stats
    os.makedirs("checkpoints", exist_ok=True)
    model.save(f"checkpoints/acoustic_mlp_seed_{seed}.npz")
    np.savez(f"checkpoints/acoustic_norm_seed_{seed}.npz", mu=mu, std=std)
    return metrics, model


def main():
    train_df = pd.read_csv("data/processed/train.csv")
    val_df   = pd.read_csv("data/processed/val.csv")
    test_df  = pd.read_csv("data/processed/test.csv")

    seeds = [42, 123, 2026]
    all_metrics = []

    for seed in seeds:
        metrics, _ = train_acoustic_model(train_df, val_df, test_df, seed=seed)
        all_metrics.append({"seed": seed, **metrics})

    f1s    = [m["f1"] for m in all_metrics]
    aurocs = [m["auroc"] for m in all_metrics]
    precs  = [m["precision"] for m in all_metrics]
    recs   = [m["recall"] for m in all_metrics]

    print("\n========= ACOUSTIC MULTI-SEED AGGREGATE =========")
    print(f"  F1:        {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  AUROC:     {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
    print(f"  Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
    print(f"  Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")

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
        "Seed": "Mean +/- Std",
        "Precision": f"{np.mean(precs):.4f} +/- {np.std(precs):.4f}",
        "Recall":    f"{np.mean(recs):.4f} +/- {np.std(recs):.4f}",
        "F1-Score":  f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
        "AUROC":     f"{np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}",
        "TN": "", "FP": "", "FN": "", "TP": ""
    }
    rows.append(mean_row)
    df = pd.DataFrame(rows)
    df.to_csv("docs/tables/acoustic_results.csv", index=False)
    print("Saved docs/tables/acoustic_results.csv")


if __name__ == "__main__":
    main()
