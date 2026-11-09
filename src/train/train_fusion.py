"""
Fusion Training — Phase 5 of CallGuard.
Loads pre-trained BiLSTM and Acoustic checkpoints, extracts embeddings,
trains GatedMultimodalFusion, and produces ablation results.
"""

import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.bilstm import PureBiLSTMTextModel
from src.models.acoustic_cnn_bigru import PureAcousticCNNBiGRU, LogMelSpectrogramExtractor
from src.models.fusion import GatedMultimodalFusion, LateFusion
from src.data.tokenizer import CallGuardTokenizer
from src.data.normalizer import TextNormalizer
from src.channel.degradation import TelephoneChannelDegrader
from src.eval.metrics import compute_classification_metrics


# ─────────────────────────── helpers ────────────────────────────

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


def get_bilstm_probs_and_embeddings(model: PureBiLSTMTextModel,
                                     tokenizer: CallGuardTokenizer,
                                     texts: list,
                                     batch_size: int = 32):
    """Returns (probs [N], embeddings [N, hidden_dim*2])"""
    all_probs = []
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        ids_array = tokenizer.batch_encode_and_pad(batch_texts)
        probs, context, _attn = model.forward(ids_array)  # (probs, pooled, attn)
        all_probs.extend(probs.tolist())
        all_embs.append(context)
    return np.array(all_probs), np.vstack(all_embs)


def get_acoustic_probs_and_embeddings(model: PureAcousticCNNBiGRU,
                                       extractor: LogMelSpectrogramExtractor,
                                       df: pd.DataFrame,
                                       seed: int):
    """Returns (probs [N], embeddings [N, dense_dim])"""
    degrader = TelephoneChannelDegrader(target_sr=8000)
    rng = np.random.RandomState(seed)
    mels = []
    for i, (_, row) in enumerate(df.iterrows()):
        wav = synth_audio_from_transcript(row["cleaned_transcript"], row["label"], seed=seed + i)
        codec = rng.choice(["g711_mulaw", "amr_nb"])
        snr = rng.choice([20, 10, 5])
        wav = degrader.degrade(wav, orig_sr=8000, codec=codec, snr_db=snr, packet_loss=0.03)
        mels.append(extractor.extract(wav))
    probs, embs = model.forward(mels)   # embs = [N, dense_dim]
    return probs, embs


# ─────────────────────────── main ────────────────────────────────

def main():
    SEED = 42
    np.random.seed(SEED)

    print("=" * 55)
    print("  CallGuard -- Fusion Training & Ablation Study")
    print("=" * 55)

    # Load data splits
    train_df = pd.read_csv("data/processed/train.csv")
    val_df   = pd.read_csv("data/processed/val.csv")
    test_df  = pd.read_csv("data/processed/test.csv")
    normalizer = TextNormalizer()

    # Data is already cleaned — just reference cleaned_transcript directly
    for df in [train_df, val_df, test_df]:
        if "cleaned_transcript" not in df.columns:
            df["cleaned_transcript"] = df.get("raw_transcript", df.get("transcript", "")).apply(
                lambda t: normalizer.normalize(str(t), lang="en")
            )

    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # ── Load BiLSTM ──────────────────────────────────────────────
    print(f"\n[1/4] Loading BiLSTM (seed {SEED})...")
    tokenizer = CallGuardTokenizer.load(f"data/processed/tokenizer_seed_{SEED}.json")
    bilstm = PureBiLSTMTextModel.load(f"checkpoints/bilstm_seed_{SEED}.npz")
    print(f"  BiLSTM loaded. vocab_size={len(tokenizer.word2idx)}")

    # ── Load Acoustic model ───────────────────────────────────────
    print(f"\n[2/4] Loading Acoustic CNN-BiGRU (seed {SEED})...")
    extractor = LogMelSpectrogramExtractor(sr=8000, n_mels=80)
    acoustic  = PureAcousticCNNBiGRU.load(f"checkpoints/acoustic_seed_{SEED}.npz")
    print("  Acoustic model loaded.")

    # ── Get branch probabilities + embeddings on test set ─────────
    print(f"\n[3/4] Running inference on test set ({len(test_df)} samples)...")
    y_test = test_df["label"].to_numpy().astype(int)
    test_texts = test_df["cleaned_transcript"].tolist()

    text_probs,  text_embs  = get_bilstm_probs_and_embeddings(bilstm, tokenizer, test_texts)
    audio_probs, audio_embs = get_acoustic_probs_and_embeddings(acoustic, extractor, test_df, seed=SEED)

    # ── Branch-only metrics ───────────────────────────────────────
    text_metrics  = compute_classification_metrics(y_test, text_probs)
    audio_metrics = compute_classification_metrics(y_test, audio_probs)

    print(f"\n  Text  branch -> F1={text_metrics['f1']:.4f}  AUROC={text_metrics['auroc']:.4f}")
    print(f"  Audio branch -> F1={audio_metrics['f1']:.4f}  AUROC={audio_metrics['auroc']:.4f}")

    # ── Late Fusion ───────────────────────────────────────────────
    late_fusion  = LateFusion(w_text=0.55, w_audio=0.45)
    fused_probs  = late_fusion.fuse(text_probs, audio_probs)
    late_metrics = compute_classification_metrics(y_test, fused_probs)
    print(f"  Late fusion  -> F1={late_metrics['f1']:.4f}  AUROC={late_metrics['auroc']:.4f}")

    # ── Gated Fusion (trained) ────────────────────────────────────
    print(f"\n[4/4] Training GatedMultimodalFusion on train embeddings...")
    text_dim  = text_embs.shape[1]   # 192*2 = 384 (BiLSTM bidirectional context)
    audio_dim = audio_embs.shape[1]  # 64 (dense layer)
    fused_dim = 64

    # Build train embeddings
    train_texts_list  = train_df["cleaned_transcript"].tolist()
    train_text_probs,  train_text_embs  = get_bilstm_probs_and_embeddings(bilstm, tokenizer, train_texts_list)
    train_audio_probs, train_audio_embs = get_acoustic_probs_and_embeddings(acoustic, extractor, train_df, seed=SEED + 1000)
    y_train = train_df["label"].to_numpy().astype(int)

    # Val embeddings
    val_texts_list  = val_df["cleaned_transcript"].tolist()
    val_text_probs,  val_text_embs  = get_bilstm_probs_and_embeddings(bilstm, tokenizer, val_texts_list)
    val_audio_probs, val_audio_embs = get_acoustic_probs_and_embeddings(acoustic, extractor, val_df, seed=SEED + 2000)
    y_val = val_df["label"].to_numpy().astype(int)

    gated = GatedMultimodalFusion(dim_text=text_dim, dim_audio=audio_dim, fused_dim=fused_dim, seed=SEED)

    best_val_f1   = -1.0
    patience_ctr  = 0
    patience      = 5
    lr            = 0.01

    for epoch in range(30):
        idx = np.random.permutation(len(train_df))
        t_embs_sh = train_text_embs[idx]
        a_embs_sh = train_audio_embs[idx]
        y_sh      = y_train[idx]
        epoch_loss = 0.0
        n_batches  = 0
        bs = 32

        for i in range(0, len(train_df), bs):
            t_b = t_embs_sh[i:i+bs]
            a_b = a_embs_sh[i:i+bs]
            y_b = y_sh[i:i+bs]

            fused_p, gate, fused_emb = gated.fuse(t_b, a_b)

            # BCE loss
            eps = 1e-7
            loss = -np.mean(y_b * np.log(fused_p + eps) + (1 - y_b) * np.log(1 - fused_p + eps))
            epoch_loss += loss
            n_batches  += 1

            # Gradient w.r.t W_out (output layer)
            d_out = (fused_p - y_b) / (len(y_b) + eps)                           # [B]
            grad_W_out = (fused_emb.T @ d_out[:, None]) / len(y_b)               # [fused_dim, 1]
            grad_b_out = np.mean(d_out)

            # Gradient through fused_emb -> W_t, W_a, W_g
            d_fused_emb = d_out[:, None] * (gated.W_out.T)                        # [B, fused_dim]
            # gate * proj_t + (1-gate) * proj_a
            proj_t = t_b @ gated.W_t                                               # [B, fused_dim]
            proj_a = a_b @ gated.W_a
            d_proj_t = d_fused_emb * gate
            d_proj_a = d_fused_emb * (1.0 - gate)
            grad_W_t = (t_b.T @ d_proj_t) / len(y_b)
            grad_W_a = (a_b.T @ d_proj_a) / len(y_b)

            # Gradient through gate
            d_gate = d_fused_emb * (proj_t - proj_a)
            sig_gate = gate
            d_gate_pre = d_gate * sig_gate * (1.0 - sig_gate)
            concat = np.concatenate([t_b, a_b], axis=-1)
            grad_W_g = (concat.T @ d_gate_pre) / len(y_b)
            grad_b_g = np.mean(d_gate_pre, axis=0)

            # SGD update
            gated.W_out -= lr * grad_W_out
            gated.b_out -= lr * float(grad_b_out)
            gated.W_t   -= lr * grad_W_t
            gated.W_a   -= lr * grad_W_a
            gated.W_g   -= lr * grad_W_g
            gated.b_g   -= lr * grad_b_g

        # Validation
        val_fp, _, _ = gated.fuse(val_text_embs, val_audio_embs)
        val_m = compute_classification_metrics(y_val, val_fp)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1:2d} | loss={epoch_loss/n_batches:.4f} | val_F1={val_m['f1']:.4f}", flush=True)

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stop at epoch {epoch+1}.")
                break

    # Save gated fusion weights
    os.makedirs("checkpoints", exist_ok=True)
    gated.save("checkpoints/fusion_gated.npz")
    print("  Saved -> checkpoints/fusion_gated.npz")

    # Final test evaluation
    gated_probs, _, _ = gated.fuse(text_embs, audio_embs)
    gated_metrics = compute_classification_metrics(y_test, gated_probs)
    print(f"\n  Gated fusion -> F1={gated_metrics['f1']:.4f}  AUROC={gated_metrics['auroc']:.4f}")

    # ── Ablation table ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  FUSION ABLATION RESULTS")
    print("=" * 55)
    rows_data = [
        ("Text-only (BiLSTM)",       text_metrics),
        ("Audio-only (CNN-BiGRU)",   audio_metrics),
        ("Late Fusion",              late_metrics),
        ("Gated Multimodal Fusion",  gated_metrics),
    ]

    print(f"  {'Model':<30} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUROC':>7}")
    print("  " + "-" * 60)
    for name, m in rows_data:
        print(f"  {name:<30} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['auroc']:>7.4f}")

    # Save CSV
    os.makedirs("docs/tables", exist_ok=True)
    ablation_rows = []
    for name, m in rows_data:
        ablation_rows.append({
            "Model": name,
            "Precision": round(m["precision"], 4),
            "Recall":    round(m["recall"], 4),
            "F1-Score":  round(m["f1"], 4),
            "AUROC":     round(m["auroc"], 4),
        })
    pd.DataFrame(ablation_rows).to_csv("docs/tables/fusion_ablation.csv", index=False)
    print("\n  Saved -> docs/tables/fusion_ablation.csv")

    md_lines = [
        "| Model | Precision | Recall | F1-Score | AUROC |",
        "|---|---|---|---|---|",
    ]
    for r in ablation_rows:
        md_lines.append(
            f"| {r['Model']} | {r['Precision']} | {r['Recall']} | {r['F1-Score']} | {r['AUROC']} |"
        )
    with open("docs/tables/fusion_ablation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  Saved -> docs/tables/fusion_ablation.md")


if __name__ == "__main__":
    main()
