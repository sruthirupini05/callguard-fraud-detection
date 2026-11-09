"""
BiLSTM Text Baseline Training and Multi-Seed Evaluation for CallGuard.
Trains across seeds (42, 123, 2026) and logs Precision, Recall, F1, AUROC, Confusion Matrix.
"""

import json
import os
import sys
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.tokenizer import CallGuardTokenizer
from src.models.bilstm import PureBiLSTMTextModel
from src.eval.metrics import compute_classification_metrics


def train_and_eval_seed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    seed: int
) -> Dict[str, Any]:
    ckpt_path = os.path.join("checkpoints", f"bilstm_seed_{seed}.npz")
    tok_path = os.path.join("data", "processed", f"tokenizer_seed_{seed}.json")
    
    # 1. Fit tokenizer on training set only (or load if cached)
    if os.path.exists(tok_path):
        tokenizer = CallGuardTokenizer.load(tok_path)
    else:
        tokenizer = CallGuardTokenizer(
            max_vocab_size=config["vocab_size"],
            max_seq_length=config.get("max_sequence_length", 128)
        )
        tokenizer.fit_on_texts(train_df["cleaned_transcript"].tolist())
        tokenizer.save(tok_path)
    
    # 2. Vectorize texts
    X_test = tokenizer.batch_encode_and_pad(test_df["cleaned_transcript"].tolist())
    y_test = test_df["label"].to_numpy().astype(int)
    
    if os.path.exists(ckpt_path):
        print(f"\n================ Loading Cached Checkpoint (Seed {seed}) ================")
        model = PureBiLSTMTextModel.load(ckpt_path)
        history = {}
    else:
        print(f"\n================ Running BiLSTM Training (Seed {seed}) ================")
        X_train = tokenizer.batch_encode_and_pad(train_df["cleaned_transcript"].tolist())
        y_train = train_df["label"].to_numpy().astype(int)
        
        X_val = tokenizer.batch_encode_and_pad(val_df["cleaned_transcript"].tolist())
        y_val = val_df["label"].to_numpy().astype(int)
        
        model = PureBiLSTMTextModel(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config.get("hidden_dim", 96),
            dense_dim=config["dense_dim"],
            dropout=config["dropout"],
            seed=seed
        )
        
        history = model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=config.get("epochs", 5),
            batch_size=config["batch_size"],
            lr=config["learning_rate"],
            patience=config["early_stopping_patience"]
        )
        os.makedirs("checkpoints", exist_ok=True)
        model.save(ckpt_path)
    
    # Evaluate on Test Set
    test_probs, test_embeddings, test_attn = model.forward(X_test)
    metrics = compute_classification_metrics(y_test, test_probs, threshold=0.5)
    
    print(f"Seed {seed} Test Results: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, AUROC={metrics['auroc']:.4f}", flush=True)
    print(f"Confusion Matrix: TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}", flush=True)
    
    return {
        "seed": seed,
        "metrics": metrics,
        "history": history
    }


def main():
    with open("configs/model.yaml", "r") as f:
        m_cfg = yaml.safe_load(f)["bilstm_text"]
    with open("configs/data.yaml", "r") as f:
        d_cfg = yaml.safe_load(f)["data"]
        
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    seeds = [42, 123, 2026]
    all_results = []
    
    for seed in seeds:
        res = train_and_eval_seed(train_df, val_df, test_df, m_cfg, seed)
        all_results.append(res)
        
    # Aggregate mean and standard deviation
    precisions = [r["metrics"]["precision"] for r in all_results]
    recalls = [r["metrics"]["recall"] for r in all_results]
    f1s = [r["metrics"]["f1"] for r in all_results]
    aurocs = [r["metrics"]["auroc"] for r in all_results]
    
    summary = {
        "model": "BiLSTM + Attention (Text Baseline)",
        "precision_mean": round(float(np.mean(precisions)), 4),
        "precision_std": round(float(np.std(precisions)), 4),
        "recall_mean": round(float(np.mean(recalls)), 4),
        "recall_std": round(float(np.std(recalls)), 4),
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s)), 4),
        "auroc_mean": round(float(np.mean(aurocs)), 4),
        "auroc_std": round(float(np.std(aurocs)), 4)
    }
    
    print("\n================ MULTI-SEED AGGREGATE SUMMARY ================")
    print(f"Precision: {summary['precision_mean']} ± {summary['precision_std']}")
    print(f"Recall:    {summary['recall_mean']} ± {summary['recall_std']}")
    print(f"F1-Score:  {summary['f1_mean']} ± {summary['f1_std']}")
    print(f"AUROC:     {summary['auroc_mean']} ± {summary['auroc_std']}")
    
    # Save tables
    os.makedirs("docs/tables", exist_ok=True)
    res_df = pd.DataFrame([{
        "Seed": r["seed"],
        "Precision": r["metrics"]["precision"],
        "Recall": r["metrics"]["recall"],
        "F1-Score": r["metrics"]["f1"],
        "AUROC": r["metrics"]["auroc"],
        "TN": r["metrics"]["confusion_matrix"]["tn"],
        "FP": r["metrics"]["confusion_matrix"]["fp"],
        "FN": r["metrics"]["confusion_matrix"]["fn"],
        "TP": r["metrics"]["confusion_matrix"]["tp"]
    } for r in all_results])
    
    # Add Mean row
    mean_row = {
        "Seed": "Mean ± Std",
        "Precision": f"{summary['precision_mean']} ± {summary['precision_std']}",
        "Recall": f"{summary['recall_mean']} ± {summary['recall_std']}",
        "F1-Score": f"{summary['f1_mean']} ± {summary['f1_std']}",
        "AUROC": f"{summary['auroc_mean']} ± {summary['auroc_std']}",
        "TN": int(np.mean([r["metrics"]["confusion_matrix"]["tn"] for r in all_results])),
        "FP": int(np.mean([r["metrics"]["confusion_matrix"]["fp"] for r in all_results])),
        "FN": int(np.mean([r["metrics"]["confusion_matrix"]["fn"] for r in all_results])),
        "TP": int(np.mean([r["metrics"]["confusion_matrix"]["tp"] for r in all_results]))
    }
    res_df = pd.concat([res_df, pd.DataFrame([mean_row])], ignore_index=True)
    
    csv_path = "docs/tables/baseline_text_results.csv"
    md_path = "docs/tables/baseline_text_results.md"
    res_df.to_csv(csv_path, index=False)
    
    md_lines = []
    md_lines.append("# CallGuard Phase 1: BiLSTM Text Baseline Multi-Seed Results\n\n")
    headers = list(res_df.columns)
    md_lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    md_lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in res_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in row) + " |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Results saved to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
