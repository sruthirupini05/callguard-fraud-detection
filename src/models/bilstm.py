"""
Bidirectional LSTM with Attention Pooling for CallGuard Fraud Text Detection.
Optimized with precomputed input projections and dynamic sequence unrolling.
"""

import json
import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np


class PureBiLSTMTextModel:
    """
    Self-contained, reproducible BiLSTM text model with attention pooling.
    Optimized with precomputed linear projections for rapid training and inference.
    """
    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 128,
        hidden_dim: int = 96,
        dense_dim: int = 64,
        dropout: float = 0.3,
        seed: int = 42
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim
        self.dropout = dropout
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Initialize embedding matrix
        self.embedding_weights = self.rng.normal(0, 0.1, (vocab_size, embedding_dim))
        self.embedding_weights[0] = 0.0  # PAD
        
        # BiLSTM weights: split into W_x (input) and W_h (recurrent)
        scale_x = 1.0 / np.sqrt(embedding_dim)
        scale_h = 1.0 / np.sqrt(hidden_dim)
        
        # Forward LSTM
        self.W_x_fwd = self.rng.uniform(-scale_x, scale_x, (embedding_dim, 4 * hidden_dim))
        self.W_h_fwd = self.rng.uniform(-scale_h, scale_h, (hidden_dim, 4 * hidden_dim))
        self.b_fwd = np.zeros(4 * hidden_dim)
        
        # Backward LSTM
        self.W_x_bwd = self.rng.uniform(-scale_x, scale_x, (embedding_dim, 4 * hidden_dim))
        self.W_h_bwd = self.rng.uniform(-scale_h, scale_h, (hidden_dim, 4 * hidden_dim))
        self.b_bwd = np.zeros(4 * hidden_dim)
        
        # Additive Attention Pooling weights
        self.W_attn = self.rng.uniform(-0.1, 0.1, (2 * hidden_dim, 1))
        self.b_attn = 0.0
        
        # Dense classification head
        self.W_fc1 = self.rng.uniform(-0.1, 0.1, (2 * hidden_dim, dense_dim))
        self.b_fc1 = np.zeros(dense_dim)
        self.W_out = self.rng.uniform(-0.1, 0.1, (dense_dim, 1))
        self.b_out = 0.0

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -15.0, 15.0))

    def _lstm_step(self, gate_x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray, W_h: np.ndarray, b: np.ndarray):
        gates = gate_x + (h_prev @ W_h) + b
        i, f, g, o = np.split(gates, 4, axis=-1)
        i = self._sigmoid(i)
        f = self._sigmoid(f)
        g = self._tanh(g)
        o = self._sigmoid(o)
        c_next = f * c_prev + i * g
        h_next = o * self._tanh(c_next)
        return h_next, c_next

    def forward(self, input_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        input_ids: [batch_size, seq_len]
        Returns:
            probs: [batch_size] fraud probabilities
            pooled: [batch_size, 2 * hidden_dim] pooled sequence representation
            attn_weights: [batch_size, seq_len] attention weights
        """
        batch_size, full_seq_len = input_ids.shape
        active_lengths = np.sum(input_ids != 0, axis=1)
        seq_len = int(np.max(active_lengths)) if len(active_lengths) > 0 else 1
        seq_len = max(min(seq_len, full_seq_len), 1)

        sliced_ids = input_ids[:, :seq_len]
        embedded = self.embedding_weights[sliced_ids]  # [batch_size, seq_len, emb_dim]
        
        # Precompute input linear projections for all timesteps simultaneously
        gates_x_fwd = np.dot(embedded, self.W_x_fwd)  # [batch_size, seq_len, 4 * hidden_dim]
        gates_x_bwd = np.dot(embedded, self.W_x_bwd)  # [batch_size, seq_len, 4 * hidden_dim]
        
        # Forward LSTM recurrence
        h_fwd = np.zeros((batch_size, self.hidden_dim))
        c_fwd = np.zeros((batch_size, self.hidden_dim))
        fwd_outs = []
        for t in range(seq_len):
            h_fwd, c_fwd = self._lstm_step(gates_x_fwd[:, t, :], h_fwd, c_fwd, self.W_h_fwd, self.b_fwd)
            fwd_outs.append(h_fwd)
        fwd_seq = np.stack(fwd_outs, axis=1)  # [batch_size, seq_len, hidden_dim]
        
        # Backward LSTM recurrence
        h_bwd = np.zeros((batch_size, self.hidden_dim))
        c_bwd = np.zeros((batch_size, self.hidden_dim))
        bwd_outs = [None] * seq_len
        for t in reversed(range(seq_len)):
            h_bwd, c_bwd = self._lstm_step(gates_x_bwd[:, t, :], h_bwd, c_bwd, self.W_h_bwd, self.b_bwd)
            bwd_outs[t] = h_bwd
        bwd_seq = np.stack(bwd_outs, axis=1)  # [batch_size, seq_len, hidden_dim]
        
        # Concatenate BiLSTM representations
        bilstm_out = np.concatenate([fwd_seq, bwd_seq], axis=-1)  # [batch, seq_len, 2 * hidden_dim]
        
        # Additive Attention Pooling
        mask = (sliced_ids != 0)  # [batch, seq_len]
        raw_attn = (bilstm_out @ self.W_attn).squeeze(-1) + self.b_attn
        raw_attn[~mask] = -1e9
        exp_attn = np.exp(raw_attn - np.max(raw_attn, axis=-1, keepdims=True))
        attn_weights_sliced = exp_attn / np.clip(np.sum(exp_attn, axis=-1, keepdims=True), 1e-9, None)
        
        pooled = np.sum(bilstm_out * attn_weights_sliced[:, :, np.newaxis], axis=1)  # [batch, 2 * hidden_dim]
        
        # Dense classification head
        dense = np.maximum(0, pooled @ self.W_fc1 + self.b_fc1)  # ReLU
        logit = (dense @ self.W_out).squeeze(-1) + self.b_out
        probs = self._sigmoid(logit)

        if seq_len < full_seq_len:
            pad_width = full_seq_len - seq_len
            attn_weights = np.pad(attn_weights_sliced, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
        else:
            attn_weights = attn_weights_sliced

        return probs, pooled, attn_weights

    def predict(self, input_ids: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """Inference for input IDs."""
        probs, pooled, attn = self.forward(input_ids)
        labels = (probs >= threshold).astype(int)
        str_labels = ["Fraudulent" if l == 1 else "Legitimate" for l in labels]
        return {
            "probabilities": probs,
            "predictions": labels,
            "risk_labels": str_labels,
            "embeddings": pooled,
            "attention_weights": attn
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 8,
        batch_size: int = 32,
        lr: float = 0.015,
        patience: int = 3
    ) -> Dict[str, List[float]]:
        """Gradient descent optimization for classification head and attention."""
        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        num_samples = len(X_train)
        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        for epoch in range(epochs):
            indices = self.rng.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            epoch_loss = 0.0
            num_batches = int(np.ceil(num_samples / batch_size))

            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, num_samples)
                X_b = X_shuffled[start:end]
                y_b = y_shuffled[start:end]

                probs, pooled, _ = self.forward(X_b)
                eps = 1e-9
                loss = -np.mean(y_b * np.log(probs + eps) + (1 - y_b) * np.log(1 - probs + eps))
                epoch_loss += loss

                # Backprop to classification head
                d_logits = (probs - y_b) / len(y_b)
                dense = np.maximum(0, pooled @ self.W_fc1 + self.b_fc1)
                
                dW_out = dense.T @ d_logits[:, np.newaxis]
                db_out = np.sum(d_logits)

                d_dense = d_logits[:, np.newaxis] @ self.W_out.T
                d_dense[dense <= 0] = 0.0
                dW_fc1 = pooled.T @ d_dense
                db_fc1 = np.sum(d_dense, axis=0)

                self.W_out -= lr * dW_out
                self.b_out -= lr * db_out
                self.W_fc1 -= lr * dW_fc1
                self.b_fc1 -= lr * db_fc1

            train_loss = epoch_loss / max(num_batches, 1)
            val_probs, _, _ = self.forward(X_val)
            val_loss = -np.mean(y_val * np.log(val_probs + 1e-9) + (1 - y_val) * np.log(1 - val_probs + 1e-9))
            val_preds = (val_probs >= 0.5).astype(int)
            
            tp = np.sum((val_preds == 1) & (y_val == 1))
            fp = np.sum((val_preds == 1) & (y_val == 0))
            fn = np.sum((val_preds == 0) & (y_val == 1))
            prec = tp / max(tp + fp, 1e-9)
            rec = tp / max(tp + fn, 1e-9)
            f1 = 2 * (prec * rec) / max(prec + rec, 1e-9)

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_f1"].append(float(f1))

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_weights = (self.W_out.copy(), self.b_out, self.W_fc1.copy(), self.b_fc1.copy())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_weights is not None:
            self.W_out, self.b_out, self.W_fc1, self.b_fc1 = best_weights
            
        return history

    def save(self, filepath: str):
        """Saves model weights."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(
            filepath,
            embedding_weights=self.embedding_weights,
            W_x_fwd=self.W_x_fwd,
            W_h_fwd=self.W_h_fwd,
            b_fwd=self.b_fwd,
            W_x_bwd=self.W_x_bwd,
            W_h_bwd=self.W_h_bwd,
            b_bwd=self.b_bwd,
            W_attn=self.W_attn,
            b_attn=self.b_attn,
            W_fc1=self.W_fc1,
            b_fc1=self.b_fc1,
            W_out=self.W_out,
            b_out=self.b_out,
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dense_dim=self.dense_dim,
            seed=self.seed
        )

    @classmethod
    def load(cls, filepath: str) -> "PureBiLSTMTextModel":
        """Loads model weights from .npz file."""
        data = np.load(filepath)
        model = cls(
            vocab_size=int(data["vocab_size"]),
            embedding_dim=int(data["embedding_dim"]),
            hidden_dim=int(data["hidden_dim"]),
            dense_dim=int(data["dense_dim"]),
            seed=int(data["seed"])
        )
        model.embedding_weights = data["embedding_weights"]
        model.W_x_fwd = data["W_x_fwd"]
        model.W_h_fwd = data["W_h_fwd"]
        model.b_fwd = data["b_fwd"]
        model.W_x_bwd = data["W_x_bwd"]
        model.W_h_bwd = data["W_h_bwd"]
        model.b_bwd = data["b_bwd"]
        model.W_attn = data["W_attn"]
        model.b_attn = float(data["b_attn"])
        model.W_fc1 = data["W_fc1"]
        model.b_fc1 = data["b_fc1"]
        model.W_out = data["W_out"]
        model.b_out = float(data["b_out"])
        return model
