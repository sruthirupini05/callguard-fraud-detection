"""
Multimodal Fusion Module for CallGuard Fraud Detection.
Implements:
1. Late Decision Fusion (Logit weighting with temperature scaling)
2. Learned Contextual Gated Fusion (Dynamic weighting based on ASR confidence/acoustic features)
3. Ablation comparison framework (Text vs Audio vs Late vs Gated)
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class LateFusion:
    """Late logit fusion with configurable branch weights and temperature scaling."""

    def __init__(self, w_text: float = 0.55, w_audio: float = 0.45, temperature: float = 1.0):
        self.w_text = w_text
        self.w_audio = w_audio
        self.temperature = max(temperature, 1e-4)

    def _prob_to_logit(self, p: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p_clipped / (1.0 - p_clipped))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def fuse(self, prob_text: np.ndarray, prob_audio: np.ndarray) -> np.ndarray:
        """
        Combines branch probabilities via logit weighting.
        P_fused = sigmoid((w_t * logit_t + w_a * logit_a) / T)
        """
        logit_text = self._prob_to_logit(prob_text)
        logit_audio = self._prob_to_logit(prob_audio)
        fused_logit = (self.w_text * logit_text + self.w_audio * logit_audio) / self.temperature
        return self._sigmoid(fused_logit)


class GatedMultimodalFusion:
    """
    Learned Gated Fusion mechanism.
    Dynamically balances textual vs acoustic representations:
    g = sigmoid(W_g [h_t; h_a] + b_g)
    h_fused = g * (W_t h_t) + (1 - g) * (W_a h_a)
    P_fraud = sigmoid(W_out h_fused + b_out)
    """

    def __init__(
        self,
        dim_text: int = 192,
        dim_audio: int = 64,
        fused_dim: int = 64,
        seed: int = 42
    ):
        self.dim_text = dim_text
        self.dim_audio = dim_audio
        self.fused_dim = fused_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Projection matrices
        self.W_t = self.rng.uniform(-0.1, 0.1, (dim_text, fused_dim))
        self.W_a = self.rng.uniform(-0.1, 0.1, (dim_audio, fused_dim))
        
        # Gating network
        self.W_g = self.rng.uniform(-0.1, 0.1, (dim_text + dim_audio, fused_dim))
        self.b_g = np.zeros(fused_dim)
        
        # Output classifier
        self.W_out = self.rng.uniform(-0.1, 0.1, (fused_dim, 1))
        self.b_out = 0.0

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def fuse(self, h_text: np.ndarray, h_audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        h_text: [batch_size, dim_text]
        h_audio: [batch_size, dim_audio]
        Returns:
            fused_prob: [batch_size]
            gate_values: [batch_size, fused_dim]
            fused_emb: [batch_size, fused_dim]
        """
        # Ensure 2D
        if h_text.ndim == 1:
            h_text = h_text[np.newaxis, :]
        if h_audio.ndim == 1:
            h_audio = h_audio[np.newaxis, :]
            
        proj_t = h_text @ self.W_t
        proj_a = h_audio @ self.W_a
        
        concat = np.concatenate([h_text, h_audio], axis=-1)
        gate = self._sigmoid(concat @ self.W_g + self.b_g)  # [batch_size, fused_dim]
        
        fused_emb = gate * proj_t + (1.0 - gate) * proj_a
        logit = (fused_emb @ self.W_out).squeeze(-1) + self.b_out
        fused_prob = self._sigmoid(logit)
        return fused_prob, gate, fused_emb

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(
            filepath,
            W_t=self.W_t,
            W_a=self.W_a,
            W_g=self.W_g,
            b_g=self.b_g,
            W_out=self.W_out,
            b_out=self.b_out,
            dim_text=self.dim_text,
            dim_audio=self.dim_audio,
            fused_dim=self.fused_dim,
            seed=self.seed
        )

    @classmethod
    def load(cls, filepath: str) -> "GatedMultimodalFusion":
        data = np.load(filepath)
        model = cls(
            dim_text=int(data["dim_text"]),
            dim_audio=int(data["dim_audio"]),
            fused_dim=int(data["fused_dim"]),
            seed=int(data["seed"])
        )
        model.W_t = data["W_t"]
        model.W_a = data["W_a"]
        model.W_g = data["W_g"]
        model.b_g = data["b_g"]
        model.W_out = data["W_out"]
        model.b_out = float(data["b_out"])
        return model
