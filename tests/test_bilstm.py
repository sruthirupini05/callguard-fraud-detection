"""
Unit tests for BiLSTM model architecture, forward pass, attention pooling, and shape consistency.
"""

import numpy as np
import pytest
from src.models.bilstm import PureBiLSTMTextModel
from src.data.tokenizer import CallGuardTokenizer


def test_tokenizer_encode_pad():
    tokenizer = CallGuardTokenizer(max_vocab_size=100, max_seq_length=10)
    texts = ["hello world from callguard", "urgent bank scam alert"]
    tokenizer.fit_on_texts(texts)
    
    encoded = tokenizer.encode_and_pad("hello world alert")
    assert len(encoded) == 10
    assert encoded[0] != 0
    assert encoded[-1] == 0  # padded


def test_bilstm_forward_shapes():
    model = PureBiLSTMTextModel(
        vocab_size=100,
        embedding_dim=32,
        hidden_dim=64,
        dense_dim=32,
        seed=42
    )
    
    # Batch of 4 sequences of length 10
    dummy_input = np.random.randint(0, 100, size=(4, 10))
    probs, pooled, attn = model.forward(dummy_input)
    
    assert probs.shape == (4,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert pooled.shape == (4, 128)  # 2 * hidden_dim
    assert attn.shape == (4, 10)     # Attention weights over seq_len
    # Attention sums to 1.0 per sequence
    np.testing.assert_allclose(np.sum(attn, axis=1), np.ones(4), atol=1e-5)


def test_bilstm_save_and_load(tmp_path):
    model = PureBiLSTMTextModel(vocab_size=50, embedding_dim=16, hidden_dim=32, seed=42)
    save_path = str(tmp_path / "model.npz")
    model.save(save_path)
    
    loaded = PureBiLSTMTextModel.load(save_path)
    dummy_input = np.random.randint(0, 50, size=(2, 8))
    
    p1, _, _ = model.forward(dummy_input)
    p2, _, _ = loaded.forward(dummy_input)
    
    np.testing.assert_allclose(p1, p2, atol=1e-6)
