"""
Subword / Word Vocabulary and Tokenizer for CallGuard Text Baseline
Handles OOV tokens, padding, truncation, and vocabulary serialization.
"""

import json
import os
from collections import Counter
from typing import List, Dict, Union
import numpy as np


class CallGuardTokenizer:
    """Configurable text tokenizer with vocabulary building, padding, and truncation."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, max_vocab_size: int = 5000, max_seq_length: int = 128):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.is_fitted = False

    def fit_on_texts(self, texts: List[str]):
        """Builds vocabulary from a corpus of training texts."""
        word_counts = Counter()
        for text in texts:
            tokens = text.lower().split()
            word_counts.update(tokens)
            
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        self.is_fitted = True

    def encode(self, text: str) -> List[int]:
        """Converts text into a list of token IDs."""
        tokens = text.lower().split()
        return [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]

    def encode_and_pad(self, text: str) -> np.ndarray:
        """Encodes text with fixed sequence length, right-padding with 0."""
        token_ids = self.encode(text)
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        else:
            token_ids = token_ids + [0] * (self.max_seq_length - len(token_ids))
        return np.array(token_ids, dtype=np.int64)

    def batch_encode_and_pad(self, texts: List[str]) -> np.ndarray:
        """Batched encoding and padding to 2D numpy array [batch_size, max_seq_length]."""
        return np.array([self.encode_and_pad(t) for t in texts], dtype=np.int64)

    def save(self, filepath: str):
        """Serializes tokenizer config and vocabulary dictionary."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "max_vocab_size": self.max_vocab_size,
            "max_seq_length": self.max_seq_length,
            "word2idx": self.word2idx
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CallGuardTokenizer":
        """Loads serialized tokenizer from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(max_vocab_size=data["max_vocab_size"], max_seq_length=data["max_seq_length"])
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        tokenizer.is_fitted = True
        return tokenizer
