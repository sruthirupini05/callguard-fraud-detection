"""
Unit tests for text normalizer.
"""

import pytest
from src.data.normalizer import TextNormalizer


def test_normalizer_basic():
    norm = TextNormalizer(lowercase=True, remove_punct=True)
    raw = "Hello Sir! [inaudible] Please share the OTP now... (cough)"
    clean = norm.clean(raw)
    assert "[inaudible]" not in clean
    assert "(cough)" not in clean
    assert "otp" in clean
    assert norm.tokenize(clean) == ["hello", "sir", "please", "share", "the", "otp", "now"]


def test_normalizer_multilingual():
    norm = TextNormalizer(lowercase=True)
    hindi_text = "Namaskar sir, OTP share kijiye."
    telugu_text = "నమస్కారం అండి, OTP చెప్పండి."
    
    assert "otp" in norm.clean(hindi_text)
    assert "otp" in norm.clean(telugu_text)
    assert len(norm.tokenize(telugu_text)) >= 3


def test_normalizer_empty_and_null():
    norm = TextNormalizer()
    assert norm.clean(None) == ""
    assert norm.clean("") == ""
    assert norm.tokenize(None) == []
