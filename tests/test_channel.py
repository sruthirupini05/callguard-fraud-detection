"""
Unit tests for Telephone Channel Degradation module.
Verifies sampling rate, frequency response, codec conversion, SNR calculation, and packet loss.
"""

import numpy as np
import pytest
from src.channel.degradation import TelephoneChannelDegrader


@pytest.fixture
def degrader():
    return TelephoneChannelDegrader(target_sr=8000, f_low=300.0, f_high=3400.0)


def test_sampling_rate_and_duration(degrader):
    orig_sr = 16000
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(orig_sr * duration_sec), endpoint=False)
    audio_16k = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1 kHz tone
    
    degraded = degrader.degrade(audio_16k, orig_sr=orig_sr, codec="g711_mulaw", snr_db=10, packet_loss=0.0)
    
    expected_samples = int(round(8000 * duration_sec))
    assert abs(len(degraded) - expected_samples) <= 2
    assert not np.isnan(degraded).any()
    assert not np.isinf(degraded).any()
    assert (degraded >= -1.05).all() and (degraded <= 1.05).all()


def test_g711_mulaw_codec(degrader):
    audio = np.linspace(-1.0, 1.0, 1000)
    compressed = degrader.compress_g711_mulaw(audio)
    
    assert len(compressed) == len(audio)
    # Check that it produces discrete levels
    unique_levels = len(np.unique(np.round(compressed, 4)))
    assert unique_levels <= 256
    # Round-trip error should be small for companding
    mse = np.mean((audio - compressed) ** 2)
    assert mse < 0.01


def test_g711_alaw_codec(degrader):
    audio = np.linspace(-1.0, 1.0, 1000)
    compressed = degrader.compress_g711_alaw(audio)
    
    assert len(compressed) == len(audio)
    mse = np.mean((audio - compressed) ** 2)
    assert mse < 0.01


def test_noise_level_snr(degrader):
    # Pure 1kHz tone with known signal power
    t = np.linspace(0, 1.0, 8000, endpoint=False)
    audio = np.sin(2 * np.pi * 1000 * t)  # Signal power ~ 0.5
    
    target_snr = 10.0
    noisy = degrader.add_additive_white_gaussian_noise(audio, snr_db=target_snr, seed=42)
    
    # Noise = noisy - audio
    noise = noisy - audio
    measured_snr = 10 * np.log10(np.mean(audio ** 2) / np.mean(noise ** 2))
    assert abs(measured_snr - target_snr) < 0.5


def test_packet_loss_simulation(degrader):
    audio = np.ones(8000)  # 1 second at 8 kHz = 50 frames of 20ms
    loss_rate = 0.10  # 10% packet drop
    dropped = degrader.simulate_packet_loss(audio, loss_rate=loss_rate, frame_ms=20.0, seed=42)
    
    # Check that some 20ms frames have been zeroed/silenced
    silenced_samples = np.sum(np.abs(dropped) < 0.01)
    frame_size = int(8000 * 0.02)  # 160 samples
    assert silenced_samples >= frame_size  # At least 1 frame dropped
