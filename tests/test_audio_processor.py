"""
Unit tests for AudioProcessor, Early Detection, and Channel Lab API endpoints.
"""

import io
import pytest
import numpy as np
from scipy.io import wavfile
from fastapi.testclient import TestClient

from src.api.main import app
from src.audio.processor import AudioProcessor, AudioProcessingError


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_wav_bytes():
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Speech-like audio
    sig = (0.5 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.random.randn(len(t))).astype(np.float32)
    int16_sig = (sig * 32767.0).astype(np.int16)
    
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16_sig)
    buf.seek(0)
    return buf.read()


def test_audio_processor_decode_valid(valid_wav_bytes):
    proc = AudioProcessor(target_sr_asr=16000, target_sr_phone=8000)
    res = proc.decode_and_validate(valid_wav_bytes, "test.wav")
    assert res["original_sample_rate"] == 16000
    assert res["original_channels"] == 1
    assert res["duration_seconds"] >= 1.9
    assert len(res["audio_16k"]) > 0
    assert len(res["audio_8k"]) > 0
    assert len(res["waveform_envelope"]) == 100
    assert res["snr_db"] > 0


def test_audio_processor_corrupted_file():
    proc = AudioProcessor()
    with pytest.raises(AudioProcessingError):
        proc.decode_and_validate(b"RIFF\x00\x00\x00\x00corrupted_bytes_only", "corrupted.wav")


def test_audio_processor_vad_segments(valid_wav_bytes):
    proc = AudioProcessor()
    res = proc.decode_and_validate(valid_wav_bytes, "test.wav")
    assert len(res["vad_segments"]) > 0
    assert 0.0 <= res["speech_ratio"] <= 1.0


def test_early_detection_endpoint(client):
    resp = client.post("/early_detection", data={"call_id": "TEST-01"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["call_id"] == "TEST-01"
    assert len(data["prefix_evaluations"]) >= 4
    assert data["time_saved_seconds"] >= 0.0


def test_channel_lab_endpoint(client):
    resp = client.post(
        "/channel_lab",
        data={
            "transcript": "CALLER: Share the OTP immediately or your account will be blocked.",
            "codec": "amr_nb",
            "snr_db": "5.0",
            "packet_loss": "0.03"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "measured_wer" in data
    assert len(data["clean_waveform"]) > 0
    assert len(data["degraded_waveform"]) > 0
    assert "text_trust_shift" in data


def test_sample_calls_endpoint(client):
    resp = client.get("/sample_calls")
    assert resp.status_code == 200
    calls = resp.json()
    assert len(calls) == 3
    assert any(c["filename"] == "call_003_high_risk.wav" for c in calls)
