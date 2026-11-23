"""
Synthetic Speech Acoustic Generator for CallGuard Research Demonstration Dataset.
Synthesizes valid 16 kHz mono 16-bit PCM WAV recordings for:
1. call_001_benign.wav - Legitimate Broadband Fiber Customer Support Call
2. call_002_suspicious.wav - Urgent Verification & Credit Points Expiration Call
3. call_003_high_risk.wav - SBI Impersonation & Coercive OTP Extraction Call
Generates data/sample_calls/metadata.csv and data/sample_calls/README.md.
"""

import os
import io
import math
import wave
import numpy as np
import pandas as pd
from scipy.io import wavfile


def synthesize_conversational_turn(
    duration: float,
    pitch_f0: float,
    formants: list,
    stress_rate: float = 3.5,
    is_agitated: bool = False,
    sr: int = 16000,
    seed: int = 42
) -> np.ndarray:
    """
    Synthesizes speech-like acoustic signals with harmonic glottal pulses,
    vocal tract formant resonance, and unvoiced consonantal noise.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Glottal source pulse train with slight jitter and pitch modulation
    f0_mod = pitch_f0 * (1.0 + 0.08 * np.sin(2 * np.pi * stress_rate * t))
    if is_agitated:
        # Scam caller agitation: higher pitch jitter & vocal tension
        f0_mod += rng.normal(0, 12.0, n_samples)
    
    phase = np.cumsum(2 * np.pi * f0_mod / sr)
    # Glottal pulse (Rosenberg glottal model approximation)
    glottal = np.sin(phase) + 0.45 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase) + 0.15 * np.sin(4 * phase)

    # Vocal tract resonance via formant filtering
    speech = np.zeros(n_samples, dtype=np.float32)
    for f_res, bw, gain in formants:
        decay = math.exp(-math.pi * bw / sr)
        omega = 2 * math.pi * f_res / sr
        a1 = -2.0 * decay * math.cos(omega)
        a2 = decay * decay
        b0 = gain * (1.0 - decay)

        # 2nd-order IIR formant resonator
        y = np.zeros(n_samples)
        for i in range(2, n_samples):
            y[i] = b0 * glottal[i] - a1 * y[i - 1] - a2 * y[i - 2]
        speech += y.astype(np.float32)

    # Unvoiced consonantal turbulence & aspiration noise
    noise = rng.normal(0, 0.08, n_samples)
    speech += (noise * (np.sin(2 * np.pi * stress_rate * 2 * t) ** 2)).astype(np.float32)

    # Syllabic amplitude envelope (speech cadence)
    syllables = np.abs(np.sin(2 * np.pi * stress_rate * t)) ** 1.6
    speech *= syllables.astype(np.float32)

    # Ambient room acoustics
    speech += rng.normal(0, 0.015, n_samples).astype(np.float32)

    # Normalize to 0.90 peak
    max_amp = np.max(np.abs(speech)) + 1e-9
    return (speech / max_amp * 0.88).astype(np.float32)


def generate_sample_call_audio(
    turns: list,
    output_path: str,
    sr: int = 16000
):
    """
    Combines conversational speaker turns with realistic speech pauses.
    """
    audio_segments = []
    # Formants for Speaker A (Male caller: F1~600, F2~1600, F3~2600)
    caller_formants = [(600, 80, 1.0), (1600, 110, 0.6), (2600, 150, 0.35)]
    # Formants for Speaker B (Female/User: F1~750, F2~1950, F3~2850)
    user_formants = [(750, 90, 1.0), (1950, 120, 0.65), (2850, 160, 0.4)]

    for idx, (speaker, duration, is_agitated) in enumerate(turns):
        if speaker == "CALLER":
            f0 = 240 if is_agitated else 165
            formants = caller_formants
            seed = 100 + idx
        else:
            f0 = 215
            formants = user_formants
            seed = 200 + idx

        turn_wave = synthesize_conversational_turn(
            duration=duration,
            pitch_f0=f0,
            formants=formants,
            stress_rate=4.2 if is_agitated else 3.2,
            is_agitated=is_agitated,
            sr=sr,
            seed=seed
        )
        audio_segments.append(turn_wave)

        # Conversational gap (0.4s to 0.7s)
        gap_samples = int(0.45 * sr)
        audio_segments.append(np.zeros(gap_samples, dtype=np.float32))

    full_audio = np.concatenate(audio_segments)
    int16_audio = (np.clip(full_audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wavfile.write(output_path, sr, int16_audio)
    return len(full_audio) / float(sr)


def build_dataset():
    audio_dir = "data/sample_calls/audio"
    os.makedirs(audio_dir, exist_ok=True)

    # 1. Benign Call (Customer support broadband ticket follow-up)
    call_001_turns = [
        ("CALLER", 4.2, False),  # Good afternoon, ACT Fibernet customer care...
        ("USER", 3.0, False),    # Yes Priya, the technician arrived...
        ("CALLER", 3.2, False),  # Wonderful! Is internet speed working...
        ("USER", 2.8, False),    # Yes all lights green speed 100mbps...
        ("CALLER", 3.5, False)   # Thank you confirming no action required...
    ]
    t1 = generate_sample_call_audio(call_001_turns, os.path.join(audio_dir, "call_001_benign.wav"))

    # 2. Suspicious Call (Urgent credit reward points expiry)
    call_002_turns = [
        ("CALLER", 4.5, True),   # Sir your credit card reward points expiring in 15 mins...
        ("USER", 2.6, False),    # How can I redeem them?
        ("CALLER", 4.8, True),   # To redeem sending OTP please verify immediately...
        ("USER", 2.5, False),    # But OTP is personal...
        ("CALLER", 4.2, True)    # Sir this is only reward redemption token hurry up...
    ]
    t2 = generate_sample_call_audio(call_002_turns, os.path.join(audio_dir, "call_002_suspicious.wav"))

    # 3. High-Risk Scam Call (SBI Fraud Department KYC OTP threat)
    call_003_turns = [
        ("CALLER", 5.2, True),   # Sir Ramesh from SBI fraud prevention unauthorized 45000 debit...
        ("USER", 3.2, False),    # What? I am at work did not do transfer!
        ("CALLER", 5.6, True),   # Do not panic account will be blocked share OTP immediately...
        ("USER", 2.5, False),    # Which branch are you calling from?
        ("CALLER", 5.8, True)    # No time security server verification do not disconnect read OTP!
    ]
    t3 = generate_sample_call_audio(call_003_turns, os.path.join(audio_dir, "call_003_high_risk.wav"))

    # Metadata records
    records = [
        {
            "call_id": "CALL-001",
            "filename": "call_001_benign.wav",
            "duration_seconds": round(t1, 2),
            "sample_rate": 16000,
            "channels": 1,
            "format": "WAV (PCM 16-bit)",
            "ground_truth_label": 0,
            "category": "Benign Customer Care Followup",
            "scam_type": "Legitimate Support",
            "reference_transcript": (
                "CALLER: Good afternoon, this is Priya from ACT Fibernet customer care. I am calling to follow up on the broadband fiber ticket from yesterday. "
                "YOU: Yes Priya, the technician arrived at 2 PM and replaced the router. "
                "CALLER: Wonderful! Is your internet speed working properly now? "
                "YOU: Yes, all lights are green and speed is back to 100 Mbps. "
                "CALLER: Thank you for confirming sir. No further action is required from your side. Have a pleasant day!"
            ),
            "is_synthetic": True,
            "acoustic_synthesis_method": "Harmonic Vocal Tract Formant Resonator Model"
        },
        {
            "call_id": "CALL-002",
            "filename": "call_002_suspicious.wav",
            "duration_seconds": round(t2, 2),
            "sample_rate": 16000,
            "channels": 1,
            "format": "WAV (PCM 16-bit)",
            "ground_truth_label": 1,
            "category": "Urgent Reward Phishing",
            "scam_type": "Credit Card Expiry Urgency",
            "reference_transcript": (
                "CALLER: Sir aapke HDFC Credit Card ke 8,500 reward points expire hone wale hain within 15 minutes. "
                "YOU: Acha, kaise redeem kar sakte hain? "
                "CALLER: Sir redeem karne ke liye main security gateway se OTP bhej raha hoon, please verify kar dijiye immediately warna points forfeit ho jayenge. "
                "YOU: Lekin OTP toh personal hota hai. "
                "CALLER: Sir yeh sirf redemption token hai, do not worry hurry up right now or points zero ho jayenge!"
            ),
            "is_synthetic": True,
            "acoustic_synthesis_method": "Harmonic Vocal Tract Formant Resonator Model"
        },
        {
            "call_id": "CALL-003",
            "filename": "call_003_high_risk.wav",
            "duration_seconds": round(t3, 2),
            "sample_rate": 16000,
            "channels": 1,
            "format": "WAV (PCM 16-bit)",
            "ground_truth_label": 1,
            "category": "Bank Impersonation & Coercive OTP Extraction",
            "scam_type": "Bank OTP / KYC Suspension Fraud",
            "reference_transcript": (
                "CALLER: Sir, this is Ramesh from SBI fraud prevention cell. An unauthorized debit of 45,000 rupees was just attempted on your card. "
                "YOU: What? I am at work, I did not make any transaction! "
                "CALLER: Do not panic sir, but your account will be blocked today permanently unless you share the OTP immediately to cancel this transfer. "
                "YOU: Which bank branch are you calling from? "
                "CALLER: Sir there is no time! This is security server automated verification. Do not disconnect the call, just read the six digits OTP right now or account freeze will occur!"
            ),
            "is_synthetic": True,
            "acoustic_synthesis_method": "Harmonic Vocal Tract Formant Resonator Model"
        }
    ]

    df = pd.DataFrame(records)
    csv_path = "data/sample_calls/metadata.csv"
    df.to_csv(csv_path, index=False)

    readme_content = """# CallGuard Synthetic Demonstration Audio Dataset

## Overview
This directory contains three standard demonstration telephone calls created for testing the audio ingestion pipeline, ASR transcription, speech feature extraction, and multimodal fraud classification.

> **Research Integrity Notice:** All audio files in this directory are **synthetic demonstration recordings** generated using acoustic vocal-tract formant synthesis with conversational turn modeling. They are provided strictly for functional testing and interface demonstration. They are **not** real-world intercepted phone calls and are not used for evaluating published academic benchmark metrics.

## Files
- `audio/call_001_benign.wav`: 16 kHz Mono WAV, Benign customer support conversation (~18s).
- `audio/call_002_suspicious.wav`: 16 kHz Mono WAV, Urgent credit card rewards redemption phishing call (~18s).
- `audio/call_003_high_risk.wav`: 16 kHz Mono WAV, SBI KYC impersonation with aggressive OTP extortion (~24s).
- `metadata.csv`: Ground-truth labels, reference transcripts, duration, and acoustic parameters.

## Academic Research Dataset Loading
For large-scale evaluation on empirical research datasets, place licensed corpora (e.g., Telecom Fraud Corpora, CommonVoice Telephone Degradations) into `data/manifests/call_splits.csv` and execute `python scripts/batch_evaluate.py`.
"""
    with open("data/sample_calls/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("Successfully created synthetic demonstration dataset in data/sample_calls/")
    print(f"- call_001_benign.wav: {t1:.1f}s")
    print(f"- call_002_suspicious.wav: {t2:.1f}s")
    print(f"- call_003_high_risk.wav: {t3:.1f}s")


if __name__ == "__main__":
    build_dataset()
