# CallGuard Synthetic Demonstration Audio Dataset

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
