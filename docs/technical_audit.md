# CallGuard Phase 1: Comprehensive Technical Audit & Research Architecture Assessment

**Author:** Senior AI/ML Research Engineer & Full-Stack Architect  
**Project:** CallGuard — Multimodal Telephone Fraud Detection Platform  
**Target:** Research-Grade Workstation for ASR Robustness, Phone-Channel Degradation & Early Detection  
**Date:** November 2026

---

## 1. Executive Summary

A comprehensive architectural audit of the CallGuard repository (`callguard-fraud-detection`) was conducted across all backend services, machine learning models, frontend interfaces, data pipelines, and evaluation scripts. 

The existing codebase contains solid mathematical foundations:
- Fully operational NumPy implementations of a **Bidirectional LSTM with Bahdanau Attention** and **Learned Gated Multimodal Fusion**.
- Calibrated **Probability Temperature Scaling** ($T = 1.25$) and **Streaming Bayesian Evidence Accumulation**.
- Realistic **ITU-T G.711 $\mu$-law/A-law and 300–3400 Hz bandpass filtering** in `src/channel/degradation.py`.
- Complete multi-seed evaluation logs (Seeds 42, 123, 2026) with zero utterance leakage across splits.

However, several critical limitations prevent it from functioning as a true research-grade workstation:
1. **Audio Upload Pipeline Contains Fallback Placeholder:** In `src/api/main.py` (lines 188–193), uploaded audio bytes were not decoded using PCM audio headers; instead, a synthetic sine-wave was synthesized in memory. Real audio decoding, PCM validation, and true waveform feature extraction are required.
2. **Missing Standalone Audio Processor:** Sample-rate detection, mono conversion, VAD (Voice Activity Detection), background noise estimation, and audio quality metrics were computed heuristically rather than from actual signal frames.
3. **No Synthetic Demonstration Dataset on Disk:** The required `data/sample_calls/` directory with playable, multi-speaker WAV files (`call_001_benign.wav`, `call_002_suspicious.wav`, `call_003_high_risk.wav`) and associated `metadata.csv` was missing.
4. **UI Architecture Needed 12-Section Workstation Structure:** The current single-page dashboard needs to be elevated into a structured 12-view Research Workstation matching the requested dark obsidian / violet aesthetic (inspired by modern observability platforms like Better Stack).

---

## 2. Component-by-Component Audit Findings

| Component | Current State | Technical Implementation | Audit Verdict & Required Action |
| :--- | :--- | :--- | :--- |
| **`src/api/main.py`** | Partially Functional | Loads real `.npz` weights for BiLSTM, Acoustic, and Gated Fusion. Implements live `/analyze`, `/health`, `/feedback`, and WebSocket `/ws/stream`. | **Fix audio decoding:** Replace synthetic sine substitution with actual WAV/PCM decoding, real sample-rate extraction, and real VAD. |
| **`src/audio/`** | Missing | No standalone audio processing module existed. | **Create `src/audio/processor.py`:** Implement true PCM decoding via `scipy.io.wavfile` & `wave`, mono conversion, RMS normalization, VAD, and SNR estimation. |
| **`src/models/bilstm.py`** | Functional | Pure NumPy forward pass, embedding layer, forward/backward LSTM cells, Bahdanau attention pooling, dense output layer. | **Preserve & enhance:** Maintain exact forward pass compatibility; expose intermediate attention weights for explainability. |
| **`src/models/acoustic_cnn_bigru.py`** | Functional | 80-band Mel filterbank, STFT with Hann window, 2D CNN layers, BiGRU, Temporal Attention Pooling. | **Preserve & integrate:** Feed actual preprocessed audio frames from uploaded WAV files directly into `LogMelSpectrogramExtractor`. |
| **`src/models/fusion.py`** | Functional | Late Fusion (logit weighting) and Learned Gated Fusion ($g = \sigma(W_g [h_t; h_a] + b_g)$). | **Preserve & expand:** Expose comparative ablation between Text-Only, Audio-Only, Late Fusion, and Contextual Gated Fusion. |
| **`src/channel/degradation.py`** | Functional | Butterworth 300–3400 Hz bandpass, G.711 $\mu$-law/A-law companding, AMR-NB quantization, AWGN, packet loss. | **Preserve & connect:** Apply directly to uploaded audio in the new Channel Robustness Laboratory. |
| **`src/asr/asr_pipeline.py`** | Functional | Dynamic DP Levenshtein WER computation and language-dependent phone degradation modeling. | **Preserve & enhance:** Align transcript tokens with audio time offsets for time-stamped transcript intelligence. |
| **`src/models/intent_classifier.py`** | Functional | Multi-intent semantic parsing across 5 vectors, speaker turn diarization, anti-warning negation shield. | **Preserve:** Keep active for both English and Indian multilingual contexts. |
| **`data/sample_calls/`** | Missing | No valid, playable demonstration audio files existed in `data/sample_calls/audio/`. | **Generate `data/sample_calls/`:** Create 3 valid, playable 16 kHz mono WAV recordings (`call_001_benign.wav`, `call_002_suspicious.wav`, `call_003_high_risk.wav`) with metadata. |
| **`app/frontend/`** | Functional | React + Vite + TypeScript + Recharts + Tailwind. Single-page layout with modal benchmarks. | **Transform into 12-Section Workstation:** Implement obsidian dark theme with violet/blurple glowing accents (Better Stack style) across all 12 research views. |

---

## 3. Core Research Question Alignment

CallGuard's central thesis:
> *"How do automatic speech recognition (ASR) errors and telephone-channel degradation affect transcript-based telephone fraud detection, and to what extent can an acoustic detection branch combined with early-stage fraud detection recover the resulting performance degradation?"*

To directly address this thesis, the system is structured around three experimental axes:
1. **Ablation Matrix:**
   $$\Delta \text{F1} = \text{F1}_{\text{Gated Multimodal}} - \max(\text{F1}_{\text{Text-Only}}, \text{F1}_{\text{Audio-Only}})$$
   Measures the exact recovery achieved by acoustic paralinguistic features when ASR transcripts deteriorate under 5 dB SNR and AMR-NB compression.
2. **Dynamic Gating Sensitivity:**
   $$g_{\text{text}} = \sigma\left(W_g [h_{\text{text}}; h_{\text{audio}}] + b_g\right)$$
   Demonstrates that as channel SNR drops from 20 dB to 0 dB and packet loss rises to 10%, the gating network adaptively downweights text trust ($g_{\text{text}} \downarrow$) and shifts confidence to acoustic spectral and prosodic stress signals ($1 - g_{\text{text}} \uparrow$).
3. **Prefix Detection Latency:**
   Evaluates fraud recall at prefix windows $\tau \in \{5\text{s}, 10\text{s}, 15\text{s}, 30\text{s}, \text{Full}\}$, validating whether coercive threats and credential demands can be terminated before financial loss occurs.

---

## 4. Prioritised Implementation Plan

- **Phase 2:** Build `src/audio/processor.py` for real PCM decoding, VAD, SNR, clipping detection, and real waveform sampling.
- **Phase 3:** Generate the playable synthetic demonstration dataset in `data/sample_calls/` with valid WAV files and `metadata.csv`.
- **Phase 4:** Upgrade `src/api/main.py` and schemas to handle real binary audio uploads, early prefix slicing, channel laboratory experiments, and ablation comparisons.
- **Phase 5:** Build the complete 12-section Research Workstation UI in `app/frontend/` using the obsidian dark theme with violet accents.
- **Phase 6:** End-to-end testing, pytest verification, build validation, and GitHub push.
