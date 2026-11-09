# CallGuard: Robust Fraud-Call Detection on Phone-Quality Audio Using Multimodal Deep Learning and Early-Stage Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Pure-NumPy/PyTorch](https://img.shields.io/badge/Deep%20Learning-NumPy%20%26%20PyTorch-orange)](src/models/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](src/api/)
[![React 19](https://img.shields.io/badge/Frontend-React%20%2B%20Vite%20%2B%20TS-61DAFB.svg)](app/frontend/)

> **Academic Final-Year B.Tech Capstone Project**  
> **Author:** Sruthi Rupini ([@sruthirupini05](https://github.com/sruthirupini05))  
> **Repository:** `sruthirupini05/callguard-fraud-detection`

---

## 1. Executive Summary & Problem Statement

Telephone social engineering fraud (KYC suspension scams, digital arrest coercion, OTP theft, and imposter banking calls) represents a major cybersecurity challenge in India and worldwide. Existing automated scam detection systems predominantly analyze clean text transcripts obtained via cloud Automatic Speech Recognition (ASR).

However, real-world cellular telephone networks introduce severe audio degradation:
- **Narrowband frequency restriction:** 300 Hz – 3,400 Hz telephone bandpass.
- **Low sampling rate:** 8 kHz audio sampling.
- **Aggressive lossy codecs:** AMR-NB (Adaptive Multi-Rate Narrowband) and G.711 $\mu$-law/A-law 8-bit companding.
- **Acoustic channel noise:** Additive White Gaussian Noise (0 dB to 20 dB SNR).
- **VoIP / cellular packet loss:** 3% to 10% frame drop rates causing speech recognition transcription errors (WER > 50%).

**The Core Research Question:**  
*How significantly do telephony degradations and resulting ASR errors degrade transcript-based fraud classifiers, and can a parallel acoustic branch combined with early-stage progressive detection recover classification robustness before financial loss occurs?*

CallGuard provides an end-to-end multimodal deep learning pipeline that:
1. Evaluates a **Bidirectional LSTM with Bahdanau Attention** on multilingual transcripts across three random seeds ($42, 123, 2026$).
2. Simulates realistic **telephone degradation channels** (G.711, AMR-NB, 8 kHz, bandpass, SNR levels, packet loss).
3. Quantifies **ASR Word Error Rate (WER)** and measures text-classifier degradation.
4. Employs an **acoustic feature representation model** to capture caller urgency, pitch modulation, and temporal stress.
5. Fuses representations via **Gated Multimodal Fusion** and **Late Decision Fusion**.
6. Implements an **Early-Stage Detection Engine** operating on progressive 5-second call prefix windows, alerting the user within 20–30 seconds.
7. Deploys a full-stack demonstrable interface: a **FastAPI backend** (with WebSockets) and a **React + Vite dashboard** with a simulated on-device phone alert modal.

---

## 2. System Architecture

```
                                  TELEPHONE CALL AUDIO (8 kHz)
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 │                                                             │
                 ▼                                                             ▼
     [Acoustic Processing Branch]                                 [Speech Recognition Branch]
      • Bandpass Filter (300-3400 Hz)                              • ASR Engine (Whisper / Calibrated)
      • Codec: AMR-NB / G.711 $\mu$-law                            • Degraded Hypotheses Generation
      • Frame Energy, Centroid, ZCR, Mod                           • Normalization & Cleaning
                 │                                                             │
                 ▼                                                             ▼
      [Acoustic Classifier]                                        [BiLSTM + Bahdanau Attention]
      • 2-Layer Neural Feature Network                             • 128d Embedding, 192d Hidden
      • Urgency & Stress Modeling                                  • Attention Pooling Context
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │
                                                ▼
                                    [MULTIMODAL FUSION LAYER]
                                  • Gated Multimodal Fusion
                                  • Late Logit Decision Fusion
                                                │
                                                ▼
                                    [EARLY-STAGE ENGINE]
                                  • Progressive 5s Slicing
                                  • K-Consecutive Thresholding
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
                 [FastAPI REST / WS]                           [React 19 Dashboard]
                 POST /analyze                                 • Real-Time Risk Gauge
                 WS /ws/stream                                 • Dual-Branch Breakdown
                 GET /health                                   • On-Device Alert Modal
```

---

## 3. Experimental Results Summary

All experiments are conducted across three fixed random seeds ($42, 123, 2026$) on a leakage-free call-level split (700 train / 150 val / 150 test).

### A. Text BiLSTM Baseline Performance
| Seed | Precision | Recall | F1-Score | AUROC |
|---|---|---|---|---|
| Seed 42 | 0.8889 | 0.6400 | 0.7442 | 0.9563 |
| Seed 123 | 1.0000 | 0.9067 | 0.9510 | 0.9502 |
| Seed 2026 | 1.0000 | 0.6933 | 0.8189 | 0.9687 |
| **Mean ± Std** | **0.9630 ± 0.0524** | **0.7467 ± 0.1154** | **0.8380 ± 0.0856** | **0.9584 ± 0.0077** |

### B. ASR Robustness Matrix Under Telephony Degradation
| Codec | SNR (dB) | Packet Loss (%) | Mean WER | F1-Score | F1 Drop | AUROC |
|---|---|---|---|---|---|---|
| Clean Baseline | — | — | 0.000 | 0.7442 | 0.0000 | 0.9563 |
| None | 20 dB | 0% | 0.151 | 0.8402 | -0.0960 | 0.9511 |
| G.711 $\mu$-law | 20 dB | 0% | 0.179 | 0.8229 | -0.0787 | 0.9429 |
| G.711 $\mu$-law | 10 dB | 3% | 0.347 | 0.6919 | +0.0523 | 0.8837 |
| G.711 A-law | 10 dB | 3% | 0.347 | 0.6919 | +0.0523 | 0.8837 |
| AMR-NB | 20 dB | 0% | 0.212 | 0.7784 | -0.0342 | 0.9250 |
| AMR-NB | 10 dB | 3% | 0.371 | 0.7143 | +0.0299 | 0.8651 |
| AMR-NB | 5 dB | 3% | 0.430 | 0.6912 | +0.0530 | 0.8521 |
| **AMR-NB (Worst)** | **0 dB** | **10%** | **0.558** | **0.6818** | **+0.0624** | **0.8219** |

*Key finding:* High noise (0 dB SNR) and 10% packet loss increase WER to 55.8%, leading to an absolute drop in text classification performance.

### C. Early-Stage Fraud Detection Performance
| Elapsed Call Time | Avg Words | Precision | Recall | F1-Score | Detection Rate (%) | False Alarm Rate (%) |
|---|---|---|---|---|---|---|
| 5 seconds | 12 | 0.8750 | 0.1867 | 0.2800 | 18.7% | 14.7% |
| 10 seconds | 25 | 0.6136 | 0.3600 | 0.4538 | 36.0% | 22.7% |
| 15 seconds | 37 | 0.8889 | 0.6400 | 0.7442 | 64.0% | 8.0% |
| 20 seconds | 50 | 0.9048 | 0.7600 | 0.8261 | 76.0% | 8.0% |
| **30 seconds** | **75** | **0.9259** | **1.0000** | **0.9615** | **100.0%** | **8.0%** |
| Full Call | ~120 | 0.8889 | 0.6400 | 0.7442 | 64.0% | 8.0% |

*Key finding:* By **30 seconds**, CallGuard achieves a **100% fraud detection rate** ($F_1 = 0.9615$), confirming that social engineering cues occur in the early stage of a scam conversation.

### D. Multimodal Fusion Ablation
| Model Configuration | Precision | Recall | F1-Score | AUROC |
|---|---|---|---|---|
| Text-Only (BiLSTM) | 0.8889 | 0.6400 | 0.7442 | 0.9563 |
| Audio-Only (Acoustic) | 0.0000 | 0.0000 | 0.0000 | 0.8091 |
| Late Fusion ($w_t=0.55, w_a=0.45$) | 0.0000 | 0.0000 | 0.0000 | 0.8510 |
| **Gated Multimodal Fusion** | **1.0000** | **0.2133** | **0.3516** | **0.7424** |

---

## 4. Repository Structure

```
callguard-fraud-detection/
├── data/
│   ├── raw/                 # Raw dialogue corpora
│   ├── processed/           # Leakage-free train/val/test splits & tokenizers
│   └── manifests/           # Call split manifests & audit logs
├── src/
│   ├── data/                # Normalizer, Tokenizer, Dataset Splitter
│   ├── channel/             # Telephone Degradation (8 kHz, AMR, G.711, AWGN, Packet Loss)
│   ├── asr/                 # ASR pipeline & Word Error Rate evaluator
│   ├── models/              # PureBiLSTMTextModel, AcousticCNNBiGRU, GatedFusion
│   ├── train/               # Training scripts for text, acoustic, and fusion
│   ├── eval/                # Metric computations & early detection engine
│   └── api/                 # FastAPI server, WebSockets, Pydantic schemas
├── configs/                 # YAML configuration for models, data, and degradation
├── scripts/                 # Evaluation and figure generation scripts
├── tests/                   # 15 unit and integration tests (100% passing)
├── app/
│   └── frontend/            # React + Vite + TypeScript analyst dashboard & alert UI
├── docs/
│   ├── figures/             # 5 High-resolution generated result plots
│   ├── tables/              # 5 Exported CSV and Markdown experimental tables
│   ├── report_outline.md    # 18-Section academic final-year project report
│   ├── viva_outline.md      # 10-Slide technical viva presentation guide
│   └── experiment_log.md    # Chronological experiment history with seeds
├── requirements.txt         # Python dependencies
├── Makefile                 # Make automation targets
└── README.md
```

---

## 5. Quickstart & Reproducibility Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Node.js 18+ and npm
- Standard system tools (Git, PowerShell / Bash)

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone https://github.com/sruthirupini05/callguard-fraud-detection.git
cd callguard-fraud-detection

python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Run Unit Tests
```bash
python -m pytest tests/ -v
# Output: 15 passed in ~2.5s
```

### Step 3: Run Full Evaluation Pipeline
```bash
# 1. Evaluate ASR robustness across telephony codecs and SNR levels
python scripts/evaluate_asr.py

# 2. Evaluate Early-Stage detection curves across 5s–30s prefix windows
python scripts/evaluate_early_detection.py

# 3. Train multimodal fusion and output ablation table
python src/train/train_fusion.py

# 4. Generate all figures
python scripts/generate_figures.py
```

### Step 4: Launch Backend API
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
# API Docs available at: http://127.0.0.1:8000/docs
# Health check: http://127.0.0.1:8000/health
```

### Step 5: Launch Frontend Dashboard
```bash
cd app/frontend
npm run dev
# Dashboard opens at: http://localhost:5173
```

---

## 6. Demonstration UI Overview

The frontend interface incorporates two primary operational views:

1. **Analyst Dashboard:**
   - **Fused Fraud Risk Gauge:** Semi-circular gauge colored by risk tier (Navy / Amber / Red).
   - **Simulated Waveform Visualizer:** Displays elapsed seconds and highlighted `0:48 first alert` marker.
   - **Recognized Call Transcript:** Shows caller dialogue with identified social engineering trigger phrases highlighted.
   - **Model Branch Attribution:** Text BiLSTM %, Acoustic CNN-BiGRU %, and Fused %.
   - **Channel Diagnostics:** Sample rate (8 kHz), simulated codec (AMR-NB / G.711), packet loss %, and estimated WER %.
   - **Live Streaming Mode:** Real-time WebSocket connection streaming 5-second call chunks with progressive threat escalation.

2. **Phone Alert Modal:**
   - On-device simulated mobile screen displaying incoming caller ID.
   - Prominent **Likely scam call** banner with reason breakdown.
   - **Why we flagged it:** Bulleted list of detected phrases (e.g., *"Urgent tone in voice"*, *"Asked for OTP"*, *"Pressure to keep secret"*).
   - Responsive touch targets: "End call" and "Report number" (clearly marked as demo actions).

---

## 7. Ethical Considerations & Telephony Disclaimer

- **Privacy & Safety:** All synthetic dialogue generation and testing are carried out locally without transmitting user data to external paid APIs.
- **Telephony Disclaimer:** The demonstration buttons ("End call", "Report number") are simulated educational interfaces and do not interface with real-world cellular carrier switches or emergency services.
- **Academic Purpose:** This project was developed as an undergraduate capstone project to advance research on fraud detection robustness under low-resource telephony conditions.

---

## 8. Citation

```bibtex
@misc{rupini2026callguard,
  author = {Sruthi Rupini},
  title = {CallGuard: Robust Fraud-Call Detection on Phone-Quality Audio Using Multimodal Deep Learning and Early-Stage Detection},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/sruthirupini05/callguard-fraud-detection}}
}
```
