# Academic Project Report Outline
## CallGuard: Robust Fraud-Call Detection on Phone-Quality Audio Using Multimodal Deep Learning and Early-Stage Detection

**Degree:** Bachelor of Technology in Computer Science & Engineering / Information Technology  
**Author:** Sruthi Rupini  
**Institution:** Final-Year B.Tech Capstone Project Evaluation  

---

### Table of Contents

1. **Title Page, Certificate & Declaration**
   - Official university declaration of original research work
   - Supervisor endorsement certificate
   - Acknowledgements

2. **Abstract**
   - High-level context of telecommunication scams in India (KYC, digital arrest, OTP theft)
   - Limitations of clean-transcript ASR models on real telephone channels
   - Multimodal methodology (BiLSTM + Acoustic Feature MLP + Gated Fusion)
   - Key findings: WER impact (up to 55.8% under 0 dB AMR-NB), 100% fraud recall by 30 seconds

3. **Chapter 1: Introduction**
   - 1.1 Background and Telecommunication Fraud Epidemic in India
   - 1.2 Motivation: Why Text-Only Approaches Fail on Real Phone Lines
   - 1.3 Research Problem Statement
   - 1.4 Research Questions and Hypotheses ($H_1$: Telephony degradation significantly increases WER and degrades text classifiers; $H_2$: Acoustic features recover resilience; $H_3$: Fraud patterns surface within the first 30 seconds)
   - 1.5 Research Objectives and Scope of Investigation
   - 1.6 Report Organization

4. **Chapter 2: Literature Review**
   - 2.1 Natural Language Processing for Social Engineering Detection
   - 2.2 Deep Learning Architectures for Sequence Classification (RNN, LSTM, BiLSTM, Transformers)
   - 2.3 Acoustic Signal Processing & Paralinguistic Emotion/Stress Detection
   - 2.4 Telephony Channel Modeling (Narrowband 8 kHz, AMR-NB, G.711 companding)
   - 2.5 Multimodal Fusion Strategies (Early, Late, Gated Contextual Fusion)
   - 2.6 Critical Review of Existing Scam Datasets (ICFD-31k, AntiPhish, TeleScam) & Research Gaps

5. **Chapter 3: Methodology & System Architecture**
   - 3.1 Overall Multimodal System Architecture Pipeline
   - 3.2 Leakage-Free Data Partitioning Methodology (Splitting at Call-ID level to prevent utterance leakage)
   - 3.3 Text Preprocessing, Multilingual Normalization (English, Hindi, Telugu) & Vocabulary Construction
   - 3.4 Speech Synthesis and Audio Simulation Pipeline

6. **Chapter 4: Telecommunication Channel Degradation Simulation**
   - 4.1 Narrowband Telephony Bandpass Filtering (300 Hz – 3400 Hz Butterworth filter)
   - 4.2 Resampling to 8,000 Hz Standard Telephone Rate
   - 4.3 Lossy Companding Codecs: G.711 $\mu$-law and A-law 8-bit Quantization
   - 4.4 Cellular Codec Modeling: AMR-NB (Adaptive Multi-Rate Narrowband LPC approximation)
   - 4.5 Additive White Gaussian Noise (AWGN) Modeling across Controlled SNR (20 dB, 10 dB, 5 dB, 0 dB)
   - 4.6 VoIP Packet Loss Emulation (3% and 10% frame drop simulation)

7. **Chapter 5: Automatic Speech Recognition & Robustness Evaluation**
   - 5.1 Speech-to-Text Pipeline (Whisper-Small and Acoustic Degradation Simulator)
   - 5.2 Dynamic Programming Levenshtein Distance for Word Error Rate (WER) Computation
   - 5.3 Measuring WER across Degradation Parameters
   - 5.4 Characterizing Transcription Degradation on Urgent Key Phrases

8. **Chapter 6: Text-Based Fraud Classification (BiLSTM with Attention)**
   - 6.1 Mathematical Formulation of Bidirectional LSTM
   - 6.2 Bahdanau Additive Attention Pooling Mechanism
   - 6.3 Loss Function (Binary Cross-Entropy with $L_2$ Regularization)
   - 6.4 Training Dynamics, Checkpointing, and Early Stopping

9. **Chapter 7: Acoustic Representation & Feature Engineering Branch**
   - 7.1 Acoustic Correlates of Telecommunication Fraud: Urgency, Fast Tempo, Energy Spikes
   - 7.2 Log-Mel Spectrogram Extraction (80 Mels, STFT, Hamming Window)
   - 7.3 Paralinguistic Acoustic Feature Vectors:
     - Root-Mean-Square (RMS) Energy (mean, std, max)
     - Zero-Crossing Rate (ZCR) and Pitch Jitter proxy
     - Spectral Centroid & Spectral Rolloff (85% energy threshold)
     - Amplitude Modulation Variance (speech envelope dynamics)
     - Speaking Rate Proxy & Silence Ratio
     - High-Frequency Energy Ratio ($\ge 1500$ Hz)
   - 7.4 Feature-MLP Architecture and Backpropagation

10. **Chapter 8: Multimodal Fusion Strategies**
    - 8.1 Late Decision Fusion (Logit weighting with Temperature Scaling)
    - 8.2 Learned Contextual Gated Multimodal Fusion
    - 8.3 Information Flow and Gradient Propagation in the Gating Network
    - 8.4 Mathematical Ablation Formulation

11. **Chapter 9: Early-Stage Fraud Detection Engine**
    - 9.1 The Need for Real-Time Intervention Before Call Termination
    - 9.2 Progressive Window Slicing ($5\text{s}, 10\text{s}, 15\text{s}, 20\text{s}, 30\text{s}$)
    - 9.3 Sequential Temporal Filtering: $K$-Consecutive High-Risk Window Criterion
    - 9.4 False Alarm Rate (FAR) vs Time-to-Detect Trade-offs

12. **Chapter 10: Experimental Setup & Implementation Details**
    - 10.1 Hardware and Software Environment (Python 3.14, NumPy, FastAPI, React 19)
    - 10.2 Pure-NumPy Implementation Rationale (Eliminating opaque library overhead, ensuring auditability)
    - 10.3 Evaluation Metrics: Precision, Recall, $F_1$-Score, AUROC, Confusion Matrix
    - 10.4 Multi-Seed Protocol (Seeds 42, 123, 2026) for Statistical Rigor

13. **Chapter 11: Experimental Results & Comparative Analysis**
    - 11.1 BiLSTM Baseline Performance Analysis across 3 Seeds (Mean $F_1 = 0.838 \pm 0.086$)
    - 11.2 Telephony Degradation Matrix Results (WER up to 55.8%, $F_1$ degradation analysis)
    - 11.3 Early Detection Results: Reaching 100% Detection Rate by 30 Seconds
    - 11.4 Multimodal Fusion Ablation Analysis
    - 11.5 Confusion Matrix Inspection and Error Case Analysis

14. **Chapter 12: Application Architecture & Demonstration System**
    - 12.1 Backend Architecture: FastAPI REST Endpoints (`POST /analyze`, `GET /health`)
    - 12.2 Real-Time WebSocket Streaming (`WS /ws/stream`)
    - 12.3 Frontend Interface Design: React 19, Vite, TypeScript, and CSS Design System
    - 12.4 Analyst Dashboard Features (Risk gauge, waveform timeline, flagged keywords, channel telemetry)
    - 12.5 Simulated Mobile Phone Alert Modal (Accessible UI, WCAG AA, safe demo actions)

15. **Chapter 13: Discussion, Limitations & Failure Analysis**
    - 13.1 Analysis of False Positives (Legitimate urgent bank calls)
    - 13.2 Limitations of Synthetic Channel Modeling vs Hardware Carrier Traces
    - 13.3 Linguistic Diversity Limitations (Hinglish and Telugu regional nuances)
    - 13.4 Computational Constraints on Edge Telephony Devices

16. **Chapter 14: Ethical, Privacy & Legal Considerations**
    - 14.1 Wiretapping & Call Recording Consent Regulations in India
    - 14.2 On-Device vs Cloud Inference Privacy Implications
    - 14.3 Responsible AI Guidelines for Automated Scam Blocking

17. **Chapter 15: Conclusions & Future Work**
    - 15.1 Summary of Research Contributions
    - 15.2 Research Question Answers
    - 15.3 Directions for Future Research (On-device neural quantization, self-supervised WavLM fine-tuning, carrier-grade SIP proxy integration)

18. **References & Appendices**
    - Complete bibliography of peer-reviewed literature
    - Appendix A: Full Hyperparameter YAML Configurations
    - Appendix B: Mathematical Derivations of Bahdanau Attention & Gated Fusion
    - Appendix C: Complete Pytest Test Suite Results
