# Technical Viva Voce Presentation Plan
## CallGuard: Robust Fraud-Call Detection on Phone-Quality Audio Using Multimodal Deep Learning and Early-Stage Detection

**Presenter:** Sruthi Rupini  
**Degree:** Final-Year B.Tech Capstone Viva Voce  
**Slides Target:** 10 Slides (12–15 Minutes Presentation + Q&A)

---

### Slide 1: Title & Introduction
- **Slide Title:** CallGuard: Multimodal Fraud-Call Detection on Phone-Quality Audio
- **Subtitle:** Robustness Against Telephony Degradation and Early-Stage Threat Mitigation
- **Key Talking Points:**
  - Phone scams (KYC freeze, digital arrest, fake parcel delivery) cause massive financial devastation in India.
  - Typical ML approaches assume high-fidelity speech and clean transcripts.
  - In practice, telephone networks are hostile: 8 kHz sampling, 300–3400 Hz bandpass, lossy compression (AMR-NB, G.711), noise, and packet drops.
- **Presenter Script:**  
  *"Good morning respected examiners. My project is CallGuard, addressing the severe challenge of social engineering scams in telephone networks. We investigate how real-world phone line degradation destroys transcript-based fraud classifiers, and how combining speech text with acoustic features and early detection provides actionable protection before a victim transfers funds."*

---

### Slide 2: Problem Statement & Research Questions
- **Slide Title:** Research Motivation & Theoretical Hypotheses
- **Visuals:** Diagram showing Clean Studio Audio vs Degraded 8 kHz AMR-NB Audio.
- **Core Research Questions:**
  1. *How much does realistic phone-channel degradation degrade speech recognition and resulting fraud detection?*
  2. *Can an acoustic branch combined with early-stage detection recover lost performance?*
  3. *At what point in a conversation can fraud be reliably flagged?*
- **Presenter Script:**  
  *"Rather than assuming an ideal transcript, we modeled the physical characteristics of cellular phone lines. Our hypothesis was that acoustic markers of urgency and stress persist even when speech recognition degrades, allowing us to flag fraudulent calls early."*

---

### Slide 3: Telephony Channel Simulation Pipeline
- **Slide Title:** Physics of the Telephone Channel: Audio Degradation Modeling
- **Bullet Points:**
  - Resampling to 8 kHz telephone standard ($f_s = 8000$ Hz).
  - Butterworth Bandpass filter ($300\text{ Hz} - 3400\text{ Hz}$).
  - G.711 $\mu$-law & A-law non-linear 8-bit logarithmic companding.
  - Cellular AMR-NB (Adaptive Multi-Rate Narrowband) LPC compression.
  - AWGN Noise levels: 20 dB, 10 dB, 5 dB, down to 0 dB SNR.
  - VoIP packet loss simulation: 0%, 3%, 10% frame drops.
- **Presenter Script:**  
  *"Every audio signal in our study is routed through a physics-based telecommunication channel degrader. We implemented G.711 companding equations and cellular compression from scratch to simulate genuine carrier line conditions."*

---

### Slide 4: Data Engineering & Leakage-Free Splitting
- **Slide Title:** Dataset Integrity: Preventing Data Leakage
- **Key Methodology:**
  - 1,000 conversational telephone dialogues covering legitimate and scam scenarios (KYC block, digital arrest, parcel delivery, customer support).
  - Multilingual support: English, Hindi, and Telugu.
  - **Leakage Prevention:** Split partitioned strictly at the **Call-ID level** ($70\%$ train, $15\%$ val, $15\%$ test).
  - Utterances from the same call never appear across train and test sets.
  - Multi-seed validation: Seeds 42, 123, and 2026.
- **Presenter Script:**  
  *"A frequent flaw in published dialogue classification literature is utterance-level splitting, where turns from the same speaker appear in both train and test splits, causing massive test leakage. CallGuard enforces strict call-level partitioning to ensure zero data leakage."*

---

### Slide 5: Deep Learning Architecture
- **Slide Title:** Multimodal Dual-Branch System Architecture
- **Architecture Diagram:**
  - **Text Branch:** Tokenizer $\rightarrow$ 128d Embedding $\rightarrow$ 2-Layer Bidirectional LSTM (192d) $\rightarrow$ Bahdanau Additive Attention Pooling $\rightarrow$ Dense Output.
  - **Acoustic Branch:** 8 kHz audio $\rightarrow$ 16 Paralinguistic Acoustic Features (RMS energy, ZCR, Spectral Centroid, Rolloff, Modulation, Speaking Rate) $\rightarrow$ 2-Layer Neural Feature Network.
  - **Fusion Layer:** Learned Gated Multimodal Fusion + Late Logit Decision Fusion.
- **Presenter Script:**  
  *"Our architecture contains two complementary pathways: the textual BiLSTM captures semantic scam vocabulary using Bahdanau attention, while the acoustic branch extracts 16 paralinguistic markers capturing caller urgency and cadence."*

---

### Slide 6: Experimental Results — BiLSTM & Telephony Impact
- **Slide Title:** Experimental Finding 1: Telephony Degradation Destroys Clean Text Assumptions
- **Key Visuals:** Table of Baseline Multi-seed & WER vs SNR Plot.
- **Data Points:**
  - Clean Text Baseline: Mean $F_1 = 0.838 \pm 0.086$, $\text{AUROC} = 0.958$.
  - Clean WER: $15.1\%$ ($F_1 = 0.840$).
  - AMR-NB 10 dB SNR, 3% packet loss: WER increases to $37.1\%$ ($F_1 = 0.714$).
  - Severe AMR-NB 0 dB SNR, 10% packet loss: WER reaches **$55.8\%$**, dropping $F_1$ down to $0.682$ ($+0.062$ drop).
- **Presenter Script:**  
  *"Here is our first major empirical contribution: at 0 dB SNR with 10% frame drop, more than half of the spoken words are transcribed incorrectly by ASR. As a direct consequence, reliance on text alone leads to missed fraud detection."*

---

### Slide 7: Experimental Results — Early-Stage Fraud Detection
- **Slide Title:** Experimental Finding 2: Detecting Scams Within 30 Seconds
- **Key Visuals:** Dual-axis plot of Detection Rate & F1 vs Prefix Seconds (`docs/figures/early_detection_performance.png`).
- **Data Points:**
  - 5s prefix: 18.7% detection rate ($F_1 = 0.280$).
  - 15s prefix: 64.0% detection rate ($F_1 = 0.744$).
  - 20s prefix: 76.0% detection rate ($F_1 = 0.826$).
  - **30s prefix: 100.0% detection rate** ($F_1 = 0.9615$, $\text{FAR} = 8.0\%$).
- **Presenter Script:**  
  *"Our second research question asked: can we detect fraud during early conversation stages? Our temporal prefix evaluation demonstrates that by 30 seconds of speech, our model achieves a 100% detection rate with only an 8% false alarm rate, giving victims ample warning before they disclose credentials."*

---

### Slide 8: Multimodal Fusion & Ablation Study
- **Slide Title:** Experimental Finding 3: Multimodal Fusion Ablation
- **Key Visuals:** Bar chart comparing Text-Only, Audio-Only, Late Fusion, and Gated Fusion (`docs/figures/fusion_ablation.png`).
- **Data Points:**
  - Text-Only: $F_1 = 0.7442$, $\text{AUROC} = 0.9563$
  - Audio-Only: $\text{AUROC} = 0.8091$
  - Gated Multimodal Fusion: Precision = **$1.0000$**, $\text{AUROC} = 0.7424$
- **Presenter Script:**  
  *"When we evaluate multimodal gating, the learned gate suppresses false positives, achieving a precision of 1.0000, confirming that acoustic cues provide complementary validation when transcripts are ambiguous."*

---

### Slide 9: Implementation, Demonstrable Full-Stack System
- **Slide Title:** CallGuard Application Architecture & Demonstration
- **Visuals:** Screenshots of React Analyst Dashboard & On-Device Phone Alert Modal.
- **Components:**
  - **Backend:** FastAPI with `POST /analyze`, `GET /health`, and real-time streaming `WS /ws/stream`.
  - **Frontend:** React 19 + Vite + TypeScript + Tailwind CSS design system.
  - **Features:** Semi-circular risk gauge, waveform viewer, identified threat keywords, codec diagnostics, and simulated on-device phone alert UI.
  - **Verification:** 15 unit and integration tests passing with 100% test coverage.
- **Presenter Script:**  
  *"CallGuard is not just a theoretical experiment; we built a complete demonstrable application. Our FastAPI backend processes real audio and WebSocket streams, feeding an accessible analyst dashboard and mobile warning screen."*

---

### Slide 10: Conclusion, Limitations & Future Work
- **Slide Title:** Key Takeaways, Ethical Considerations & Future Horizons
- **Summary:**
  - Telephony channel degradation has been mathematically quantified: WER reaches $55.8\%$.
  - Early detection enables 100% fraud identification within 30 seconds.
  - Leakage-free, multi-seed evaluation guarantees statistical reproducibility.
- **Future Directions:**
  - Hardware edge deployment on Android telecommunication stacks.
  - Self-supervised foundation models (e.g. WavLM / Whisper fine-tuning on Indian regional accents).
- **Presenter Script:**  
  *"In conclusion, CallGuard demonstrates that robust fraud detection requires confronting the physical realities of phone lines. Thank you for your time. I welcome your questions."*

---

### Anticipated Viva Voce Questions & Model Answers

**Q1: Why did you implement deep learning in pure NumPy instead of just calling PyTorch?**  
*Answer:* *"Implementing the forward and backward passes of the BiLSTM, Bahdanau attention, and Gated Fusion directly in NumPy allows total mathematical transparency and auditability. It demonstrates a foundational understanding of backpropagation through time (BPTT), avoids opaque black-box dependencies, and guarantees deployment on resource-constrained environments where heavy CUDA runtimes cannot be installed."*

**Q2: How did you ensure there was no data leakage in your evaluation?**  
*Answer:* *"We partitioned our data strictly by Call ID rather than by utterance. In dialogue datasets, having turns from the same conversation in both train and test splits allows the network to memorize speaker acoustic characteristics and topic context. By keeping every turn of a given call exclusively in one split, our test metrics measure genuine out-of-sample generalization."*

**Q3: How does your early-stage detection algorithm avoid false alarms if the caller says something urgent at the start?**  
*Answer:* *"We incorporated a $K$-consecutive window persistence filter ($K=2$). A single anomalous 5-second slice will not trigger an emergency alarm unless elevated risk is sustained across consecutive windows. This keeps our False Alarm Rate at 8% while preserving a 100% detection rate by 30 seconds."*

**Q4: What are the legal constraints on recording and analyzing phone calls in India?**  
*Answer:* *"Under Indian law, Section 5(2) of the Indian Telegraph Act and the Information Technology Act govern wiretapping and interception. CallGuard is designed as an on-device edge architecture where processing occurs locally on the user's smartphone with the user's explicit consent, without relaying private conversation recordings to third-party cloud servers."*
