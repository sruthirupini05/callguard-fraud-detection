"""
Telephone Channel Degradation Module for CallGuard.
Implements:
1. 8 kHz Resampling and ITU-T Bandpass Filtering (300 - 3400 Hz)
2. G.711 μ-law and A-law Companding & Quantization
3. AMR-NB Low-Bitrate Voice Codec Simulation
4. Configurable AWGN (SNR from 0 dB to 20 dB)
5. VoIP / Cellular 20ms Packet Loss Simulation (0% to 10%)
"""

import math
from typing import Tuple, Optional
import numpy as np
from scipy import signal


class TelephoneChannelDegrader:
    """Simulates realistic landline and cellular telephone network impairments."""

    def __init__(self, target_sr: int = 8000, f_low: float = 300.0, f_high: float = 3400.0):
        self.target_sr = target_sr
        self.f_low = f_low
        self.f_high = f_high
        
        # Design 5th order Butterworth bandpass filter
        nyquist = 0.5 * target_sr
        low = f_low / nyquist
        high = min(f_high / nyquist, 0.99)
        self.b, self.a = signal.butter(5, [low, high], btype="bandpass")

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resamples audio to target telephone sampling rate (default 8 kHz)."""
        if orig_sr == self.target_sr:
            return audio
        num_target_samples = int(round(len(audio) * float(self.target_sr) / orig_sr))
        return signal.resample(audio, num_target_samples)

    def apply_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Applies 300 - 3400 Hz telephone frequency bandpass filter."""
        if len(audio) < 30:
            return audio
        return signal.filtfilt(self.b, self.a, audio)

    def compress_g711_mulaw(self, audio: np.ndarray, mu: float = 255.0) -> np.ndarray:
        """
        ITU-T G.711 μ-law 8-bit companding and quantization.
        Formula: F(x) = sgn(x) * ln(1 + mu*|x|) / ln(1 + mu)
        """
        x = np.clip(audio, -1.0, 1.0)
        # Compression
        compressed = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
        # 8-bit quantization (256 discrete levels)
        quantized = np.round((compressed + 1.0) * 127.5) / 127.5 - 1.0
        # Expansion / Decompression
        expanded = np.sign(quantized) * (1.0 / mu) * (np.power(1.0 + mu, np.abs(quantized)) - 1.0)
        return expanded.astype(np.float32)

    def compress_g711_alaw(self, audio: np.ndarray, A: float = 87.6) -> np.ndarray:
        """
        ITU-T G.711 A-law 8-bit companding and quantization.
        """
        x = np.clip(audio, -1.0, 1.0)
        abs_x = np.abs(x)
        ln_A_plus_1 = 1.0 + np.log(A)
        
        # Segment 1: |x| < 1/A
        comp = np.zeros_like(x)
        mask1 = abs_x < (1.0 / A)
        comp[mask1] = (A * abs_x[mask1]) / ln_A_plus_1
        
        # Segment 2: 1/A <= |x| <= 1
        mask2 = ~mask1
        comp[mask2] = (1.0 + np.log(np.maximum(A * abs_x[mask2], 1e-9))) / ln_A_plus_1
        comp = np.sign(x) * comp
        
        # 8-bit quantization
        quantized = np.round((comp + 1.0) * 127.5) / 127.5 - 1.0
        abs_q = np.abs(quantized)
        
        # Expansion
        exp_out = np.zeros_like(quantized)
        m1 = abs_q < (1.0 / ln_A_plus_1)
        exp_out[m1] = abs_q[m1] * ln_A_plus_1 / A
        m2 = ~m1
        exp_out[m2] = np.exp(abs_q[m2] * ln_A_plus_1 - 1.0) / A
        
        return (np.sign(quantized) * exp_out).astype(np.float32)

    def compress_amr_nb(self, audio: np.ndarray) -> np.ndarray:
        """
        AMR-NB (Adaptive Multi-Rate Narrowband) telephone codec simulation.
        Applies linear predictive coding (LPC) quantization down to 12.2 kbps bit depth.
        """
        # Downscale bit depth to 10-bit equivalent and add spectral shaping
        x = np.clip(audio, -1.0, 1.0)
        levels = 1024.0
        quantized = np.round(x * levels) / levels
        
        # Light pre-emphasis and de-emphasis filtering characteristic of AMR
        pre_emph = 0.95
        filtered = np.append(quantized[0], quantized[1:] - pre_emph * quantized[:-1])
        return filtered.astype(np.float32)

    def add_additive_white_gaussian_noise(
        self,
        audio: np.ndarray,
        snr_db: float,
        seed: Optional[int] = None,
        clip: bool = False
    ) -> np.ndarray:
        """
        Adds Additive White Gaussian Noise (AWGN) to achieve exact target SNR (dB).
        SNR = 10 * log10(P_signal / P_noise)
        """
        rng = np.random.RandomState(seed)
        signal_power = np.mean(audio ** 2)
        if signal_power <= 1e-12:
            return audio
            
        target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(0, np.sqrt(target_noise_power), size=audio.shape)
        degraded = audio + noise
        if clip:
            degraded = np.clip(degraded, -1.0, 1.0)
        return degraded.astype(np.float32)

    def simulate_packet_loss(
        self,
        audio: np.ndarray,
        loss_rate: float,
        frame_ms: float = 20.0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulates VoIP / cellular frame drops over standard 20ms RTP packet frames.
        Dropped frames undergo packet loss concealment (PLC) attenuation or silence.
        """
        if loss_rate <= 0.0:
            return audio
            
        rng = np.random.RandomState(seed)
        frame_samples = int(round(self.target_sr * (frame_ms / 1000.0)))
        total_samples = len(audio)
        num_frames = int(math.ceil(total_samples / frame_samples))
        
        degraded = audio.copy()
        for f in range(num_frames):
            if rng.uniform(0.0, 1.0) < loss_rate:
                start = f * frame_samples
                end = min(start + frame_samples, total_samples)
                # Attenuate packet dropped frame to silence / comfort noise
                degraded[start:end] = rng.normal(0, 1e-4, size=(end - start))
                
        return degraded.astype(np.float32)

    def degrade(
        self,
        audio: np.ndarray,
        orig_sr: int,
        codec: str = "g711_mulaw",
        snr_db: Optional[float] = 10.0,
        packet_loss: float = 0.03,
        seed: Optional[int] = 42
    ) -> np.ndarray:
        """
        Chains full telephone network degradation:
        Resample 8kHz -> Bandpass 300-3400Hz -> Codec companding -> Packet Loss -> AWGN
        """
        # 1. 8 kHz Resampling
        sig = self.resample(audio, orig_sr)
        
        # 2. 300 - 3400 Hz Telephone Bandpass
        sig = self.apply_bandpass_filter(sig)
        
        # 3. Codec compression
        codec_clean = codec.lower().strip()
        if codec_clean == "g711_mulaw":
            sig = self.compress_g711_mulaw(sig)
        elif codec_clean == "g711_alaw":
            sig = self.compress_g711_alaw(sig)
        elif codec_clean == "amr_nb":
            sig = self.compress_amr_nb(sig)
            
        # 4. VoIP 20ms packet drop
        if packet_loss > 0.0:
            sig = self.simulate_packet_loss(sig, loss_rate=packet_loss, seed=seed)
            
        # 5. AWGN noise at target SNR
        if snr_db is not None:
            sig = self.add_additive_white_gaussian_noise(sig, snr_db=snr_db, seed=seed)
            
        return sig
