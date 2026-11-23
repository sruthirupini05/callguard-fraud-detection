"""
Audio Preprocessing and Analysis Engine for CallGuard.
Implements:
1. True PCM audio decoding (WAV 8/16/24/32-bit PCM & Float32)
2. Sample-rate detection and stereo-to-mono downmixing
3. Resampling to 16 kHz (speech standard) and 8 kHz (telephone bandwidth)
4. Peak and RMS normalization
5. Frame-level Voice Activity Detection (VAD) via Short-Time Energy & Zero-Crossing Rate
6. Background noise floor estimation and Signal-to-Noise Ratio (SNR dB)
7. Audio quality assessment (clipping ratio, dynamic range, RMS)
8. Waveform peak envelope extraction for frontend visualization
"""

import io
import math
import struct
import wave
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy import signal
from scipy.io import wavfile


class AudioProcessingError(Exception):
    """Raised when audio decoding, validation, or processing fails."""
    pass


class AudioProcessor:
    """Robust audio ingestion and quality assessment engine."""

    def __init__(self, target_sr_asr: int = 16000, target_sr_phone: int = 8000):
        self.target_sr_asr = target_sr_asr
        self.target_sr_phone = target_sr_phone

    def decode_and_validate(self, file_bytes: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        """
        Decodes raw audio bytes into float32 mono waveform normalized to [-1.0, 1.0].
        Returns metadata dictionary including samples, duration, sample rate, and channels.
        """
        if len(file_bytes) < 44:
            raise AudioProcessingError("Uploaded file is too small to be a valid audio file (< 44 bytes).")

        # Try standard scipy.io.wavfile first
        try:
            byte_io = io.BytesIO(file_bytes)
            sr, raw_data = wavfile.read(byte_io)
        except Exception:
            # Fallback to standard library wave module
            try:
                byte_io = io.BytesIO(file_bytes)
                with wave.open(byte_io, 'rb') as wf:
                    sr = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    n_frames = wf.getnframes()
                    frames = wf.readframes(n_frames)
                    
                    if sampwidth == 1:
                        raw_data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
                        raw_data /= 128.0
                    elif sampwidth == 2:
                        raw_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 4:
                        raw_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        raise AudioProcessingError(f"Unsupported sample width: {sampwidth} bytes.")
                    
                    if n_channels > 1:
                        raw_data = raw_data.reshape(-1, n_channels)
            except Exception as e:
                raise AudioProcessingError(f"Failed to decode audio file. Must be a valid WAV/PCM format: {str(e)}")

        orig_channels = 1 if raw_data.ndim == 1 else raw_data.shape[1]

        # Convert to float32 normalized to [-1.0, 1.0]
        if raw_data.dtype == np.int16:
            audio = raw_data.astype(np.float32) / 32768.0
        elif raw_data.dtype == np.int32:
            audio = raw_data.astype(np.float32) / 2147483648.0
        elif raw_data.dtype == np.uint8:
            audio = (raw_data.astype(np.float32) - 128.0) / 128.0
        elif raw_data.dtype in (np.float32, np.float64):
            audio = raw_data.astype(np.float32)
        else:
            audio = raw_data.astype(np.float32)

        # Stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Peak normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0.001:
            audio = audio / max_val * 0.95

        duration = float(len(audio)) / float(sr)
        if duration < 0.2:
            raise AudioProcessingError(f"Audio duration is too short ({duration:.2f}s). Minimum is 0.5s.")

        # Resample to 16 kHz for ASR and 8 kHz for telephone models
        audio_16k = self.resample(audio, sr, self.target_sr_asr)
        audio_8k = self.resample(audio, sr, self.target_sr_phone)

        # Voice Activity Detection & Quality Analysis
        vad_segments, speech_ratio = self.compute_vad(audio_16k, self.target_sr_asr)
        quality_metrics = self.assess_quality(audio_16k, self.target_sr_asr)
        waveform_envelope = self.extract_envelope(audio_16k, num_points=100)

        return {
            "filename": filename,
            "original_sample_rate": sr,
            "original_channels": orig_channels,
            "duration_seconds": round(duration, 3),
            "samples_count": len(audio),
            "audio_16k": audio_16k,
            "audio_8k": audio_8k,
            "vad_segments": vad_segments,
            "speech_ratio": round(speech_ratio, 3),
            "snr_db": quality_metrics["snr_db"],
            "noise_floor_db": quality_metrics["noise_floor_db"],
            "clipping_ratio": quality_metrics["clipping_ratio"],
            "dynamic_range_db": quality_metrics["dynamic_range_db"],
            "waveform_envelope": waveform_envelope
        }

    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """High-quality polyphase resampling to target sampling rate."""
        if orig_sr == target_sr:
            return audio.astype(np.float32)
        
        # Calculate rational gcd approximation
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        
        # Use resample_poly for speed and antialiasing
        try:
            resampled = signal.resample_poly(audio, up, down)
        except Exception:
            # Fallback to Fourier method
            num_samples = int(round(len(audio) * float(target_sr) / orig_sr))
            resampled = signal.resample(audio, num_samples)
            
        return resampled.astype(np.float32)

    def compute_vad(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        frame_ms: int = 30
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Energy and Zero-Crossing Rate (ZCR) based Voice Activity Detection.
        Identifies conversational speech vs silence/background noise segments.
        """
        frame_len = int(sr * (frame_ms / 1000.0))
        if frame_len == 0 or len(audio) < frame_len:
            return [{"start": 0.0, "end": round(len(audio) / sr, 2), "is_speech": True}], 1.0

        num_frames = len(audio) // frame_len
        energies = []
        zcrs = []

        for i in range(num_frames):
            frame = audio[i * frame_len : (i + 1) * frame_len]
            # Short-Time Energy
            ste = float(np.mean(frame ** 2))
            energies.append(ste)
            # Zero-Crossing Rate
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)
            zcrs.append(zcr)

        energies = np.array(energies)
        zcrs = np.array(zcrs)

        # Dynamic energy threshold (25th percentile + adaptive margin)
        silence_floor = np.percentile(energies, 15)
        speech_threshold = max(silence_floor * 3.5, 0.0005)

        is_speech_frame = (energies > speech_threshold) | ((energies > silence_floor * 1.8) & (zcrs > 0.08))

        # Smooth VAD with morphological closing (min 3 frames)
        smoothed = is_speech_frame.copy()
        for i in range(1, len(smoothed) - 1):
            if not smoothed[i] and smoothed[i - 1] and smoothed[i + 1]:
                smoothed[i] = True

        # Build contiguous intervals
        segments = []
        if len(smoothed) == 0:
            return [], 0.0

        current_state = smoothed[0]
        start_frame = 0

        for i in range(1, len(smoothed)):
            if smoothed[i] != current_state:
                segments.append({
                    "start": round(start_frame * (frame_ms / 1000.0), 3),
                    "end": round(i * (frame_ms / 1000.0), 3),
                    "is_speech": bool(current_state)
                })
                current_state = smoothed[i]
                start_frame = i

        segments.append({
            "start": round(start_frame * (frame_ms / 1000.0), 3),
            "end": round(len(smoothed) * (frame_ms / 1000.0), 3),
            "is_speech": bool(current_state)
        })

        speech_frames = int(np.sum(smoothed))
        speech_ratio = float(speech_frames) / float(len(smoothed)) if len(smoothed) > 0 else 0.0
        return segments, speech_ratio

    def assess_quality(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """
        Assesses SNR, noise floor, dynamic range, and digital clipping.
        """
        if len(audio) == 0:
            return {"snr_db": 0.0, "noise_floor_db": -90.0, "clipping_ratio": 0.0, "dynamic_range_db": 0.0}

        abs_audio = np.abs(audio)
        clipping_samples = np.sum(abs_audio >= 0.985)
        clipping_ratio = float(clipping_samples) / float(len(audio))

        # Dynamic range
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-12
        rms_db = 20.0 * np.log10(rms)

        # Estimate noise floor from quietest 10% frames (30ms frames)
        frame_len = int(sr * 0.030)
        num_frames = len(audio) // max(1, frame_len)
        if num_frames > 5:
            frame_energies = [np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2) for i in range(num_frames)]
            frame_energies.sort()
            noise_power = np.mean(frame_energies[: max(1, int(num_frames * 0.15))]) + 1e-12
            signal_power = np.mean(frame_energies[int(num_frames * 0.50) :]) + 1e-12
            snr_db = 10.0 * np.log10(max(signal_power, 1e-12) / max(noise_power, 1e-12))
            noise_floor_db = 10.0 * np.log10(noise_power)
        else:
            snr_db = 15.0
            noise_floor_db = -45.0

        peak = np.max(abs_audio) + 1e-12
        dynamic_range_db = 20.0 * np.log10(peak / (np.min(abs_audio[abs_audio > 0]) if np.any(abs_audio > 0) else 1e-6))

        return {
            "snr_db": round(float(np.clip(snr_db, 0.0, 50.0)), 2),
            "noise_floor_db": round(float(noise_floor_db), 2),
            "clipping_ratio": round(float(clipping_ratio), 4),
            "dynamic_range_db": round(float(np.clip(dynamic_range_db, 10.0, 96.0)), 2)
        }

    def extract_envelope(self, audio: np.ndarray, num_points: int = 100) -> List[float]:
        """
        Extracts peak-amplitude waveform envelope array for frontend canvas rendering.
        """
        if len(audio) == 0:
            return [0.0] * num_points

        chunk_size = len(audio) // num_points
        if chunk_size == 0:
            return [float(abs(x)) for x in audio] + [0.0] * (num_points - len(audio))

        envelope = []
        for i in range(num_points):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            peak = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0
            envelope.append(round(peak, 4))

        return envelope
