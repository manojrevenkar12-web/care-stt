"""Speech-to-text backends (Whisper via sherpa-onnx / ONNX Runtime).

sherpa-onnx runs the same exported Whisper graphs on Linux, Android and iOS,
so the latency numbers measured here are a meaningful proxy for the on-device
path, and int8 vs fp32 is a real deployment choice rather than a lab toggle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MODELS_DIR, SR


@dataclass
class Transcript:
    text: str
    seconds: float          # wall-clock decode time
    audio_seconds: float

    @property
    def rtf(self) -> float:
        """Real-time factor: decode time / audio duration (<1 = faster than real time)."""
        return self.seconds / max(self.audio_seconds, 1e-9)


class WhisperASR:
    def __init__(self, size: str = "base", quant: str = "int8", language: str = "de", threads: int = 2):
        import sherpa_onnx

        d = Path(MODELS_DIR) / f"sherpa-onnx-whisper-{size}"
        suffix = ".int8.onnx" if quant == "int8" else ".onnx"
        enc, dec = d / f"{size}-encoder{suffix}", d / f"{size}-decoder{suffix}"
        if not enc.exists():
            raise FileNotFoundError(f"{enc} missing - run scripts/download_models.sh")
        self.name = f"whisper-{size}-{quant}"
        self.size_mb = round((enc.stat().st_size + dec.stat().st_size) / 1e6, 1)
        self._rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(enc), decoder=str(dec), tokens=str(d / f"{size}-tokens.txt"),
            language=language, task="transcribe", num_threads=threads)

    def transcribe(self, audio: np.ndarray, sr: int = SR) -> Transcript:
        if sr != SR:
            raise ValueError(f"expected {SR} Hz audio, got {sr}")
        t0 = time.perf_counter()
        st = self._rec.create_stream()
        st.accept_waveform(SR, audio.astype(np.float32))
        self._rec.decode_stream(st)
        return Transcript(st.result.text.strip(), time.perf_counter() - t0, len(audio) / SR)


class VadSegmenter:
    """Silero VAD: split long recordings (a full shift handover) into speech
    segments under Whisper's 30 s window, dropping silence."""

    def __init__(self, max_segment_s: float = 20.0):
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(Path(MODELS_DIR) / "silero_vad.onnx")
        cfg.silero_vad.min_silence_duration = 0.4
        cfg.silero_vad.max_speech_duration = max_segment_s
        cfg.sample_rate = SR
        self._cfg = cfg

    def segments(self, audio: np.ndarray) -> list[np.ndarray]:
        import sherpa_onnx

        vad = sherpa_onnx.VoiceActivityDetector(self._cfg, buffer_size_in_seconds=600)
        win = self._cfg.silero_vad.window_size
        out = []
        for i in range(0, len(audio), win):
            vad.accept_waveform(audio[i: i + win].astype(np.float32))
            while not vad.empty():
                out.append(np.asarray(vad.front.samples, dtype=np.float32))
                vad.pop()
        vad.flush()
        while not vad.empty():
            out.append(np.asarray(vad.front.samples, dtype=np.float32))
            vad.pop()
        return out


def transcribe_long(asr: WhisperASR, vad: VadSegmenter, audio: np.ndarray) -> Transcript:
    parts = [asr.transcribe(seg) for seg in vad.segments(audio)]
    return Transcript(" ".join(p.text for p in parts if p.text),
                      sum(p.seconds for p in parts), len(audio) / SR)
