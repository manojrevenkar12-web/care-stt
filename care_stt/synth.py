"""Render the corpus to speech with several German Piper (VITS) voices.

Voices are split so that the speakers used to tune the post-correction
threshold (dev) are never the speakers it is evaluated on (test).
Test utterances also get a randomised speaking rate (0.9-1.15x) because
real dictation is faster and less even than read speech.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .config import MODELS_DIR, SR

DEV_VOICES = ["de_DE-thorsten-medium", "de_DE-eva_k-x_low", "de_DE-karlsson-low", "de_DE-kerstin-low"]
TEST_VOICES = ["de_DE-ramona-low", "de_DE-pavoque-low"]

_cache: dict[str, object] = {}


def _tts(voice: str):
    if voice not in _cache:
        import sherpa_onnx

        d = Path(MODELS_DIR) / f"vits-piper-{voice}"
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(d / f"{voice}.onnx"), tokens=str(d / "tokens.txt"),
                    data_dir=str(d / "espeak-ng-data")),
                num_threads=2))
        _cache[voice] = sherpa_onnx.OfflineTts(cfg)
    return _cache[voice]


def seed_of(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def speaking_rate(utt_id: str, voice: str, split: str) -> float:
    if split == "dev":
        return 1.0
    return float(np.random.default_rng(seed_of(utt_id, voice)).uniform(0.9, 1.15))


def synthesize(text: str, voice: str, speed: float = 1.0) -> np.ndarray:
    """Return 16 kHz mono float32 audio, peak-normalised to -3 dBFS, with
    150 ms of leading/trailing silence (like a real push-to-talk recording)."""
    a = _tts(voice).generate(text, sid=0, speed=speed)
    x = np.asarray(a.samples, dtype=np.float32)
    if a.sample_rate != SR:
        x = resample_poly(x, SR, a.sample_rate).astype(np.float32)
    x = x / (np.abs(x).max() + 1e-9) * 0.707
    pad = np.zeros(int(0.15 * SR), dtype=np.float32)
    return np.concatenate([pad, x, pad])


def voices_for(split: str) -> list[str]:
    return DEV_VOICES if split == "dev" else TEST_VOICES
