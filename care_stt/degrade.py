"""Real-world acoustic conditions for a care setting.

Every condition is deterministic given a seed so runs are reproducible and
two model versions are always compared on identical audio.

  clean      - push-to-talk recording in a quiet room
  babble_10  - common room / corridor: overlapping speech at 10 dB SNR
  babble_5   - busy shift handover: overlapping speech at 5 dB SNR
  reverb     - tiled bathroom or corridor, RT60 ~0.6 s
  telephone  - 300-3400 Hz band, 8 kHz, GSM full-rate codec (phone dictation)
  opus_8k    - Opus at 8 kbit/s (weak mobile connection / VoIP)
  combined   - reverb + 10 dB babble + Opus 8 kbit/s (realistic worst case)
"""
from __future__ import annotations

import subprocess

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from .config import SR

CONDITIONS = ["clean", "babble_10", "babble_5", "reverb", "telephone", "opus_8k", "combined"]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def _peak_norm(x: np.ndarray, peak: float = 0.707) -> np.ndarray:
    return (x / (np.abs(x).max() + 1e-9) * peak).astype(np.float32)


def make_babble(sources: list[np.ndarray], length: int, rng: np.random.Generator) -> np.ndarray:
    """Overlay several talkers, each starting at a random offset and looped."""
    out = np.zeros(length, dtype=np.float64)
    for s in sources:
        s = s / _rms(s)
        reps = int(np.ceil((length + len(s)) / len(s))) + 1
        tiled = np.tile(s, reps)
        off = int(rng.integers(0, len(s)))
        out += tiled[off: off + length]
    return out.astype(np.float32)


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = noise[: len(x)]
    gain = _rms(x) / (_rms(noise) * 10 ** (snr_db / 20))
    return _peak_norm(x + gain * noise)


def synthetic_rir(rt60: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Exponentially decaying noise tail with a direct path - a standard
    stand-in for a measured room impulse response."""
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail[: int(0.003 * sr)] = 0.0          # gap before first reflections
    rir = 0.35 * tail / np.abs(tail).max()
    rir[0] = 1.0
    return rir


def reverberate(x: np.ndarray, rng: np.random.Generator, rt60: float = 0.6) -> np.ndarray:
    y = fftconvolve(x, synthetic_rir(rt60, rng))[: len(x)]
    return _peak_norm(y)


def _ffmpeg(x: np.ndarray, encode: list[str], decode_sr: int = SR) -> np.ndarray:
    """Encode then decode through ffmpeg, returning float32 audio at decode_sr."""
    enc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", *encode, "-"],
        input=x.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    dec = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "-", "-f", "f32le", "-ar", str(decode_sr), "-ac", "1", "-"],
        input=enc, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, dtype=np.float32).copy()
    if len(y) >= len(x):
        return y[: len(x)]
    return np.pad(y, (0, len(x) - len(y)))


def telephone(x: np.ndarray) -> np.ndarray:
    sos = butter(6, [300, 3400], btype="bandpass", fs=SR, output="sos")
    y = sosfilt(sos, x).astype(np.float32)
    y = _ffmpeg(y, ["-ar", "8000", "-c:a", "libgsm", "-f", "gsm"])
    return _peak_norm(y)


def opus(x: np.ndarray, kbps: int = 8) -> np.ndarray:
    return _peak_norm(_ffmpeg(x, ["-c:a", "libopus", "-b:a", f"{kbps}k", "-application", "voip", "-f", "ogg"]))


def apply(condition: str, x: np.ndarray, babble_sources: list[np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if condition == "clean":
        return x
    if condition in ("babble_10", "babble_5"):
        snr = 10 if condition == "babble_10" else 5
        return add_noise(x, make_babble(babble_sources, len(x), rng), snr)
    if condition == "reverb":
        return reverberate(x, rng)
    if condition == "telephone":
        return telephone(x)
    if condition == "opus_8k":
        return opus(x, 8)
    if condition == "combined":
        y = reverberate(x, rng)
        y = add_noise(y, make_babble(babble_sources, len(y), rng), 10)
        return opus(y, 8)
    raise ValueError(f"unknown condition {condition!r}")
