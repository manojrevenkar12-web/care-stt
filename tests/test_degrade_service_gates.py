import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from care_stt import degrade
from care_stt.asr import Transcript
from care_stt.config import SR
from care_stt.gates import check
from care_stt.service import create_app


def tone(seconds=1.0, f=220.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)


# ---------- degradation ----------

def test_add_noise_hits_target_snr():
    rng = np.random.default_rng(0)
    x, n = tone(), rng.standard_normal(SR).astype(np.float32)
    for snr in (0, 5, 10, 20):
        y = degrade.add_noise(x, n, snr)
        # undo peak normalisation by projecting onto the clean signal
        scale = float(np.dot(y, x) / np.dot(x, x))
        resid = y - scale * x
        measured = 10 * np.log10(np.sum((scale * x) ** 2) / np.sum(resid ** 2))
        assert abs(measured - snr) < 0.5


@pytest.mark.parametrize("cond", degrade.CONDITIONS)
def test_conditions_keep_length_and_are_deterministic(cond):
    x = tone(1.5)
    babble = [tone(0.7, f) for f in (300, 410, 530, 650)]
    a = degrade.apply(cond, x, babble, seed=1)
    b = degrade.apply(cond, x, babble, seed=1)
    assert len(a) == len(x) and a.dtype == np.float32
    assert np.allclose(a, b)
    assert np.abs(a).max() <= 1.0


def test_telephone_removes_low_frequencies():
    y = degrade.telephone(tone(1.0, 100.0) + tone(1.0, 1000.0))
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    assert spec[np.argmin(abs(freqs - 100))] < 0.1 * spec[np.argmin(abs(freqs - 1000))]


# ---------- service (fake ASR, no model download needed) ----------

class FakeASR:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def transcribe(self, audio):
        return Transcript(self.text, 0.01, len(audio) / SR)


def wav_bytes(x, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return buf.getvalue()


def test_service_corrects_extracts_and_flags():
    app = create_app(asr=FakeASR("Blutdruck 130-80, Puls 22. Ramibril 5 Milligramm gegeben."), threshold=0.75)
    r = TestClient(app).post("/transcribe", files={"audio": ("a.wav", wav_bytes(tone(2.0)), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert "Ramipril" in body["transcript"]
    assert body["corrections"][0]["replaced_with"] == "Ramipril"
    assert body["vitals"]["bp_sys"] == 130
    assert body["needs_review"] and {"field": "pulse", "reason": "implausible"} in body["review_flags"]


def test_service_resamples_other_rates():
    app = create_app(asr=FakeASR("Puls 72"))
    r = TestClient(app).post("/transcribe", files={"audio": ("a.wav", wav_bytes(tone(1.0)[:: 2], 8000), "audio/wav")})
    assert r.status_code == 200 and abs(r.json()["timing"]["audio_s"] - 1.0) < 0.05


def test_service_rejects_garbage():
    app = create_app(asr=FakeASR(""))
    r = TestClient(app).post("/transcribe", files={"audio": ("a.wav", b"not audio", "audio/wav")})
    assert r.status_code == 400


# ---------- gates ----------

def test_gates_pass_and_fail():
    metrics = {"test": {"m": {"clean": {"raw": {"wer": 0.2, "number_utt_acc": 0.9}}}}}
    res = check(metrics, [
        {"name": "wer", "model": "m", "condition": "clean", "metric": "raw.wer", "max": 0.25},
        {"name": "nums", "model": "m", "condition": "clean", "metric": "raw.number_utt_acc", "min": 0.95},
        {"name": "missing", "model": "x", "condition": "clean", "metric": "raw.wer", "max": 1},
    ])
    assert [r["passed"] for r in res] == [True, False, False]
