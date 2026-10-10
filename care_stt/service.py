"""HTTP service: audio in, chart-ready care note out.

  uvicorn care_stt.service:app --port 8000
  curl -F audio=@note.wav localhost:8000/transcribe

Response: raw transcript, lexicon-corrected transcript (with every change
listed, so nothing is rewritten invisibly), extracted vitals/doses, review
flags, and timing. Recordings longer than 25 s are split with Silero VAD.
"""
from __future__ import annotations

import io
import os
import time

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from scipy.signal import resample_poly

from .config import SR
from .correct import LexiconCorrector
from .extract import extract
from .lexicon import Lexicon

LONG_FORM_S = 25.0
DEFAULT_THRESHOLD = float(os.environ.get("CARE_STT_THRESHOLD", "0.75"))


def load_audio(data: bytes) -> np.ndarray:
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"unreadable audio: {e}") from e
    x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def create_app(asr=None, vad=None, lexicon: Lexicon | None = None, threshold: float = DEFAULT_THRESHOLD) -> FastAPI:
    app = FastAPI(title="care-stt", version="0.1.0")
    lex = lexicon or Lexicon.load()
    corrector = LexiconCorrector(lex, threshold)
    state = {"asr": asr, "vad": vad}

    def get_asr():
        if state["asr"] is None:
            from .asr import WhisperASR
            size, quant = os.environ.get("CARE_STT_MODEL", "base-int8").split("-")
            state["asr"] = WhisperASR(size, quant)
        return state["asr"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model": getattr(state["asr"], "name", "not loaded")}

    @app.post("/transcribe")
    async def transcribe(audio: UploadFile = File(...)):
        t0 = time.perf_counter()
        x = load_audio(await audio.read())
        if len(x) < 0.2 * SR:
            raise HTTPException(status_code=400, detail="audio shorter than 0.2 s")
        model = get_asr()
        if len(x) / SR > LONG_FORM_S:
            from .asr import VadSegmenter, transcribe_long
            state["vad"] = state["vad"] or VadSegmenter()
            tr = transcribe_long(model, state["vad"], x)
        else:
            tr = model.transcribe(x)
        fixed, corrs = corrector.correct(tr.text)
        ex = extract(fixed, lex)
        return {
            "transcript_raw": tr.text,
            "transcript": fixed,
            "corrections": [{"heard": c.original, "replaced_with": c.replacement, "score": c.score} for c in corrs],
            "vitals": ex.vitals,
            "doses": [{"drug": d[0], "amount": d[1], "unit": d[2]} for d in ex.doses],
            "review_flags": [{"field": f[0], "reason": f[1]} for f in ex.flags],
            "needs_review": ex.needs_review,
            "timing": {"audio_s": round(tr.audio_seconds, 2), "decode_s": round(tr.seconds, 3),
                       "total_s": round(time.perf_counter() - t0, 3), "rtf": round(tr.rtf, 3)},
            "model": getattr(model, "name", "unknown"),
        }

    return app


app = create_app()
