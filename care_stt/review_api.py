"""v2 HTTP surface: the transcription endpoint the app would call, and the data the review console
reads.

  POST /v2/transcribe     audio + facility (or an explicit phrase list) -> transcript, phrases the
                          facility list completed, extracted fields, and the gate's decision:
                          "file" (goes to the chart) or "confirm" (read back to the nurse first)
  GET  /v2/systems        the systems benchmarked and their headline metrics
  GET  /v2/utterances     benchmark utterances for one system (filters: set, condition, errors)
  GET  /v2/audio/{id}     the rendered test audio as WAV, for listening in the console

  uvicorn care_stt.review_api:app --port 8000   (serves console/dist at / when it is built)
"""
from __future__ import annotations

import io
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from . import corpus2
from .config import MODELS_DIR, ROOT, SR
from .eval2 import OUT, audio, fold, items, scoring_lexicon
from .extract import extract
from .gate import FEATURES, Gate, chart_error, features
from .metrics import score_utt
from .service import load_audio


@lru_cache(maxsize=1)
def _meta() -> dict[str, dict]:
    return {it["id"]: it for ts in ("v1", "facility", "v1-dev", "facility-dev") for it in items(ts)}


@lru_cache(maxsize=1)
def _rows() -> list[dict]:
    p = OUT / "utterances.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


@lru_cache(maxsize=1)
def _gate() -> Gate | None:
    p = OUT / "gate.pt"
    if not p.exists():
        return None
    d = torch.load(p)
    g = Gate(len(d["features"]))
    g.load_state_dict(d["state"])
    g.threshold = d["threshold"]
    return g.eval()


def _runtime():
    from .runtime import Runtime

    model = os.environ.get("CARE_STT_V2_MODEL", str(Path(MODELS_DIR) / "finetuned" / "base-care-export"))
    return Runtime(model, "base", "int8", encoder_dir=Path(MODELS_DIR) / "sherpa-onnx-whisper-base")


def create_app(runtime=None) -> FastAPI:
    app = FastAPI(title="care-stt v2", version="0.2.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
    state = {"rt": runtime}

    @app.post("/v2/transcribe")
    async def transcribe(audio_file: UploadFile = File(..., alias="audio"), facility: str | None = Form(None),
                         phrases: str | None = Form(None)):
        x = load_audio(await audio_file.read())
        if len(x) < 0.2 * SR:
            raise HTTPException(400, "audio shorter than 0.2 s")
        if facility and facility not in {**corpus2.FACILITIES, **corpus2.DEV_FACILITY}:
            raise HTTPException(404, f"unknown facility {facility}")
        bias = corpus2.bias_list(facility) if facility else [p.strip() for p in (phrases or "").split(",") if p.strip()]
        state["rt"] = state["rt"] or _runtime()
        r = state["rt"].transcribe(x, bias=bias or None, beam=4 if bias else 1)
        g = _gate()
        row = {"hyp": r.text, "logprobs": r.logprobs, "audio_seconds": r.audio_seconds}
        risk = float(g(torch.from_numpy(features(row))[None])[0]) if g else None
        ex = extract(r.text, scoring_lexicon())
        decision = "confirm" if (g and risk >= g.threshold) or ex.needs_review else "file"
        return {"transcript": r.text, "completed_from_facility_list": r.biased, "risk": risk, "decision": decision,
                "vitals": ex.vitals, "doses": [{"drug": d[0], "amount": d[1], "unit": d[2]} for d in ex.doses],
                "review_flags": [{"field": f[0], "reason": f[1]} for f in ex.flags],
                "timing": {"decode_s": round(r.seconds, 3), "audio_s": round(r.audio_seconds, 2), "rtf": round(r.rtf, 3)}}

    @app.get("/v2/systems")
    def systems():
        p = OUT / "metrics.json"
        if not p.exists():
            raise HTTPException(404, "no benchmark yet: run scripts/v2_benchmark.sh")
        g = OUT / "gate.json"
        return {"metrics": json.loads(p.read_text()), "gate": json.loads(g.read_text()) if g.exists() else None}

    @app.get("/v2/utterances")
    def utterances(decoder: str = Query(...), set: str | None = None, condition: str | None = None,  # noqa: A002
                   errors_only: bool = False, limit: int = Query(200, le=1000), offset: int = 0):
        meta, lex = _meta(), scoring_lexicon()
        sel = [r for r in _rows() if r["decoder"] == decoder and (not set or r["set"] == set)
               and (not condition or r["condition"] == condition)]
        out = []
        g = _gate()
        for r in sel:
            err = chart_error(r, meta, lex)
            if errors_only and not err:
                continue
            s = score_utt(r["ref"], r["hyp"], lex)
            m = meta.get(r["id"], {})
            risk = float(g(torch.from_numpy(features(r))[None])[0]) if g else None
            out.append({**{k: r[k] for k in ("id", "set", "voice", "condition", "ref", "hyp", "biased", "seconds", "audio_seconds")},
                        "chart_error": err, "terms_missed": [t for t in s.terms_ref if t not in s.terms_hit],
                        "numbers_ref": s.nums_ref, "numbers_hyp": s.nums_hyp,
                        "resident": m.get("resident"), "resident_heard": bool(m.get("resident") and fold(m["resident"]) in fold(r["hyp"])),
                        "facility": m.get("facility"), "risk": risk, "flagged": bool(g and risk is not None and risk >= g.threshold)})
        return {"total": len(out), "items": out[offset: offset + limit]}

    @app.get("/v2/audio/{utt_id}")
    def wav(utt_id: str, voice: str, condition: str = "clean"):
        it = _meta().get(utt_id)
        if not it:
            raise HTTPException(404, f"no utterance {utt_id}")
        buf = io.BytesIO()
        sf.write(buf, np.clip(audio(it, voice, condition), -1, 1), SR, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav")

    dist = ROOT / "console" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="console")
    return app


app = create_app()
_ = FEATURES  # re-exported for clients that want the gate's feature order
