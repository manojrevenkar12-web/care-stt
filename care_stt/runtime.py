"""The decoding loop the app runs, over the deployed ONNX graphs, with contextual biasing.

sherpa-onnx decodes Whisper greedily and has no way to tell the model what a facility knows in
advance. An app can: it knows who lives in the care home and what is on the medication plan. This
loop runs the same encoder and decoder graphs through ONNX Runtime (the engine under sherpa-onnx on
Android and iOS) and adds two things:

  beam search   a few hypotheses instead of one, so a rare word has a chance to survive
  biasing       a prefix tree over the tokens of the facility's names and drugs. Each token that
                extends a phrase earns a bonus; when a partial match breaks off, the bonus it
                earned is taken back, so a phrase is only favoured once it is completed. This is
                shallow-fusion contextual biasing in the style of Zhao et al. (Interspeech 2019)
                and the WFST boosting used in production ASR, here at the token level.

It also returns per-token log-probabilities, which the quality gate reads.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import torch_whisper as tw
from .config import SR

TAIL_FRAMES = 1000


@dataclass
class Result:
    text: str
    tokens: list[int]
    logprobs: list[float]
    seconds: float
    audio_seconds: float
    biased: list[str] = field(default_factory=list)  # phrases the bias tree completed in the output

    @property
    def rtf(self) -> float:
        return self.seconds / max(self.audio_seconds, 1e-9)


class BiasTree:
    """Token prefix tree of the phrases a facility expects. Each phrase is added with and without
    a leading space and in the casing given, because Whisper's BPE tokenises a word differently at
    the start of a sentence, after a space or after punctuation."""

    def __init__(self, tok: tw.Tokens, phrases: list[str]):
        self.root: dict = {}
        self.phrases = list(dict.fromkeys(phrases))
        for p in self.phrases:
            for variant in (" " + p, p):
                ids = tok.encode(variant) if variant.startswith(" ") else _encode_nospace(tok, variant)
                node = self.root
                for depth, t in enumerate(ids, 1):
                    node = node.setdefault(t, {"_d": depth})
                node["$"] = p

    def step(self, active: list[dict], token: int) -> tuple[list[dict], int, int, str | None]:
        """Advance the partial matches by one token. Returns the new partial matches, how many
        match tokens were added (bonus), how many were lost (to take back), and a completed phrase."""
        nxt, completed = [], None
        for node in active + [self.root]:
            child = node.get(token)
            if child is not None:
                nxt.append(child)
                completed = completed or child.get("$")
        gained = 1 if nxt else 0
        lost = 0 if nxt else max((_depth(n) for n in active), default=0)
        keep = [n for n in nxt if any(isinstance(k, int) for k in n)]  # finished leaves stop being active
        return keep, gained, lost, completed

    def boosts(self, active: list[dict]) -> set[int]:
        out: set[int] = set()
        for node in active + [self.root]:
            out.update(k for k in node if isinstance(k, int))
        return out


def _encode_nospace(tok: tw.Tokens, s: str) -> list[int]:
    ids = tok.encode(s)
    first = tok.pieces[ids[0]] if ids else b""
    if first.startswith(b" "):  # encode() always adds a space; drop it for the no-space variant
        raw = s.encode("utf-8")
        out, i = [], 0
        while i < len(raw):
            for n in range(len(raw) - i, 0, -1):
                j = tok.ids.get(raw[i: i + n])
                if j is not None and j < tok.eot:
                    out.append(j)
                    i += n
                    break
            else:
                return ids
        return out
    return ids


def _depth(node: dict) -> int:
    return node.get("_d", 1)


class Runtime:
    def __init__(self, model_dir: Path | str, size: str = "base", quant: str = "int8", threads: int = 2,
                 encoder_dir: Path | str | None = None):
        d = Path(model_dir)
        e = Path(encoder_dir) if encoder_dir else d
        suffix = ".int8.onnx" if quant == "int8" else ".onnx"
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.enc = ort.InferenceSession(str(e / f"{size}-encoder{suffix}"), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(d / f"{size}-decoder{suffix}"), so, providers=["CPUExecutionProvider"])
        self.tok = tw.Tokens(size)
        shp = self.dec.get_inputs()[1].shape
        self.layers, self.ctx, self.dim = int(shp[0]), int(shp[2]), int(shp[3])
        self.size_mb = round(sum(p.stat().st_size for p in (e / f"{size}-encoder{suffix}", d / f"{size}-decoder{suffix}")) / 1e6, 1)
        self.suppress = np.array([self.tok.no_timestamps, self.tok.sot, 50360, 50361, 50362] + list(range(50259, 50358)))

    def _features(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio) // tw.HOP + TAIL_FRAMES
        return tw.log_mel(audio, pad_to=n + (n % 2)).numpy()[None]

    def _step(self, tokens, sk, sv, ck, cv, offset):
        lg, sk, sv = self.dec.run(None, {"tokens": tokens, "in_n_layer_self_k_cache": sk, "in_n_layer_self_v_cache": sv,
                                         "n_layer_cross_k": ck, "n_layer_cross_v": cv,
                                         "offset": np.array([offset], dtype=np.int64)})
        return lg[:, -1].astype(np.float64), sk, sv

    def transcribe(self, audio: np.ndarray, bias: list[str] | None = None, beam: int = 1, bonus: float = 2.0,
                   max_len: int = 96, sr: int = SR) -> Result:
        if sr != SR:
            raise ValueError(f"expected {SR} Hz audio")
        t0 = time.perf_counter()
        ck, cv = self.enc.run(None, {"mel": self._features(audio)})
        tree = BiasTree(self.tok, bias) if bias else None
        prefix = self.tok.prefix("de")
        L, ctx, d = self.layers, self.ctx, self.dim
        sk = np.zeros((L, 1, ctx, d), np.float32)
        sv = np.zeros_like(sk)
        lg, sk, sv = self._step(np.array([prefix], dtype=np.int64), sk, sv, ck, cv, 0)
        # hypotheses: (score without bias, bias earned, tokens, logprobs, active partial matches, completed)
        hyps = [(0.0, 0.0, [], [], [], [])]
        caches = (sk, sv)
        done = []
        offset = len(prefix)
        logits = lg
        for _ in range(max_len):
            logp = _log_softmax(logits)
            logp[:, self.suppress] = -np.inf
            cand = []
            for h, (s, b, toks, lps, active, comp) in enumerate(hyps):
                row = logp[h]
                boost = np.zeros_like(row)
                if tree is not None:
                    for t in tree.boosts(active):
                        boost[t] = bonus
                top = np.argsort(-(row + boost))[: max(beam, 2)]
                for t in top:
                    t = int(t)
                    nb, nact, ncomp = b, active, comp
                    if tree is not None:
                        nact, gained, lost, c = tree.step(active, t)
                        nb = b + bonus * gained - bonus * lost
                        ncomp = comp + [c] if c else comp
                    cand.append((s + row[t], nb, toks + [t], lps + [float(row[t])], nact, ncomp, h))
            cand.sort(key=lambda c: -(c[0] + c[1]))
            hyps, origin = [], []
            for c in cand:
                if c[2][-1] == self.tok.eot:
                    # a phrase left half-matched at the end gives its bonus back
                    pending = max((_depth(n) for n in c[4]), default=0) if tree else 0
                    done.append((c[0], c[1] - bonus * pending, c[2][:-1], c[3][:-1], c[5]))
                else:
                    hyps.append(c[:6])
                    origin.append(c[6])
                if len(hyps) == beam:
                    break
            if not hyps or len(done) >= beam:
                break
            idx = np.array(origin)
            sk, sv = caches[0][:, idx], caches[1][:, idx]
            n = len(hyps)
            logits, sk, sv = self._step(np.array([[h[2][-1]] for h in hyps], dtype=np.int64), sk, sv,
                                        np.repeat(ck, n, axis=1), np.repeat(cv, n, axis=1), offset)
            caches = (sk, sv)
            offset += 1
        if not done:
            done = [(h[0], h[1], h[2], h[3], h[5]) for h in hyps]
        best = max(done, key=lambda x: (x[0] + x[1]) / max(1, len(x[2])) ** 0.6)
        return Result(self.tok.decode(best[2]), best[2], best[3], time.perf_counter() - t0, len(audio) / SR,
                      sorted(set(best[4])))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))
