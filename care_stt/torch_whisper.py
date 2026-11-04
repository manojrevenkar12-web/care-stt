"""Whisper in PyTorch, loaded from the same ONNX graphs the phone runtime uses.

sherpa-onnx ships Whisper as two ONNX graphs (encoder, decoder with a KV cache) and runs them on
Linux, Android and iOS. To train, we need the model in PyTorch. This module rebuilds the
architecture (Radford et al. 2022; the layout of openai/whisper's model.py) and copies every weight
out of the fp32 ONNX graphs, so fine-tuning starts from exactly the model the app would run.
`tests/test_torch_whisper.py` checks that both produce the same numbers, and `export.py` writes a
fine-tuned decoder back into the same ONNX interface, so the phone runtime loads it unchanged.

The ONNX export stores most linear weights under anonymous names (onnx::MatMul_842). They are
mapped back by the order in which the graph uses them, which follows the module order of the
original model: per encoder block q, k, v, out, fc1, fc2; then the cross-attention keys and values
of every decoder block (sherpa computes those in the encoder graph); per decoder block q, k, v,
out, cross q, cross out, fc1, fc2.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import MODELS_DIR, SR

N_FFT, HOP, N_MELS, CHUNK_S = 400, 160, 80, 30
N_FRAMES = CHUNK_S * SR // HOP  # 3000 mel frames = 1500 encoder positions


@dataclass(frozen=True)
class Dims:
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int
    n_vocab: int = 51865
    n_audio_ctx: int = 1500
    n_text_ctx: int = 448


# ---------------------------------------------------------------- features

@lru_cache(maxsize=1)
def mel_filters() -> np.ndarray:
    """80 Slaney-scale triangular filters over a 400-point FFT at 16 kHz, Slaney-normalised:
    the filterbank Whisper was trained with (librosa.filters.mel defaults)."""
    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        lin = f / (200.0 / 3)
        log = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
        return np.where(f >= 1000.0, log, lin)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        lin = m * (200.0 / 3)
        log = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
        return np.where(m >= 15.0, log, lin)

    fft_f = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(SR / 2), N_MELS + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fft_f[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (mel_pts[2:N_MELS + 2] - mel_pts[:N_MELS]))[:, None]
    return w.astype(np.float32)


def log_mel(audio: np.ndarray | Tensor, pad_to: int | None = N_FRAMES) -> Tensor:
    """Whisper's log-Mel spectrogram, (80, frames); padded or cut to 30 s by default."""
    x = torch.as_tensor(audio, dtype=torch.float32)
    if pad_to is not None:
        x = F.pad(x, (0, max(0, pad_to * HOP - x.shape[-1])))[: pad_to * HOP]
    spec = torch.stft(x, N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True)
    power = spec[..., :-1].abs() ** 2
    mel = torch.from_numpy(mel_filters()) @ power
    logm = torch.clamp(mel, min=1e-10).log10()
    logm = torch.maximum(logm, logm.max() - 8.0)
    return (logm + 4.0) / 4.0


# ---------------------------------------------------------------- model

class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query, self.key, self.value = nn.Linear(d, d), nn.Linear(d, d, bias=False), nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def attend(self, q: Tensor, k: Tensor, v: Tensor, causal: bool, offset: int = 0) -> Tensor:
        b, tq, d = q.shape
        h = self.heads
        scale = (d // h) ** -0.25
        q = q.view(b, tq, h, -1).transpose(1, 2) * scale
        k = k.view(b, k.shape[1], h, -1).transpose(1, 2) * scale
        v = v.view(b, v.shape[1], h, -1).transpose(1, 2)
        w = q @ k.transpose(-1, -2)
        if causal:
            tk = k.shape[2]
            mask = torch.ones(tq, tk, dtype=torch.bool).tril(diagonal=offset)
            w = w.masked_fill(~mask, float("-inf"))
        return self.out((w.softmax(-1) @ v).transpose(1, 2).reshape(b, tq, d))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, cross: bool):
        super().__init__()
        self.attn, self.attn_ln = MultiHeadAttention(d, heads), nn.LayerNorm(d)
        self.cross_attn = MultiHeadAttention(d, heads) if cross else None
        self.cross_attn_ln = nn.LayerNorm(d) if cross else None
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.mlp_ln = nn.LayerNorm(d)


class Whisper(nn.Module):
    def __init__(self, dims: Dims):
        super().__init__()
        self.dims = dims
        da, dt = dims.n_audio_state, dims.n_text_state
        self.conv1 = nn.Conv1d(N_MELS, da, 3, padding=1)
        self.conv2 = nn.Conv1d(da, da, 3, stride=2, padding=1)
        self.enc_pos = nn.Parameter(torch.zeros(dims.n_audio_ctx, da))
        self.enc_blocks = nn.ModuleList(Block(da, dims.n_audio_head, False) for _ in range(dims.n_audio_layer))
        self.enc_ln = nn.LayerNorm(da)
        self.tok = nn.Embedding(dims.n_vocab, dt)
        self.dec_pos = nn.Parameter(torch.zeros(dims.n_text_ctx, dt))
        self.dec_blocks = nn.ModuleList(Block(dt, dims.n_text_head, True) for _ in range(dims.n_text_layer))
        self.dec_ln = nn.LayerNorm(dt)

    def encode(self, mel: Tensor) -> Tensor:
        """(batch, 80, frames) -> audio features (batch, frames/2, d)."""
        x = F.gelu(self.conv2(F.gelu(self.conv1(mel)))).transpose(1, 2)
        x = x + self.enc_pos[: x.shape[1]]
        for b in self.enc_blocks:
            y = b.attn_ln(x)
            x = x + b.attn.attend(b.attn.query(y), b.attn.key(y), b.attn.value(y), causal=False)
            x = x + b.mlp(b.mlp_ln(x))
        return self.enc_ln(x)

    def cross_kv(self, audio: Tensor) -> tuple[Tensor, Tensor]:
        """Per decoder layer, cross-attention keys and values of the audio: (layers, batch, T, d)."""
        ks = torch.stack([b.cross_attn.key(audio) for b in self.dec_blocks])
        vs = torch.stack([b.cross_attn.value(audio) for b in self.dec_blocks])
        return ks, vs

    def decode(self, tokens: Tensor, cross_k: Tensor, cross_v: Tensor) -> Tensor:
        """Teacher-forced logits for a whole token sequence (training and scoring)."""
        x = self.tok(tokens) + self.dec_pos[: tokens.shape[1]]
        for i, b in enumerate(self.dec_blocks):
            y = b.attn_ln(x)
            x = x + b.attn.attend(b.attn.query(y), b.attn.key(y), b.attn.value(y), causal=True)
            y = b.cross_attn_ln(x)
            x = x + b.cross_attn.attend(b.cross_attn.query(y), cross_k[i], cross_v[i], causal=False)
            x = x + b.mlp(b.mlp_ln(x))
        return self.dec_ln(x) @ self.tok.weight.T

    def decode_step(self, tokens: Tensor, self_k: Tensor, self_v: Tensor, cross_k: Tensor, cross_v: Tensor, offset: int):
        """One incremental step with a KV cache, the exact interface of sherpa's decoder graph:
        caches are (layers, batch, n_text_ctx, d) and are updated at positions offset.."""
        n = tokens.shape[1]
        x = self.tok(tokens) + self.dec_pos[offset: offset + n]
        self_k, self_v = self_k.clone(), self_v.clone()
        for i, b in enumerate(self.dec_blocks):
            y = b.attn_ln(x)
            self_k[i, :, offset: offset + n] = b.attn.key(y)
            self_v[i, :, offset: offset + n] = b.attn.value(y)
            k, v = self_k[i, :, : offset + n], self_v[i, :, : offset + n]
            x = x + b.attn.attend(b.attn.query(y), k, v, causal=True, offset=offset)
            y = b.cross_attn_ln(x)
            x = x + b.cross_attn.attend(b.cross_attn.query(y), cross_k[i], cross_v[i], causal=False)
            x = x + b.mlp(b.mlp_ln(x))
        return self.dec_ln(x) @ self.tok.weight.T, self_k, self_v


# ---------------------------------------------------------------- loading from ONNX

def _onnx_weights(path: Path) -> tuple[dict[str, np.ndarray], list[np.ndarray], dict[str, str]]:
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(path))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    seen, seq = set(), []
    for n in m.graph.node:
        for x in n.input:
            if x in inits and x not in seen and not x.startswith(("audioEncoder.", "textDecoder.")):
                seen.add(x)
                seq.append(inits[x])
    meta = {p.key: p.value for p in m.metadata_props}
    return inits, seq, meta


def _set(lin: nn.Linear, w: np.ndarray, b: np.ndarray | None = None) -> None:
    lin.weight.data = torch.from_numpy(np.array(w.T, dtype=np.float32, copy=True))  # ONNX MatMul stores W^T
    if b is not None:
        lin.bias.data = torch.from_numpy(np.array(b, dtype=np.float32, copy=True))


def model_dir(size: str) -> Path:
    return Path(MODELS_DIR) / f"sherpa-onnx-whisper-{size}"


def load(size: str = "tiny") -> tuple[Whisper, dict[str, str]]:
    d = model_dir(size)
    enc_named, enc_seq, meta = _onnx_weights(d / f"{size}-encoder.onnx")
    dec_named, dec_seq, _ = _onnx_weights(d / f"{size}-decoder.onnx")
    dims = Dims(int(meta["n_audio_state"]), int(meta["n_audio_head"]), int(meta["n_audio_layer"]),
                int(meta["n_text_state"]), int(meta["n_text_head"]), int(meta["n_text_layer"]),
                int(meta["n_vocab"]), int(meta["n_audio_ctx"]), int(meta["n_text_ctx"]))
    m = Whisper(dims)
    t = lambda a: torch.from_numpy(np.array(a, dtype=np.float32, copy=True))  # noqa: E731
    # encoder
    e = iter(enc_seq)
    m.conv1.weight.data, m.conv1.bias.data = t(next(e)), t(next(e))
    m.conv2.weight.data, m.conv2.bias.data = t(next(e)), t(next(e))
    m.enc_pos.data = t(enc_named["audioEncoder.positional_embedding"])
    for i, b in enumerate(m.enc_blocks):
        p = f"audioEncoder.blocks.{i}."
        b.attn_ln.weight.data, b.attn_ln.bias.data = t(enc_named[p + "attn_ln.weight"]), t(enc_named[p + "attn_ln.bias"])
        b.mlp_ln.weight.data, b.mlp_ln.bias.data = t(enc_named[p + "mlp_ln.weight"]), t(enc_named[p + "mlp_ln.bias"])
        wq, bq, wk, wv, bv, wo, bo = (next(e) for _ in range(7))
        _set(b.attn.query, wq, bq), _set(b.attn.key, wk), _set(b.attn.value, wv, bv), _set(b.attn.out, wo, bo)
        w1, b1, w2, b2 = (next(e) for _ in range(4))
        _set(b.mlp[0], w1, b1), _set(b.mlp[2], w2, b2)
    m.enc_ln.weight.data, m.enc_ln.bias.data = t(enc_named["audioEncoder.ln_post.weight"]), t(enc_named["audioEncoder.ln_post.bias"])
    for b in m.dec_blocks:  # cross-attention keys and values live in the encoder graph
        wk, wv, bv = (next(e) for _ in range(3))
        _set(b.cross_attn.key, wk), _set(b.cross_attn.value, wv, bv)
    assert next(e, None) is None, "unexpected weights left in the encoder graph"
    # decoder
    q = iter(dec_seq)
    m.tok.weight.data = t(dec_named["textDecoder.token_embedding.weight"])
    m.dec_pos.data = t(dec_named["textDecoder.positional_embedding"])
    for i, b in enumerate(m.dec_blocks):
        p = f"textDecoder.blocks.{i}."
        for ln in ("attn_ln", "cross_attn_ln", "mlp_ln"):
            getattr(b, ln).weight.data = t(dec_named[p + ln + ".weight"])
            getattr(b, ln).bias.data = t(dec_named[p + ln + ".bias"])
        wq, bq, wk, wv, bv, wo, bo = (next(q) for _ in range(7))
        _set(b.attn.query, wq, bq), _set(b.attn.key, wk), _set(b.attn.value, wv, bv), _set(b.attn.out, wo, bo)
        cq, cbq, co, cbo = (next(q) for _ in range(4))
        _set(b.cross_attn.query, cq, cbq), _set(b.cross_attn.out, co, cbo)
        w1, b1, w2, b2 = (next(q) for _ in range(4))
        _set(b.mlp[0], w1, b1), _set(b.mlp[2], w2, b2)
    assert next(q, None) is None, "unexpected weights left in the decoder graph"
    m.dec_ln.weight.data, m.dec_ln.bias.data = t(dec_named["textDecoder.ln.weight"]), t(dec_named["textDecoder.ln.bias"])
    return m.eval(), meta


# ---------------------------------------------------------------- tokens

class Tokens:
    """Whisper's byte-level BPE vocabulary as sherpa ships it (base64 token, id)."""

    def __init__(self, size: str = "tiny"):
        self.pieces: dict[int, bytes] = {}
        for line in (model_dir(size) / f"{size}-tokens.txt").read_text().splitlines():
            tok, i = line.rsplit(" ", 1)
            self.pieces[int(i)] = base64.b64decode(tok)
        self.ids = {v: k for k, v in self.pieces.items()}
        self.eot, self.sot, self.de, self.en = 50257, 50258, 50261, 50259
        self.transcribe, self.no_timestamps = 50359, 50363

    def prefix(self, language: str = "de") -> list[int]:
        return [self.sot, self.de if language == "de" else self.en, self.transcribe, self.no_timestamps]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.pieces.get(i, b"") for i in ids if i < self.eot).decode("utf-8", errors="replace").strip()

    def encode(self, text: str) -> list[int]:
        """Byte-level BPE exactly as tiktoken applies Whisper's vocabulary: split into words with the
        GPT-2 pattern, then repeatedly merge the adjacent pair whose merged token has the lowest id
        (in tiktoken a token's id is its merge rank)."""
        out: list[int] = []
        for word in _PRETOKEN.findall(" " + text.strip()):
            parts = [bytes([b]) for b in word.encode("utf-8")]
            while len(parts) > 1:
                best, at = None, -1
                for i in range(len(parts) - 1):
                    r = self.ids.get(parts[i] + parts[i + 1])
                    if r is not None and r < self.eot and (best is None or r < best):
                        best, at = r, i
                if best is None:
                    break
                parts[at: at + 2] = [parts[at] + parts[at + 1]]
            out.extend(self.ids[p] for p in parts)
        return out


_PRETOKEN = __import__("regex").compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
