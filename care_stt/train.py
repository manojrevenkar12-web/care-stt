"""Fine-tune Whisper's decoder on care documentation, in PyTorch, on CPU.

What is trained and what is not:
  * the audio encoder and the decoder's cross-attention key/value projections are frozen. In the
    deployed graphs both live in the encoder ONNX file, so the phone keeps the encoder it already
    has and only the decoder file changes (export.py);
  * the token embedding (tied to the output layer) is frozen, so words the model was never shown
    keep the representation they had: that is what lets contextual biasing reach them later;
  * everything else in the decoder is trained: self-attention, cross-attention queries and output,
    MLPs and layer norms.

Data: data/train.jsonl rendered by the four training voices at a random speaking rate, and per
epoch a fresh random acoustic condition per utterance (ward babble, reverb, phone line, Opus,
clean), so the decoder learns the domain's words and number formats under realistic audio rather
than one fixed rendering. Input is padded the way the deployed runtime pads it (TAIL_FRAMES).
Early stopping on the v1 dev split (different sentences, written by hand).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import corpus2, data, degrade, synth
from . import torch_whisper as tw
from .config import CACHE_DIR, MODELS_DIR, SR
from .lexicon import Lexicon
from .metrics import aggregate, score_utt

TAIL_FRAMES = 1000  # 10 s of padding after the speech: what sherpa-onnx's Whisper runtime adds by default
TRAIN_CONDITIONS = ["clean", "clean", "babble_10", "babble_5", "reverb", "telephone", "opus_8k"]


@dataclass
class Config:
    size: str = "base"
    epochs: int = 3
    batch: int = 8
    lr: float = 2e-5
    warmup: int = 30
    weight_decay: float = 0.01
    eval_every: int = 60
    dev_n: int = 40
    seed: int = 20261104
    limit: int | None = None


def features(audio: np.ndarray, tail: int = TAIL_FRAMES) -> torch.Tensor:
    n = len(audio) // tw.HOP + tail
    return tw.log_mel(audio, pad_to=n + (n % 2))


def train_audio(item: dict, voice: str, speed: float) -> np.ndarray:
    p = CACHE_DIR / "train_audio" / voice / f"{item['id']}.npy"
    if p.exists():
        return np.load(p).astype(np.float32)
    x = synth.synthesize(item["text"], voice, speed)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, x.astype(np.float16))
    return x


def rendering(item: dict, rng: np.random.Generator) -> tuple[str, float]:
    voice = synth.DEV_VOICES[int(rng.integers(len(synth.DEV_VOICES)))]
    return voice, float(rng.uniform(0.9, 1.15))


def example(item: dict, pool: list[dict], epoch: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(synth.seed_of(item["id"], "render"))
    voice, speed = rendering(item, rng)
    x = train_audio(item, voice, speed)
    erng = np.random.default_rng(synth.seed_of(item["id"], str(epoch)))
    cond = TRAIN_CONDITIONS[int(erng.integers(len(TRAIN_CONDITIONS)))]
    if cond == "clean":
        return x, cond
    srcs = []
    if cond.startswith("babble"):
        for j in erng.choice(len(pool), size=4, replace=False):
            o = pool[int(j)]
            if o["id"] != item["id"]:
                v, s = rendering(o, np.random.default_rng(synth.seed_of(o["id"], "render")))
                srcs.append(train_audio(o, v, s))
    return degrade.apply(cond, x, srcs, synth.seed_of(item["id"], cond, str(epoch))), cond


def trainable(m: tw.Whisper) -> list[torch.nn.Parameter]:
    for p in m.parameters():
        p.requires_grad = False
    params = []
    for b in m.dec_blocks:
        for mod in (b.attn, b.attn_ln, b.cross_attn_ln, b.mlp, b.mlp_ln):
            params += list(mod.parameters())
        params += list(b.cross_attn.query.parameters()) + list(b.cross_attn.out.parameters())
    params += list(m.dec_ln.parameters())
    for p in params:
        p.requires_grad = True
    return params


@torch.no_grad()
def encode_batch(m: tw.Whisper, audios: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    mels = [features(a) for a in audios]
    T = max(x.shape[-1] for x in mels)
    batch = torch.stack([F.pad(x, (0, T - x.shape[-1]), value=float(x.min())) for x in mels])
    return m.cross_kv(m.encode(batch))


def targets(tok: tw.Tokens, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher forcing: inputs are prefix + text, labels are text + <eot>; prefix positions are ignored."""
    seqs = [tok.prefix("de") + tok.encode(t) + [tok.eot] for t in texts]
    L = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), L - 1), tok.eot, dtype=torch.long)
    lab = torch.full((len(seqs), L - 1), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        inp[i, : len(s) - 1] = torch.tensor(s[:-1])
        lab[i, 3: len(s) - 1] = torch.tensor(s[4:])  # predict from the last prefix token onwards
    return inp, lab


@torch.no_grad()
def greedy(m: tw.Whisper, tok: tw.Tokens, audio: np.ndarray, max_len: int = 96) -> str:
    k, v = encode_batch(m, [audio])
    ids = tok.prefix("de")
    L, d = len(m.dec_blocks), m.dims.n_text_state
    sk = torch.zeros(L, 1, m.dims.n_text_ctx, d)
    sv = sk.clone()
    lg, sk, sv = m.decode_step(torch.tensor([ids]), sk, sv, k, v, 0)
    off, out = len(ids), []
    for _ in range(max_len):
        n = int(lg[0, -1].argmax())
        if n == tok.eot:
            break
        out.append(n)
        lg, sk, sv = m.decode_step(torch.tensor([[n]]), sk, sv, k, v, off)
        off += 1
    return tok.decode(out)


def dev_score(m: tw.Whisper, tok: tw.Tokens, lex: Lexicon, n: int) -> dict:
    """WER, term recall and number accuracy on hand-written v1 dev notes, dev voices, clean and babble."""
    dev = data.load_corpus(split="dev")[:n]
    sc = []
    for i, u in enumerate(dev):
        voice = synth.DEV_VOICES[i % len(synth.DEV_VOICES)]
        cond = "clean" if i % 2 == 0 else "babble_10"
        a = data.audio(u, voice, cond, dev)
        sc.append(score_utt(u.text, greedy(m, tok, a), lex))
    ag = aggregate(sc).as_dict()
    return {k: ag[k] for k in ("wer", "term_recall", "number_recall", "number_utt_acc")}


def run(cfg: Config = Config(), out_dir: Path | None = None, log=print) -> Path:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(2)
    m, meta = tw.load(cfg.size)
    tok, lex = tw.Tokens(cfg.size), Lexicon.load()
    items = corpus2.load("train.jsonl")[: cfg.limit]
    params = trainable(m)
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(items) / cfg.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * max(0.0, 1 - s / max(1, steps)))
    tag = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:8]
    out_dir = out_dir or Path(MODELS_DIR) / "finetuned" / f"{cfg.size}-care-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = {"config": asdict(cfg), "trainable_params": n_train, "steps": [], "dev": []}
    m.eval()  # no dropout in Whisper; eval() keeps behaviour identical to inference
    base = dev_score(m, tok, lex, cfg.dev_n)
    history["dev"].append({"step": 0, **base})
    log(f"trainable parameters {n_train / 1e6:.1f}M; dev before {base}")
    best, step, t0 = base["wer"] - base["term_recall"], 0, time.time()
    torch.save({k: v for k, v in m.state_dict().items() if k.startswith(("dec_", "tok"))}, out_dir / "decoder.pt")
    for epoch in range(cfg.epochs):
        order = np.random.default_rng(cfg.seed + epoch).permutation(len(items))
        for i in range(0, len(order), cfg.batch):
            batch = [items[int(j)] for j in order[i: i + cfg.batch]]
            audios = [example(it, items, epoch)[0] for it in batch]
            k, v = encode_batch(m, audios)
            inp, lab = targets(tok, [it["text"] for it in batch])
            logits = m.decode(inp, k, v)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            history["steps"].append({"step": step, "epoch": epoch, "loss": round(float(loss), 4),
                                     "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)})
            if step % 10 == 0:
                log(f"step {step}/{steps} epoch {epoch} loss {float(loss):.3f} ({time.time() - t0:.0f}s)")
            if step % cfg.eval_every == 0 or step == steps:
                d = dev_score(m, tok, lex, cfg.dev_n)
                history["dev"].append({"step": step, **d})
                score = d["wer"] - d["term_recall"]
                log(f"dev at step {step}: {d}")
                if score < best:
                    best = score
                    torch.save({k: v for k, v in m.state_dict().items() if k.startswith(("dec_", "tok"))}, out_dir / "decoder.pt")
                    history["best_step"] = step
                (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    history["seconds"] = round(time.time() - t0)
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    (out_dir / "meta.json").write_text(json.dumps(meta))
    log(f"done in {history['seconds']}s; best step {history.get('best_step', 0)} -> {out_dir}")
    return out_dir


def load_finetuned(path: Path, size: str = "base") -> tw.Whisper:
    m, _ = tw.load(size)
    m.load_state_dict(torch.load(Path(path) / "decoder.pt"), strict=False)
    return m.eval()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=60)
    ap.add_argument("--dev-n", type=int, default=40)
    a = ap.parse_args()
    run(Config(size=a.size, epochs=a.epochs, limit=a.limit, eval_every=a.eval_every, dev_n=a.dev_n))
