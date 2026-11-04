"""Write a (fine-tuned) PyTorch decoder back into sherpa-onnx's Whisper decoder interface.

The phone app keeps the encoder file it already has. Only the decoder graph is replaced, with the
same inputs, outputs and metadata as sherpa's export:

  tokens (n_audio, n_tokens) int64, in_n_layer_self_k_cache / v (layers, n_audio, 448, d),
  n_layer_cross_k / v (layers, n_audio, T, d), offset (1,) int64
  -> logits (n_audio, n_tokens, vocab), out_n_layer_self_k_cache / v

Then an int8 copy is made with ONNX Runtime's dynamic quantisation (weights int8, activations
quantised at run time), the same kind of file sherpa ships as *-decoder.int8.onnx.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import onnx
import torch
from torch import Tensor, nn

from . import torch_whisper as tw


class ExportDecoder(nn.Module):
    """decode_step with the offset as a tensor, so the graph works at every position."""

    def __init__(self, m: tw.Whisper):
        super().__init__()
        self.m = m

    def forward(self, tokens: Tensor, self_k: Tensor, self_v: Tensor, cross_k: Tensor, cross_v: Tensor, offset: Tensor):
        m = self.m
        n = tokens.shape[1]
        ctx = self_k.shape[2]
        pos = torch.arange(ctx)
        o = offset[0]
        new = (pos >= o) & (pos < o + n)  # cache rows this call writes
        idx = torch.clamp(pos - o, 0, n - 1)  # which new token goes into each written row
        x = m.tok(tokens) + m.dec_pos.index_select(0, (o + torch.arange(n)))
        outs_k, outs_v = [], []
        # attention over the whole fixed-size cache, masked to positions <= the query's own
        q_pos = o + torch.arange(n)
        mask = pos[None, :] <= q_pos[:, None]
        for i, b in enumerate(m.dec_blocks):
            y = b.attn_ln(x)
            k_new, v_new = b.attn.key(y), b.attn.value(y)
            ki = torch.where(new[None, :, None], k_new.index_select(1, idx), self_k[i])
            vi = torch.where(new[None, :, None], v_new.index_select(1, idx), self_v[i])
            outs_k.append(ki)
            outs_v.append(vi)
            x = x + _attend(b.attn, b.attn.query(y), ki, vi, mask)
            y = b.cross_attn_ln(x)
            x = x + _attend(b.cross_attn, b.cross_attn.query(y), cross_k[i], cross_v[i], None)
            x = x + b.mlp(b.mlp_ln(x))
        logits = torch.nn.functional.linear(m.dec_ln(x), m.tok.weight)
        return logits, torch.stack(outs_k), torch.stack(outs_v)


def _attend(att: tw.MultiHeadAttention, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
    b, tq, d = q.shape
    h = att.heads
    scale = (d // h) ** -0.25
    q = q.view(b, tq, h, -1).transpose(1, 2) * scale
    k = k.reshape(b, k.shape[1], h, -1).transpose(1, 2) * scale
    v = v.reshape(b, v.shape[1], h, -1).transpose(1, 2)
    w = q @ k.transpose(-1, -2)
    if mask is not None:
        w = w.masked_fill(~mask, float("-inf"))
    return att.out((w.softmax(-1) @ v).transpose(1, 2).reshape(b, tq, d))


def _fold_weight_transposes(g: onnx.ModelProto) -> None:
    """Exported without constant folding (so the tied embedding is stored once), every linear layer
    reads its weight through a Transpose node, which the quantiser cannot see through. Fold those
    transposes into the weights, except where the weight has other readers (the embedding, which
    the token lookup also uses): the result is sherpa's layout, int8 linears and one fp32 embedding."""
    from onnx import numpy_helper

    inits = {i.name: i for i in g.graph.initializer}
    readers: dict[str, int] = {}
    for n in g.graph.node:
        for x in n.input:
            readers[x] = readers.get(x, 0) + 1
    keep = []
    for n in g.graph.node:
        w = n.input[0] if n.input else None
        if n.op_type == "Transpose" and w in inits and readers.get(w) == 1:
            perm = next((list(a.ints) for a in n.attribute if a.name == "perm"), None)
            arr = numpy_helper.to_array(inits[w])
            t = numpy_helper.from_array(arr.transpose(perm) if perm else arr.T, name=n.output[0])
            g.graph.initializer.remove(inits[w])
            g.graph.initializer.append(t)
            continue
        keep.append(n)
    del g.graph.node[:]
    g.graph.node.extend(keep)


def export(m: tw.Whisper, size: str, out_dir: Path, int8: bool = True) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d, L, ctx = m.dims.n_text_state, len(m.dec_blocks), m.dims.n_text_ctx
    T = 700
    args = (torch.tensor([[50258, 50261, 50359, 50363]]), torch.zeros(L, 1, ctx, d), torch.zeros(L, 1, ctx, d),
            torch.randn(L, 1, T, d), torch.randn(L, 1, T, d), torch.tensor([0]))
    fp32 = out_dir / f"{size}-decoder.onnx"
    torch.onnx.export(
        ExportDecoder(m.eval()), args, str(fp32), dynamo=False, opset_version=17,
        do_constant_folding=False,
        input_names=["tokens", "in_n_layer_self_k_cache", "in_n_layer_self_v_cache", "n_layer_cross_k", "n_layer_cross_v", "offset"],
        output_names=["logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache"],
        dynamic_axes={"tokens": {0: "n_audio", 1: "n_tokens"}, "in_n_layer_self_k_cache": {1: "n_audio"},
                      "in_n_layer_self_v_cache": {1: "n_audio"}, "n_layer_cross_k": {1: "n_audio", 2: "T"},
                      "n_layer_cross_v": {1: "n_audio", 2: "T"}, "logits": {0: "n_audio", 1: "n_tokens"},
                      "out_n_layer_self_k_cache": {1: "n_audio"}, "out_n_layer_self_v_cache": {1: "n_audio"}},
    )
    # sherpa reads the model description from the metadata it wrote on its own export
    src = tw.model_dir(size)
    meta = {p.key: p.value for p in onnx.load(str(src / f"{size}-encoder.onnx"), load_external_data=False).metadata_props}
    g = onnx.load(str(fp32))
    del g.metadata_props[:]
    for k, v in meta.items():
        g.metadata_props.add(key=k, value=v)
    _fold_weight_transposes(g)
    onnx.save(g, str(fp32))
    out = {"decoder": fp32}
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = out_dir / f"{size}-decoder.int8.onnx"
        # linear layers only, like sherpa's own int8 files: the embedding and output projection stay fp32
        quantize_dynamic(str(fp32), str(q), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
        out["decoder_int8"] = q
    # the encoder and the tokens are unchanged: copy them next to the new decoder
    for name in (f"{size}-encoder.onnx", f"{size}-encoder.int8.onnx", f"{size}-tokens.txt"):
        if not (out_dir / name).exists():
            shutil.copy(src / name, out_dir / name)
    return out
