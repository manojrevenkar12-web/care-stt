"""v2 with real Whisper weights (tiny, from the sherpa-onnx release); skipped when absent.

These are the claims the training and deployment path rest on: the PyTorch port computes what the
ONNX graphs compute, the exported decoder is a drop-in for sherpa's, and the BPE encoder produces
the token sequences Whisper itself produces."""
import numpy as np
import pytest
import torch

from care_stt.config import MODELS_DIR

needs_tiny = pytest.mark.skipif(not (MODELS_DIR / "sherpa-onnx-whisper-tiny").exists(),
                                reason="whisper-tiny not downloaded (scripts/download_models.sh --ci)")


def _wav():
    import soundfile as sf

    a, _ = sf.read(str(MODELS_DIR / "sherpa-onnx-whisper-tiny/test_wavs/0.wav"), dtype="float32")
    return a


@needs_tiny
def test_pytorch_port_matches_onnx_graphs():
    import onnxruntime as ort

    from care_stt import torch_whisper as tw

    m, meta = tw.load("tiny")
    assert meta["n_text_layer"] == "4"
    mel = tw.log_mel(_wav())[None]
    d = tw.model_dir("tiny")
    ck, cv = ort.InferenceSession(str(d / "tiny-encoder.onnx")).run(None, {"mel": mel.numpy()})
    with torch.no_grad():
        k, v = m.cross_kv(m.encode(mel))
    assert np.abs(k.numpy() - ck).max() < 5e-3 * np.abs(ck).max()
    assert np.abs(v.numpy() - cv).max() < 5e-3 * np.abs(cv).max()
    tok = tw.Tokens("tiny")
    pre = np.array([tok.prefix("en")], dtype=np.int64)
    L, ctx, D = 4, 448, 384
    z = np.zeros((L, 1, ctx, D), np.float32)
    dec = ort.InferenceSession(str(d / "tiny-decoder.onnx"))
    logits, _, _ = dec.run(None, {"tokens": pre, "in_n_layer_self_k_cache": z, "in_n_layer_self_v_cache": z,
                                  "n_layer_cross_k": ck, "n_layer_cross_v": cv, "offset": np.array([0])})
    with torch.no_grad():
        mine = m.decode(torch.from_numpy(pre), torch.from_numpy(ck), torch.from_numpy(cv)).numpy()
    assert np.abs(mine - logits).max() < 1e-3
    assert (mine.argmax(-1) == logits.argmax(-1)).all()


@needs_tiny
def test_incremental_decoding_equals_teacher_forcing():
    from care_stt import torch_whisper as tw

    m, _ = tw.load("tiny")
    torch.manual_seed(0)
    k, v = torch.randn(4, 1, 300, 384), torch.randn(4, 1, 300, 384)
    seq = torch.tensor([[50258, 50261, 50359, 50363, 1000, 2000, 3000]])
    with torch.no_grad():
        full = m.decode(seq, k, v)
        sk = torch.zeros(4, 1, 448, 384)
        sv = sk.clone()
        lg, sk, sv = m.decode_step(seq[:, :4], sk, sv, k, v, 0)
        steps = [lg]
        for i in range(4, 7):
            lg, sk, sv = m.decode_step(seq[:, i: i + 1], sk, sv, k, v, i)
            steps.append(lg)
    inc = torch.cat(steps, dim=1)
    assert torch.allclose(full, inc, atol=1e-4)


@needs_tiny
def test_exported_decoder_is_a_drop_in_for_sherpa(tmp_path):
    import onnxruntime as ort

    from care_stt import export
    from care_stt import torch_whisper as tw

    m, _ = tw.load("tiny")
    out = export.export(m, "tiny", tmp_path, int8=False)
    ref = ort.InferenceSession(str(tw.model_dir("tiny") / "tiny-decoder.onnx"))
    new = ort.InferenceSession(str(out["decoder"]))
    assert [i.name for i in ref.get_inputs()] == [i.name for i in new.get_inputs()]
    assert [o.name for o in ref.get_outputs()] == [o.name for o in new.get_outputs()]
    rng = np.random.default_rng(1)
    ck = rng.standard_normal((4, 2, 200, 384)).astype(np.float32)
    feeds = {"tokens": np.array([[50258, 50261, 50359, 50363]] * 2), "n_layer_cross_k": ck, "n_layer_cross_v": ck,
             "in_n_layer_self_k_cache": np.zeros((4, 2, 448, 384), np.float32),
             "in_n_layer_self_v_cache": np.zeros((4, 2, 448, 384), np.float32), "offset": np.array([0])}
    a, b = ref.run(None, feeds), new.run(None, feeds)
    assert np.abs(a[0] - b[0]).max() < 1e-3 and np.abs(a[1] - b[1]).max() < 1e-5
    feeds.update(tokens=np.array([[1000], [2000]]), in_n_layer_self_k_cache=a[1], in_n_layer_self_v_cache=a[2], offset=np.array([4]))
    a2, b2 = ref.run(None, feeds), new.run(None, feeds)
    assert np.abs(a2[0] - b2[0]).max() < 1e-3
    import onnx
    meta = {p.key: p.value for p in onnx.load(str(out["decoder"])).metadata_props}
    assert meta["model_type"] == "whisper-tiny" and meta["n_text_layer"] == "4"


@needs_tiny
def test_bpe_reproduces_whispers_own_tokenisation():
    """For every transcript Whisper generates, encoding its text gives back Whisper's token ids."""
    from care_stt import torch_whisper as tw

    tok = tw.Tokens("tiny")
    for s in ["After early nightfall, the yellow lamps would light up here and there.",
              "Blutdruck 130 zu 80, Puls 72.", "Frau Wiśniewski erhielt Edoxaban 20 Milligramm.", "Größe: 1,72 m; 37,8 °C"]:
        ids = tok.encode(s)
        assert tok.decode(ids) == s
    m, _ = tw.load("tiny")
    with torch.no_grad():
        mel = tw.log_mel(_wav())[None]
        k, v = m.cross_kv(m.encode(mel))
        ids = tok.prefix("en")
        for _ in range(40):
            n = int(m.decode(torch.tensor([ids]), k, v)[0, -1].argmax())
            if n == tok.eot:
                break
            ids.append(n)
    generated = ids[4:]
    assert tok.encode(tok.decode(generated)) == generated
