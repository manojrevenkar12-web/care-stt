"""Integration tests with real models; skipped when weights are not present."""
import pytest

from care_stt.config import MODELS_DIR

needs_models = pytest.mark.skipif(
    not (MODELS_DIR / "sherpa-onnx-whisper-tiny").exists()
    or not (MODELS_DIR / "vits-piper-de_DE-thorsten-medium").exists(),
    reason="models not downloaded (scripts/download_models.sh)")


@needs_models
def test_tts_to_asr_roundtrip():
    import numpy as np

    from care_stt.asr import WhisperASR
    from care_stt.normalize import normalize
    from care_stt.synth import synthesize

    ref = "Die Bewohnerin hat heute Nacht gut geschlafen und morgens gefrühstückt."
    x = synthesize(ref, "de_DE-thorsten-medium")
    assert x.dtype == np.float32 and len(x) > 8000
    out = WhisperASR("tiny", "int8").transcribe(x)
    assert out.text and out.rtf < 5
    overlap = set(normalize(ref).split()) & set(normalize(out.text).split())
    assert len(overlap) >= 5, out.text


@needs_models
def test_vad_splits_long_audio():
    import numpy as np

    from care_stt.asr import VadSegmenter
    from care_stt.config import SR
    from care_stt.synth import synthesize

    a = synthesize("Lagerung nach Plan durchgeführt.", "de_DE-thorsten-medium")
    long = np.concatenate([a, np.zeros(SR * 2, dtype=np.float32), a])
    if not (MODELS_DIR / "silero_vad.onnx").exists():
        pytest.skip("silero_vad.onnx missing")
    assert len(VadSegmenter().segments(long)) == 2
