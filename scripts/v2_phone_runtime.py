"""Does the fine-tuned decoder run, unchanged, in the runtime the phone uses?

For stock and fine-tuned decoders: decode the same held-out notes with sherpa-onnx itself (greedy,
10 s tail padding) and with this repository's ONNX Runtime loop (greedy), and report how often the
two give the same text, each one's term recall, size on disk and real-time factor at 1 and 2 threads.
Writes results/v2/phone_runtime.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import sherpa_onnx  # noqa: E402

from care_stt import eval2, synth  # noqa: E402
from care_stt.config import MODELS_DIR  # noqa: E402
from care_stt.metrics import aggregate, score_utt  # noqa: E402
from care_stt.runtime import TAIL_FRAMES, Runtime  # noqa: E402

STOCK = Path(MODELS_DIR) / "sherpa-onnx-whisper-base"
FT = Path(MODELS_DIR) / "finetuned" / "base-care-export"


def main(n: int = 40) -> None:
    notes = eval2.items("facility")[:n // 2] + eval2.items("v1")[:n // 2]
    lex = eval2.scoring_lexicon()
    out = {}
    for name, d in (("stock", STOCK), ("finetuned", FT)):
        res = {"size_mb": round(sum(p.stat().st_size for p in (STOCK / "base-encoder.int8.onnx", d / "base-decoder.int8.onnx")) / 1e6, 1)}
        for threads in (1, 2):
            rec = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(STOCK / "base-encoder.int8.onnx"), decoder=str(d / "base-decoder.int8.onnx"),
                tokens=str(STOCK / "base-tokens.txt"), language="de", num_threads=threads, tail_paddings=TAIL_FRAMES)
            rt = Runtime(d, "base", "int8", threads=threads, encoder_dir=STOCK)
            same, sh_scores, rt_scores, rtf = 0, [], [], []
            for it in notes:
                a = eval2.audio(it, synth.TEST_VOICES[0], "clean")
                st = rec.create_stream()
                st.accept_waveform(16000, a)
                t0 = time.perf_counter()
                rec.decode_stream(st)
                rtf.append((time.perf_counter() - t0) / (len(a) / 16000))
                text = st.result.text.strip()
                mine = rt.transcribe(a).text
                same += text == mine
                sh_scores.append(score_utt(it["text"], text, lex))
                rt_scores.append(score_utt(it["text"], mine, lex))
            res[f"threads_{threads}"] = {
                "sherpa_rtf_mean": round(float(np.mean(rtf)), 3), "sherpa_rtf_p95": round(float(np.percentile(rtf, 95)), 3),
                "same_text_as_this_runtime": round(same / len(notes), 3),
                "sherpa_term_recall": round(aggregate(sh_scores).term_recall, 4),
                "this_runtime_term_recall": round(aggregate(rt_scores).term_recall, 4),
                "sherpa_wer": round(aggregate(sh_scores).wer, 4)}
            print(name, threads, res[f"threads_{threads}"])
        out[name] = res
    (eval2.OUT / "phone_runtime.json").write_text(json.dumps({"notes": len(notes), "voice": synth.TEST_VOICES[0], **out}, indent=1) + "\n")


if __name__ == "__main__":
    main()
