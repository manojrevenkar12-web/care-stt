#!/usr/bin/env bash
# Download Whisper (ONNX, int8+fp32), German Piper voices and Silero VAD
# from the sherpa-onnx GitHub releases into $CARE_STT_MODELS (default ~/models).
#   scripts/download_models.sh            # everything (~1.2 GB compressed)
#   scripts/download_models.sh --ci       # whisper-tiny + 3 voices + VAD only
set -euo pipefail
DEST="${CARE_STT_MODELS:-$HOME/models}"
R=https://github.com/k2-fsa/sherpa-onnx/releases/download
mkdir -p "$DEST" && cd "$DEST"

ASR=(whisper-tiny whisper-base whisper-small)
VOICES=(de_DE-thorsten-medium de_DE-eva_k-x_low de_DE-karlsson-low de_DE-kerstin-low de_DE-ramona-low de_DE-pavoque-low)
if [[ "${1:-}" == "--ci" ]]; then
  ASR=(whisper-tiny)
  VOICES=(de_DE-thorsten-medium de_DE-ramona-low de_DE-pavoque-low de_DE-eva_k-x_low de_DE-karlsson-low de_DE-kerstin-low)
fi

fetch() {  # name url
  [[ -d "$1" ]] && { echo "have $1"; return; }
  echo "get  $1"; curl -fsSL "$2" | tar xj
}
for m in "${ASR[@]}"; do fetch "sherpa-onnx-$m" "$R/asr-models/sherpa-onnx-$m.tar.bz2"; done
for v in "${VOICES[@]}"; do fetch "vits-piper-$v" "$R/tts-models/vits-piper-$v.tar.bz2"; done
[[ -f silero_vad.onnx ]] || curl -fsSL -o silero_vad.onnx "$R/asr-models/silero_vad.onnx"
echo "models in $DEST"
