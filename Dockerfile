FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl bzip2 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# whisper-base (int8 + fp32) and Silero VAD, from the sherpa-onnx GitHub releases
ENV CARE_STT_MODELS=/models CARE_STT_MODEL=base-int8 CARE_STT_THRESHOLD=0.75
RUN mkdir -p /models && cd /models \
 && R=https://github.com/k2-fsa/sherpa-onnx/releases/download \
 && curl -fsSL $R/asr-models/sherpa-onnx-whisper-base.tar.bz2 | tar xj \
 && curl -fsSL -o silero_vad.onnx $R/asr-models/silero_vad.onnx

COPY care_stt care_stt
COPY data data
EXPOSE 8000
CMD ["uvicorn", "care_stt.service:app", "--host", "0.0.0.0", "--port", "8000"]
