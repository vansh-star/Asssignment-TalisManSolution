#!/usr/bin/env bash
# Downloads the Whisper medium.en ONNX model + Silero VAD (only needed for --asr sherpa-onnx)
set -e
mkdir -p models && cd models
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
curl -L -O $BASE/silero_vad.onnx
curl -L $BASE/sherpa-onnx-whisper-medium.en.tar.bz2 | tar xj
