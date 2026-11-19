"""Part A - Speech-to-text.

Two interchangeable, fully LOCAL backends (no audio ever leaves the machine):

1. ``faster-whisper`` (default)  - CTranslate2 re-implementation of OpenAI Whisper.
   ~4x faster than the reference implementation, int8 on CPU / fp16 on GPU,
   built-in Silero VAD, and supports an ``initial_prompt`` which we use to bias
   decoding toward medical vocabulary.

2. ``sherpa-onnx``  - the same Whisper weights exported to ONNX and run with
   onnxruntime. Useful on machines where CTranslate2 wheels are unavailable or
   where models must be fetched from GitHub instead of Hugging Face.
   (This backend produced the transcript committed in ``outputs/``.)

Both return the same ``Transcript`` object so the rest of the pipeline does not
care which engine was used.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict

from .audio import SAMPLE_RATE, duration_seconds, load_audio
from .medical_terms import MEDICAL_PROMPT, fix_medical_terms


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    segments: list[Segment] = field(default_factory=list)
    backend: str = ""
    model: str = ""
    audio_seconds: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Backend 1: faster-whisper
# --------------------------------------------------------------------------- #
def _transcribe_faster_whisper(audio_path: str, model_size: str, device: str) -> Transcript:
    from faster_whisper import WhisperModel  # imported lazily: optional dep

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    t0 = time.time()
    seg_iter, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        temperature=0.0,               # deterministic, verbatim decoding
        initial_prompt=MEDICAL_PROMPT,  # vocabulary biasing for drug/anatomy names
        condition_on_previous_text=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    segments = [Segment(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in seg_iter]
    return Transcript(
        text=" ".join(s.text for s in segments),
        segments=segments,
        backend="faster-whisper",
        model=model_size,
        audio_seconds=round(info.duration, 2),
        elapsed_seconds=round(time.time() - t0, 1),
    )


# --------------------------------------------------------------------------- #
# Backend 2: sherpa-onnx (Whisper ONNX + Silero VAD)
# --------------------------------------------------------------------------- #
def _transcribe_sherpa(audio_path: str, model_dir: str, vad_model: str, num_threads: int) -> Transcript:
    import sherpa_onnx

    name = os.path.basename(model_dir.rstrip("/")).replace("sherpa-onnx-whisper-", "")
    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=os.path.join(model_dir, f"{name}-encoder.int8.onnx"),
        decoder=os.path.join(model_dir, f"{name}-decoder.int8.onnx"),
        tokens=os.path.join(model_dir, f"{name}-tokens.txt"),
        language="en",
        task="transcribe",
        num_threads=num_threads,
        decoding_method="greedy_search",
        tail_paddings=800,
    )

    # Whisper's window is 30 s, so we split on silences with Silero VAD and keep
    # every chunk under ~25 s. Splitting on pauses (not fixed windows) avoids
    # cutting words in half.
    vad_cfg = sherpa_onnx.VadModelConfig()
    vad_cfg.silero_vad.model = vad_model
    vad_cfg.silero_vad.threshold = 0.4
    vad_cfg.silero_vad.min_silence_duration = 0.4
    vad_cfg.silero_vad.min_speech_duration = 0.25
    vad_cfg.silero_vad.max_speech_duration = 25
    vad_cfg.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=60)

    samples = load_audio(audio_path)
    t0 = time.time()
    segments: list[Segment] = []

    def _drain() -> None:
        while not vad.empty():
            seg = vad.front
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, seg.samples)
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                start = seg.start / SAMPLE_RATE
                segments.append(Segment(round(start, 2), round(start + len(seg.samples) / SAMPLE_RATE, 2), text))
            vad.pop()

    window = vad_cfg.silero_vad.window_size
    for i in range(0, len(samples), window):
        vad.accept_waveform(samples[i:i + window])
        _drain()
    vad.flush()
    _drain()

    return Transcript(
        text=" ".join(s.text for s in segments),
        segments=segments,
        backend="sherpa-onnx",
        model=f"whisper-{name} (int8 ONNX)",
        audio_seconds=round(duration_seconds(samples), 2),
        elapsed_seconds=round(time.time() - t0, 1),
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def transcribe(
    audio_path: str,
    backend: str = "faster-whisper",
    model: str = "medium.en",
    device: str = "cpu",
    sherpa_model_dir: str = "models/sherpa-onnx-whisper-medium.en",
    vad_model: str = "models/silero_vad.onnx",
    num_threads: int = os.cpu_count() or 2,
    postprocess: bool = True,
) -> Transcript:
    """Transcribe ``audio_path`` verbatim with a local Whisper model."""
    if backend == "faster-whisper":
        tr = _transcribe_faster_whisper(audio_path, model, device)
    elif backend == "sherpa-onnx":
        tr = _transcribe_sherpa(audio_path, sherpa_model_dir, vad_model, num_threads)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    if postprocess:
        # Conservative, dictionary-based fixes for known Whisper mis-hearings of
        # medical terms. Only exact phrase matches are touched -> stays verbatim.
        tr.text = fix_medical_terms(tr.text)
        for s in tr.segments:
            s.text = fix_medical_terms(s.text)
    return tr
