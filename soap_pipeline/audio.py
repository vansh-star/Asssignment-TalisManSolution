"""Audio loading utilities.

All ASR backends expect 16 kHz mono float32 PCM. We decode with ffmpeg
(handles mp3/m4a/wav/etc.) so there is a single, predictable code path.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np

SAMPLE_RATE = 16_000


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio file to a mono float32 numpy array in [-1, 1]."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install it first (see README).")

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-ac", "1",                 # mono
        "-ar", str(sample_rate),    # resample
        "-f", "s16le", "-",         # raw 16-bit PCM to stdout
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def duration_seconds(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return len(samples) / float(sample_rate)
