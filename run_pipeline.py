#!/usr/bin/env python3
"""End-to-end CLI: audio -> verbatim transcript -> validated SOAP note.

Examples
--------
# fully local (faster-whisper + Ollama)
python run_pipeline.py data/sample_dictation.mp3

# Whisper via sherpa-onnx, SOAP via Claude API
python run_pipeline.py data/sample_dictation.mp3 --asr sherpa-onnx --llm anthropic

# skip ASR and reuse an existing transcript
python run_pipeline.py --transcript outputs/transcript.txt --llm ollama
"""
from __future__ import annotations

import argparse
import json
import os

from soap_pipeline.soap import generate_soap, to_markdown
from soap_pipeline.transcribe import transcribe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", default="data/sample_dictation.mp3")
    ap.add_argument("--transcript", help="Use an existing transcript .txt instead of running ASR")
    ap.add_argument("--asr", default="faster-whisper", choices=["faster-whisper", "sherpa-onnx"])
    ap.add_argument("--asr-model", default="medium.en", help="faster-whisper size: small.en / medium.en / large-v3")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--engine", default="llm", choices=["llm", "rules"])
    ap.add_argument("--llm", default="ollama", choices=["ollama", "openai", "anthropic"])
    ap.add_argument("--llm-model", default=None, help="e.g. llama3.1:8b, mistral, gpt-4o-mini")
    ap.add_argument("--no-llm-review", action="store_true", help="Disable LLM-as-judge Objective audit")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---------------- Part A: transcription ----------------
    if args.transcript:
        text = open(args.transcript, encoding="utf-8").read().strip()
    else:
        print(f"[asr] transcribing {args.audio} with {args.asr} ...")
        raw = transcribe(args.audio, backend=args.asr, model=args.asr_model, device=args.device, postprocess=False)
        with open(f"{args.out}/transcript_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw.text + "\n")
        from soap_pipeline.medical_terms import fix_medical_terms
        raw.text = fix_medical_terms(raw.text)
        for s in raw.segments:
            s.text = fix_medical_terms(s.text)
        text = raw.text
        with open(f"{args.out}/transcript.txt", "w", encoding="utf-8") as f:
            f.write(text + "\n")
        with open(f"{args.out}/transcript_segments.json", "w", encoding="utf-8") as f:
            json.dump(raw.to_dict(), f, indent=2)
        print(f"[asr] {raw.audio_seconds:.0f}s audio in {raw.elapsed_seconds:.0f}s -> {args.out}/transcript.txt")

    # ---------------- Part B: SOAP ----------------
    print(f"[soap] generating note (engine={args.engine}, llm={args.llm}) ...")
    note = generate_soap(text, engine=args.engine, provider=args.llm, model=args.llm_model,
                         llm_review=not args.no_llm_review)
    with open(f"{args.out}/soap_note.json", "w", encoding="utf-8") as f:
        json.dump(note, f, indent=2, ensure_ascii=False)
    with open(f"{args.out}/soap_note.md", "w", encoding="utf-8") as f:
        f.write(to_markdown(note) + "\n")
    v = note["_meta"]["validation"]
    print(f"[soap] done via {note['_meta']['engine']} | moved {len(v['subjective_in_objective_moved'])} "
          f"subjective item(s) out of Objective | ungrounded numbers: {v['ungrounded_numbers'] or 'none'}")
    print(f"[soap] -> {args.out}/soap_note.json, {args.out}/soap_note.md")


if __name__ == "__main__":
    main()
