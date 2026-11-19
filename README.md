# Medical Dictation → SOAP Note Pipeline

Transcribes the provided clinical audio (`data/sample_dictation.mp3`) **verbatim with a local Whisper model** and turns the transcript into a **structured, validated SOAP note** (JSON + Markdown).

```
 sample_dictation.mp3
        │  ffmpeg → 16 kHz mono
        ▼
 ┌─────────────── Part A: ASR (100% local) ───────────────┐
 │ Silero VAD → split on pauses (<25 s, Whisper window)   │
 │ Whisper medium.en (faster-whisper / sherpa-onnx int8)  │
 │ medical-vocabulary prompt + auditable term corrections │
 └────────────────────────────────────────────────────────┘
        │  outputs/transcript.txt
        ▼
 ┌─────────────── Part B: Text → SOAP ────────────────────┐
 │ LLM (Ollama local default | Anthropic | OpenAI)        │
 │ strict section definitions + routing rules + few-shot  │
 │ JSON-schema output, deterministic decoding             │
 │ rule-based extractor as offline fallback               │
 └────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────── Bonus: validation ──────────────────────┐
 │ 1. Subjective-in-Objective check (regex + LLM-judge),  │
 │    auto-moves leaked items and logs why                │
 │ 2. Grounding check: every number must be in transcript │
 └────────────────────────────────────────────────────────┘
        ▼
 outputs/soap_note.json · outputs/soap_note.md
```

## Deliverables

| File | What it is |
|---|---|
| `notebooks/soap_pipeline.ipynb` | Notebook that loads audio, transcribes, generates and validates the note (executed, outputs included) |
| `outputs/transcript.txt` | Final verbatim transcript |
| `outputs/transcript_raw.txt` | Untouched Whisper output (before term corrections), for comparison |
| `outputs/transcript_segments.json` | Timestamped segments + run metadata |
| `outputs/soap_note.json` / `.md` | Final SOAP note + validation report (`_meta`) |
| `outputs/soap_note_rules_baseline.md` | Output of the no-LLM fallback engine, for comparison |
| `soap_pipeline/` | Source package (modular, commented) |
| `run_pipeline.py` | One-command CLI |

## Setup

Requires **Python 3.10+** and **ffmpeg**.

```bash
git clone https://github.com/vansh-star/Asssignment-TalisManSolution.git
cd Asssignment-TalisManSolution
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# ffmpeg:  Ubuntu: sudo apt install ffmpeg | macOS: brew install ffmpeg | Windows: winget install ffmpeg
```

Pick one LLM for Part B:

```bash
# Option 1 (recommended, fully local / privacy-preserving)
#   install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b

# Option 2 (no GPU): Claude API
export ANTHROPIC_API_KEY=sk-ant-...        # Windows: set ANTHROPIC_API_KEY=...

# Option 3: any OpenAI-compatible endpoint (OpenAI, vLLM, LM Studio)
export OPENAI_API_KEY=...  [OPENAI_BASE_URL=http://localhost:1234/v1]
```

## Run

```bash
# Full pipeline: local Whisper (faster-whisper, model auto-downloads) + local Ollama
python run_pipeline.py data/sample_dictation.mp3

# Same, with the Claude API for the SOAP step
python run_pipeline.py data/sample_dictation.mp3 --llm anthropic

# Alternative ASR backend (Whisper ONNX; models from GitHub, not Hugging Face)
bash scripts/download_sherpa_models.sh
python run_pipeline.py data/sample_dictation.mp3 --asr sherpa-onnx

# Only Part B, reusing the committed transcript
python run_pipeline.py --transcript outputs/transcript.txt --llm ollama

# No LLM at all (deterministic fallback)
python run_pipeline.py --transcript outputs/transcript.txt --engine rules

# Notebook
jupyter notebook notebooks/soap_pipeline.ipynb

# Tests for the validation logic
python -m pytest -q
```

Useful flags: `--asr-model large-v3` (best accuracy, GPU recommended), `--device cuda`, `--llm-model mistral`, `--no-llm-review`.

## How the committed outputs were produced

* **ASR:** Whisper `medium.en`, int8 ONNX via `sherpa-onnx`, Silero VAD, CPU (2 cores): 825 s of audio in ~980 s. (The default `faster-whisper` backend runs the same Whisper weights and is faster; sherpa-onnx was used because the build machine could only fetch models from GitHub.)
* **SOAP:** `claude-sonnet-5-5` via the Anthropic API with the prompt in `soap_pipeline/prompts.py`. Validation passed: 0 Subjective items in Objective, 0 ungrounded numbers.

## Tool selection - rationale

| Choice | Why |
|---|---|
| **Whisper (local)** | Robust to accents/disfluent speech, strong on medical vocabulary for an open model; runs offline so PHI never leaves the machine (HIPAA-friendly). |
| **`medium.en`** | Best accuracy/speed trade-off on CPU. English-only variants are more accurate than multilingual ones at the same size. `large-v3` is a flag away on GPU. |
| **faster-whisper** (default) | CTranslate2: ~4x faster than reference Whisper, int8 on CPU, built-in VAD, supports `initial_prompt` for vocabulary biasing. |
| **sherpa-onnx** (alt) | Same weights on onnxruntime; portable, no PyTorch, models hosted on GitHub. |
| **Silero VAD chunking** | Whisper sees 30 s windows; splitting on silences instead of fixed windows avoids cut words and reduces hallucination on silence. |
| **Deterministic decoding** | `temperature=0`, beam search → verbatim, reproducible transcripts. |
| **Ollama (default LLM)** | Keeps Part B local too; JSON-grammar-constrained output. Cloud APIs are an opt-in for weak hardware. |
| **Rule-based fallback** | Pipeline never hard-fails; also a grounded baseline to compare LLM output against. |

## Prompt engineering (Part B)

`soap_pipeline/prompts.py`:

1. **Precise section definitions** (who is the source of the information: patient vs. clinician).
2. **Explicit routing rules for the classic failure cases**: patient-rated pain score → S; "denies…" → S; test *ordered* → P but test *result* → O; diagnosis mentioned during plan → A; current meds → S, new/changed meds → P.
3. **Grounding rules**: only transcript facts, preserve numbers/doses/laterality, empty list instead of guessing.
4. **Fixed JSON schema** with sub-fields (HPI, PMH, meds, allergies, social/family, ROS; vitals, exam, diagnostics) → machine-parseable, consistent.
5. **Few-shot**: the example from the brief anchors concise clinical register.
6. **Deterministic decoding** + JSON mode where the provider supports it.

## Bonus: Subjective ≠ Objective check

`soap_pipeline/validate.py` runs after generation:

* **Rule layer:** each Objective item is scanned for patient-report cues (`reports`, `states`, `denies`, `complains`, `feels`, self-rated `x/10`, `at home` readings, `history of`). Hits are **moved** to Subjective and logged with the reason.
* **LLM-as-judge layer** (optional, on by default when an LLM is used): the model audits every remaining Objective item to catch paraphrased leaks regexes can't.
* **Grounding check:** any number in the note that is absent from the transcript is reported (hallucinated doses/vitals).

Everything is recorded in `soap_note.json → _meta.validation`. See `tests/test_validate.py`.

## Accuracy notes

The recording is a clinician-patient encounter (lateral epicondylitis case). Whisper handled the medical terms well (*lateral epicondylitis, tennis elbow, C-section, NSAIDs, pronation/supination, palpate, medial/lateral aspect, wrist extensor*). A few mis-hearings were fixed by the small, auditable map in `soap_pipeline/medical_terms.py` (e.g. "paint and swallowing" → "pain and swelling", "Brashes" → "rashes", "Tom is facing" → "palm is facing"); the raw output is kept in `outputs/transcript_raw.txt`. Speaker diarization is not performed (Whisper outputs a single stream) - the LLM infers speaker roles from context.

## Scaling this pipeline

* **ASR throughput:** faster-whisper on GPU (`large-v3`, fp16, batched pipeline) gives ~50-100x real time; queue audio jobs (Celery/RabbitMQ or SQS) with stateless GPU workers autoscaled on queue depth.
* **LLM:** serve an open model with vLLM (continuous batching) behind the OpenAI-compatible client already supported here; cache the system prompt.
* **Quality:** add diarization (pyannote) for speaker-attributed notes, a domain medical vocabulary for `initial_prompt` / hotwords, WER tracking against ground truth, and clinician-in-the-loop review for low-confidence notes.
* **Compliance:** on-prem / VPC deployment, encryption at rest, PHI-free logs, audit trail of every validator correction (already emitted in `_meta`).

## Project structure

```
├── data/sample_dictation.mp3        # provided audio
├── notebooks/soap_pipeline.ipynb    # main notebook
├── outputs/                         # transcript + SOAP note
├── soap_pipeline/
│   ├── audio.py          # ffmpeg decoding
│   ├── transcribe.py     # Part A: faster-whisper / sherpa-onnx backends
│   ├── medical_terms.py  # vocab prompt + term corrections
│   ├── prompts.py        # Part B: system prompt, schema, few-shot, judge prompt
│   ├── llm.py            # Ollama / OpenAI / Anthropic clients
│   ├── soap.py           # orchestration + Markdown rendering
│   ├── rules.py          # deterministic fallback extractor
│   └── validate.py       # Bonus: S/O leakage + grounding checks
├── scripts/download_sherpa_models.sh
├── tests/test_validate.py
├── run_pipeline.py
└── requirements.txt
```
