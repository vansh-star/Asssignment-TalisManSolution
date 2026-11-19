"""Bonus - logic checks on the generated SOAP note.

1. ``check_subjective_leakage``: no patient-reported content in OBJECTIVE.
   Each Objective item is scanned for patient-report cues ("reports", "states",
   "denies", "complains", "feels", a self-rated pain score, home readings ...).
   Offending items are MOVED to Subjective.HPI and logged, so the final note is
   corrected automatically, and the reason is kept for audit.
   An optional second pass asks the LLM to judge each Objective item
   (LLM-as-judge) to catch paraphrased leaks a regex cannot see.

2. ``check_grounding``: every number in the note (doses, vitals, durations)
   must appear in the transcript -> catches hallucinated values.
"""
from __future__ import annotations

import json
import re

from .prompts import REVIEW_PROMPT

SUBJECTIVE_CUES = [
    r"\b(patient|pt|he|she|they)\s+(reports?|states?|says?|said|describes?|endorses?|denies|denied|"
    r"complains?|feels?|felt|notes?|noticed|mentions?|admits?|believes?|thinks?)\b",
    r"\b(reports?|reported|complains? of|c/o|endorses|denies|per patient|according to the patient)\b",
    r"\b(feels?|feeling)\b",
    r"\brates?\b.*\b\d+\s*(/|out of)\s*10\b",
    r"\b(at home|home (blood pressure|bp|glucose|readings?|monitoring)|self[- ]reported|patient[- ]reported)\b",
    r"\b(history of|hx of)\b",
]
_SUBJ_RE = [re.compile(p, re.IGNORECASE) for p in SUBJECTIVE_CUES]


def _subjective_reason(item: str) -> str | None:
    for rx in _SUBJ_RE:
        m = rx.search(item)
        if m:
            return f"patient-report cue: '{m.group(0)}'"
    return None


def check_subjective_leakage(note: dict, llm=None) -> list[dict]:
    """Move subjective items out of OBJECTIVE. Mutates ``note``; returns findings."""
    findings: list[dict] = []
    objective = note.get("objective", {})
    hpi = note.setdefault("subjective", {}).setdefault("history_of_present_illness", [])

    for key, items in objective.items():
        keep = []
        for item in items:
            reason = _subjective_reason(item)
            if reason:
                findings.append({"item": item, "from": f"objective.{key}",
                                 "to": "subjective.history_of_present_illness",
                                 "reason": reason, "checker": "rules"})
                hpi.append(item)
            else:
                keep.append(item)
        objective[key] = keep

    if llm is not None:
        flat = [(k, it) for k, its in objective.items() for it in its]
        if flat:
            listing = "\n".join(f"- {it}" for _, it in flat)
            try:
                verdict = llm.chat_json("You are a strict clinical documentation auditor.",
                                        [{"role": "user", "content": REVIEW_PROMPT.format(items=listing)}])
                flagged = {v["text"].strip(): v.get("reason", "")
                           for v in verdict.get("items", []) if v.get("is_subjective")}
            except Exception as e:  # judge is best-effort; never break the pipeline
                flagged = {}
                findings.append({"warning": f"LLM review skipped: {e}"})
            for key, item in flat:
                if item.strip() in flagged:
                    objective[key].remove(item)
                    hpi.append(item)
                    findings.append({"item": item, "from": f"objective.{key}",
                                     "to": "subjective.history_of_present_illness",
                                     "reason": flagged[item.strip()], "checker": "llm"})
    return findings


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_NUMS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
              "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twelve": "12",
              "fifteen": "15", "twenty": "20", "thirty": "30"}


def check_grounding(note: dict, transcript: str) -> list[str]:
    """Return numbers that appear in the note but nowhere in the transcript."""
    src = transcript.lower()
    for w, d in _WORD_NUMS.items():
        src = re.sub(rf"\b{w}\b", d, src)
    src_nums = set(_NUM_RE.findall(src.replace(",", "")))
    note_txt = json.dumps({k: v for k, v in note.items() if k != "_meta"})
    missing = sorted({n for n in _NUM_RE.findall(note_txt) if n not in src_nums and n not in {"0", "10"}})
    return missing
