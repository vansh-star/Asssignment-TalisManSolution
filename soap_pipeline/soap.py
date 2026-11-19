"""Part B - orchestrates Text -> SOAP generation, validation and rendering."""
from __future__ import annotations

import json
import time

from .llm import LLMClient
from .prompts import FEW_SHOT_INPUT, FEW_SHOT_OUTPUT, SOAP_SCHEMA, SYSTEM_PROMPT
from .rules import _empty_note, extract_soap_rules
from .validate import check_grounding, check_subjective_leakage


def _normalise(note: dict) -> dict:
    """Coerce any model output onto the fixed schema (missing keys -> empty)."""
    base = _empty_note()
    for sec, val in base.items():
        got = note.get(sec, val)
        if isinstance(val, dict):
            base[sec] = {k: got.get(k, v) if isinstance(got, dict) else v for k, v in val.items()}
        else:
            base[sec] = got if isinstance(got, list) else ([got] if got else [])
    return base


def generate_soap_llm(transcript: str, client: LLMClient) -> dict:
    system = SYSTEM_PROMPT.format(schema=json.dumps(SOAP_SCHEMA, indent=2))
    messages = [
        {"role": "user", "content": f"TRANSCRIPT:\n{FEW_SHOT_INPUT}"},
        {"role": "assistant", "content": json.dumps(FEW_SHOT_OUTPUT)},
        {"role": "user", "content": f"TRANSCRIPT:\n{transcript}"},
    ]
    return _normalise(client.chat_json(system, messages))


def generate_soap(transcript: str, engine: str = "llm", provider: str = "ollama",
                  model: str | None = None, llm_review: bool = True) -> dict:
    """Return a validated SOAP note dict.

    engine="llm"   -> LLM (falls back to rules if the LLM is unreachable)
    engine="rules" -> deterministic extractor only
    """
    t0 = time.time()
    client, used = None, "rules"
    if engine == "llm":
        client = LLMClient(provider, model)
        try:
            note = generate_soap_llm(transcript, client)
            used = f"llm:{client.provider}:{client.model}"
        except Exception as e:
            print(f"[soap] LLM unavailable ({e}); falling back to rule-based engine.")
            client, note = None, extract_soap_rules(transcript)
    else:
        note = extract_soap_rules(transcript)

    leaks = check_subjective_leakage(note, llm=client if llm_review else None)
    ungrounded = check_grounding(note, transcript)
    note["_meta"] = {
        "engine": used,
        "elapsed_seconds": round(time.time() - t0, 1),
        "validation": {
            "subjective_in_objective_moved": leaks,
            "ungrounded_numbers": ungrounded,
            "passed": not ungrounded,
        },
    }
    return note


def to_markdown(note: dict) -> str:
    def bullets(items):
        return "\n".join(f"- {x}" for x in items) if items else "- _Not documented_"

    p, s, o = note["patient"], note["subjective"], note["objective"]
    demo = ", ".join(f"{k}: {v}" for k, v in p.items() if v) or "Not stated"
    out = ["# SOAP Note", "", f"**Patient:** {demo}", "",
           "## S - Subjective", f"**Chief complaint:** {s['chief_complaint'] or '_Not documented_'}", ""]
    for key, title in [("history_of_present_illness", "History of present illness"),
                       ("past_medical_history", "Past medical / surgical history"),
                       ("medications", "Medications"), ("allergies", "Allergies"),
                       ("social_family_history", "Social / family history"),
                       ("review_of_systems", "Review of systems")]:
        out += [f"**{title}**", bullets(s[key]), ""]
    out += ["## O - Objective"]
    for key, title in [("vital_signs", "Vital signs"), ("physical_exam", "Physical exam"),
                       ("diagnostics", "Labs / imaging")]:
        out += [f"**{title}**", bullets(o[key]), ""]
    out += ["## A - Assessment", bullets(note["assessment"]), "",
            "## P - Plan", bullets(note["plan"]), ""]
    meta = note.get("_meta", {})
    if meta:
        v = meta["validation"]
        out += ["---", f"_Engine: {meta['engine']} | Subjective items moved out of Objective: "
                f"{len([x for x in v['subjective_in_objective_moved'] if 'item' in x])} | "
                f"Ungrounded numbers: {v['ungrounded_numbers'] or 'none'}_"]
    return "\n".join(out)
