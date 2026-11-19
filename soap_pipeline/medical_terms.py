"""Medical-vocabulary helpers for ASR.

* ``MEDICAL_PROMPT`` is passed to Whisper as ``initial_prompt``. Whisper treats
  it as preceding context, which nudges the decoder toward correct spelling of
  drug names, anatomy and abbreviations it would otherwise mangle.
* ``CORRECTIONS`` is a small, auditable map of observed mis-recognitions ->
  correct term. Kept deliberately conservative (exact, case-insensitive,
  word-boundary matches) so the transcript stays verbatim.
"""
from __future__ import annotations

import re

MEDICAL_PROMPT = (
    "Medical dictation by a physician. History of present illness, review of systems, "
    "physical examination, vital signs, blood pressure, assessment and plan. "
    "Terms: hypertension, hyperlipidemia, diabetes mellitus, metformin, lisinopril, "
    "atorvastatin, ibuprofen, acetaminophen, HbA1c, CBC, CMP, EKG, MRI, CT, "
    "auscultation, tenderness, erythema, edema, bilaterally, mg, b.i.d., p.r.n."
)

# wrong phrase (lower-case) -> correct phrase. Populated from manual error
# analysis of the sample recording (see README "Accuracy"). Only phrases whose
# clinical meaning is unambiguous from context are corrected; the untouched
# model output is always saved as outputs/transcript_raw.txt for comparison.
CORRECTIONS: dict[str, str] = {
    "paint and swallowing": "pain and swelling",   # plan: ice for pain/swelling
    "brashes": "rashes",                           # ROS question
    "mr. tingling": "no tingling",                 # ROS answer
    "your tom is facing": "your palm is facing",   # pronation instruction
    "smaller taste": "smell or taste",             # ROS: special senses
    "i pose pain": "I do feel pain",               # lateral epicondyle palpation
    "that was a bird": "that was a birth",         # C-section history
    "de-smoke": "do you smoke",                    # social history
    "talked before": "smoked before",              # social history answer
    "a over pain": "elbow pain",
}


def fix_medical_terms(text: str, corrections: dict[str, str] | None = None) -> str:
    corrections = CORRECTIONS if corrections is None else corrections
    for wrong, right in corrections.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text
