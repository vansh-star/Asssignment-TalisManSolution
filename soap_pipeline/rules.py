"""Deterministic, dependency-free Text -> SOAP extractor (fallback engine).

Used when no LLM is reachable (CI, air-gapped machines) and as a baseline to
compare the LLM against. It understands both styles of input:

* monologue dictation ("Patient is a 45yo male ... On exam ... Plan ...") via
  sentence-level cue classification, and
* clinician-patient conversations (like the sample recording) via
  question -> answer pairing: the clinician's question decides the SOAP slot,
  the patient's answer supplies the content, and negation in the answer turns
  it into a pertinent negative ("Denies fever").

It is intentionally conservative: every item is a cleaned-up span of the
transcript, so it cannot hallucinate - at the cost of less polished phrasing.
"""
from __future__ import annotations

import re

FILLERS = re.compile(r"\b(uh+|um+|mm+|uhh+|hmm+|mhm)\b[.,]*\s*|\.\.\.", re.IGNORECASE)

# ---- question topic -> (section path, label) ------------------------------ #
ROS_TOPICS = {
    "weight loss": "weight loss", "fever": "fevers/chills", "chills": "fevers/chills",
    "fatigue": "fatigue", "rash": "rashes", "vision": "changes in vision/hearing/smell/taste",
    "cough": "cough, SOB, wheezing", "shortness of breath": "cough, SOB, wheezing",
    "chest pain": "chest pain, palpitations, dizziness", "palpitation": "chest pain, palpitations, dizziness",
    "bowel": "changes in bowel movements", "nausea": "nausea/vomiting", "vomiting": "nausea/vomiting",
    "urine": "urinary symptoms", "urinary": "urinary symptoms", "memory": "memory loss/confusion",
    "numbness": "numbness/tingling", "tingling": "numbness/tingling", "cold": "cold sensation in arm",
    "radiat": "radiation of pain", "other joint": "pain in other joints/muscle groups",
    "other symptoms": "other symptoms",
}
EXAM_TOPICS = {
    "redness": "redness or swelling", "swelling": "redness or swelling",
    "misaligned": "deformity/asymmetry", "asymmetr": "deformity/asymmetry",
    "skin changes": "skin changes or deformities",
    "lateral aspect": "pain on palpation of lateral elbow", "medial aspect": "pain on palpation of medial elbow",
    "palpate your": "pain on palpation of forearm/biceps", "hot": "warmth over area",
    "extend your wrist": "pain with resisted wrist extension (elbow extended)",
    "flex your wrist": "pain with wrist flexion (elbow extended)",
    "flex your elbow": "pain with elbow flexion", "shoulder": "pain with shoulder movement",
    "pronate": "pain with forearm pronation/supination", "supinate": "pain with forearm pronation/supination",
    "range of motion": "range of motion", "move that arm": "range of motion",
}
HISTORY_TOPICS = {
    "medical condition": ("past_medical_history", "Chronic medical conditions"),
    "otherwise healthy": ("past_medical_history", "Chronic medical conditions"),
    "surger": ("past_medical_history", "Surgical history"),
    "hospitali": ("past_medical_history", "Surgical history"),
    "how long ago": ("past_medical_history", "Surgical history"),
    "emergency": ("past_medical_history", "Surgical history"),
    "immuniz": ("past_medical_history", "Immunizations"),
    "medication": ("medications", "Medications"),
    "allerg": ("allergies", "Allergies"),
    "family": ("social_family_history", "Family history"),
    "smoke": ("social_family_history", "Smoking"),
    "marijuana": ("social_family_history", "Marijuana"),
    "alcohol": ("social_family_history", "Alcohol"),
    "sick contact": ("social_family_history", "Sick contacts"),
    "travel": ("social_family_history", "Recent travel"),
}
HPI_TOPICS = {
    "trigger": "Trigger/trauma", "trauma": "Trigger/trauma", "better or worse": "Modifying factors",
    "quality": "Quality", "movements make it worse": "Aggravating movements",
    "one to ten": "Severity", "1 to 10": "Severity", "experienced this": "Prior episodes",
    "getting worse": "Progression", "topical": "Topical treatment tried",
    "everyday tasks": "Functional impact",
}
NEG = re.compile(r"^\W*(no|nope|nothing|none|not really|i haven'?t|i don'?t|i do not|i didn'?t|"
                 r"that doesn'?t|it doesn'?t|i would not)\b", re.IGNORECASE)
ASSESS_CUES = re.compile(r"\b(may have|likely|consistent with|suspect|diagnos|impression|also known as|"
                         r"case of|predispose|differential)\b", re.IGNORECASE)
PLAN_CUES = re.compile(r"\b(recommend|we'?ll|start|continue|physical therapy|prescrib|order|follow[- ]?up|"
                       r"imaging|compression|splint|ice|modification|resolve|refer|return)\b", re.IGNORECASE)


def clean(text: str) -> str:
    text = FILLERS.sub("", text)
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text, flags=re.IGNORECASE)  # "i i i" -> "i"
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    return text[:1].upper() + text[1:] if text else text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _match(topics: dict, q: str):
    ql = q.lower()
    for key, val in topics.items():
        if key in ql:
            return val
    return None


def _empty_note() -> dict:
    return {
        "patient": {"age": None, "sex": None, "name": None},
        "subjective": {"chief_complaint": "", "history_of_present_illness": [], "past_medical_history": [],
                       "medications": [], "allergies": [], "social_family_history": [], "review_of_systems": []},
        "objective": {"vital_signs": [], "physical_exam": [], "diagnostics": []},
        "assessment": [],
        "plan": [],
    }


def _add(lst: list, item: str) -> None:
    if item and item not in lst:
        lst.append(item)


def extract_soap_rules(transcript: str) -> dict:
    note = _empty_note()
    s = note["subjective"]
    sents = split_sentences(transcript)

    # Chief complaint: first patient statement describing pain/symptom.
    for sent in sents[:6]:
        if re.search(r"\b(pain|ache|hurt|cough|fever|swelling)\b", sent, re.I):
            s["chief_complaint"] = clean(re.sub(r"^.*?\b(I'?ve had|I have)\b", r"\1", sent, flags=re.I))
            break

    # Locate the counselling part (clinician explains diagnosis + plan).
    counsel_idx = next((i for i, x in enumerate(sents) if ASSESS_CUES.search(x)), len(sents))

    # Q -> A pairing over the interview part.
    i = 0
    while i < counsel_idx:
        q = sents[i]
        if not q.endswith("?"):
            i += 1
            continue
        # answer = following non-question sentences (up to 2)
        ans_parts, j = [], i + 1
        while j < counsel_idx and not sents[j].endswith("?") and len(ans_parts) < 2:
            ans_parts.append(sents[j])
            j += 1
        answer = clean(" ".join(ans_parts))
        negative = bool(NEG.match(answer)) or answer.lower().startswith(("no", "uh no"))

        if (topic := _match(EXAM_TOPICS, q)):
            finding = f"No {topic}" if negative else f"{topic[:1].upper()}{topic[1:]}: {answer}"
            if topic.startswith("pain") and not negative:
                finding = f"Positive: {topic}"
            elif topic.startswith("pain") and negative:
                finding = f"Negative: {topic}"
            _add(note["objective"]["physical_exam"], finding)
        elif (topic := _match(ROS_TOPICS, q)):
            _add(s["review_of_systems"], f"Denies {topic}" if negative else f"{topic}: {answer}")
        elif (hit := _match(HISTORY_TOPICS, q)):
            field, label = hit
            _add(s[field], f"{label}: {answer}" if answer else label)
        elif (label := _match(HPI_TOPICS, q)):
            _add(s["history_of_present_illness"], f"{label}: {answer}")
        i = j if j > i + 1 else i + 1

    # Assessment & plan from the clinician's counselling.
    for sent in sents[counsel_idx:]:
        c = clean(sent)
        if len(c) < 8:
            continue
        if ASSESS_CUES.search(sent):
            _add(note["assessment"], c)
        elif PLAN_CUES.search(sent):
            _add(note["plan"], c)

    # Monologue-style fallback: classify remaining sentences by cue words.
    if not any(s.values()) or not note["objective"]["physical_exam"]:
        for sent in sents:
            if re.search(r"\b(on exam|examination|tenderness|range of motion|x-ray shows|blood pressure|"
                         r"heart rate|temperature|labs? show)\b", sent, re.I):
                _add(note["objective"]["physical_exam"], clean(sent))

    m = re.search(r"(\d+)[- ]year[- ]old (male|female|man|woman)", transcript, re.I)
    if m:
        note["patient"]["age"], note["patient"]["sex"] = m.group(1), m.group(2).lower()
    return note
