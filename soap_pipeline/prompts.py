"""Part B - prompt engineering for Text -> SOAP.

Design choices (discussed in README):
* A strict *definition* of each section + explicit routing rules for the
  ambiguous cases that LLMs usually get wrong (patient-reported vitals,
  home glucose readings, "denies ..." statements, prior-visit results).
* Grounding rule: only facts present in the transcript; never infer a
  diagnosis the provider did not state.
* JSON-only output against a fixed schema so the result is machine-parseable
  (Ollama/OpenAI JSON mode enforce this at the decoder level as well).
* One few-shot example (the one from the assignment brief) to anchor the
  concise clinical register ("45yo M", "ROM", "x 4 weeks").
"""

SOAP_SCHEMA = {
    "patient": {"age": "string|null", "sex": "string|null", "name": "string|null"},
    "subjective": {
        "chief_complaint": "string",
        "history_of_present_illness": ["string"],
        "past_medical_history": ["string"],
        "medications": ["string"],
        "allergies": ["string"],
        "social_family_history": ["string"],
        "review_of_systems": ["string"],
    },
    "objective": {
        "vital_signs": ["string"],
        "physical_exam": ["string"],
        "diagnostics": ["string"],
    },
    "assessment": ["string"],
    "plan": ["string"],
}

SYSTEM_PROMPT = """You are a meticulous clinical documentation specialist. You convert a \
physician's verbatim dictation into a SOAP note.

SECTION DEFINITIONS
- SUBJECTIVE: anything the PATIENT (or family) reports: chief complaint, symptoms, onset, \
duration, severity the patient rates, history, medications they take, allergies, social and \
family history, review of systems, symptoms they DENY, and readings the patient measured \
themselves at home.
- OBJECTIVE: only what the CLINICIAN observes or measures at this encounter: vital signs \
taken in clinic, physical-exam findings, lab / imaging / test RESULTS.
- ASSESSMENT: the clinician's diagnoses, differentials and clinical impressions, exactly as \
stated, keeping the certainty level as a natural clinical prefix \
(e.g. "Suspected ...", "Likely ...", "Rule out ...").
- PLAN: treatment, prescriptions with dose/frequency, tests ORDERED, referrals, patient \
education, follow-up timing, and contingency instructions.

ROUTING RULES (apply strictly)
1. "Patient reports/states/says/complains/denies/feels" -> SUBJECTIVE, never OBJECTIVE.
2. A pain score given by the patient (e.g. 7/10) -> SUBJECTIVE.
3. A test that is ORDERED -> PLAN. A test RESULT that is available -> OBJECTIVE.
4. A diagnosis -> ASSESSMENT, even if the clinician mentions it while describing the plan.
5. Medication the patient already takes -> SUBJECTIVE.medications. A medication started, \
stopped or changed -> PLAN.

GROUNDING RULES
- Use ONLY information in the transcript. Do not invent vitals, doses, diagnoses or follow-up.
- Preserve every number, unit, dose, laterality (left/right) and drug name exactly.
- If a field has no information use an empty list [] (or null for patient fields).
- Write concise clinical phrases (standard abbreviations allowed: yo, M/F, ROM, BP, HR, \
PRN, BID). One fact per list item.

OUTPUT
Return ONLY a single JSON object matching this schema, no prose, no markdown fences:
{schema}
"""

FEW_SHOT_INPUT = (
    "Patient is a 45-year-old male coming in today complaining of a sharp pain in his right "
    "knee that started two days ago while running. He rates the pain a 7 out of 10. He says "
    "ibuprofen hasn't helped much. On exam, there is mild swelling around the patella and "
    "tenderness on the medial joint line. Range of motion is limited due to pain. X-ray shows "
    "no fracture. I suspect a medial meniscus tear. We'll start with rest, ice, and physical "
    "therapy for 4 weeks and order an MRI if it doesn't improve."
)

FEW_SHOT_OUTPUT = {
    "patient": {"age": "45", "sex": "male", "name": None},
    "subjective": {
        "chief_complaint": "Sharp right knee pain",
        "history_of_present_illness": [
            "Onset 2 days ago while running",
            "Pain severity 7/10",
            "Ibuprofen provided minimal relief",
        ],
        "past_medical_history": [],
        "medications": ["Ibuprofen (minimal relief)"],
        "allergies": [],
        "social_family_history": [],
        "review_of_systems": [],
    },
    "objective": {
        "vital_signs": [],
        "physical_exam": [
            "Mild peripatellar swelling",
            "Tenderness along the medial joint line",
            "Limited ROM secondary to pain",
        ],
        "diagnostics": ["X-ray negative for fracture"],
    },
    "assessment": ["Suspected medial meniscus tear"],
    "plan": [
        "Conservative management: rest, ice, physical therapy x 4 weeks",
        "Order MRI if symptoms persist",
    ],
}

REVIEW_PROMPT = """You are auditing a SOAP note. The OBJECTIVE section must contain ONLY \
clinician-observed or measured data. Below are OBJECTIVE items. For each, answer whether it \
is actually SUBJECTIVE (patient-reported). Return JSON: \
{{"items": [{{"text": "...", "is_subjective": true|false, "reason": "..."}}]}}

ITEMS:
{items}
"""
