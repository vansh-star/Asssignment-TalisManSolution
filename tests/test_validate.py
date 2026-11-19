"""Unit tests for the Subjective/Objective logic check (run: python -m pytest -q)."""
from soap_pipeline.rules import _empty_note
from soap_pipeline.validate import check_grounding, check_subjective_leakage


def test_patient_report_is_moved_out_of_objective():
    note = _empty_note()
    note["objective"]["physical_exam"] = [
        "Tenderness on palpation of lateral elbow",
        "Patient reports pain is 4/10",
        "Denies numbness or tingling",
    ]
    findings = check_subjective_leakage(note)
    assert note["objective"]["physical_exam"] == ["Tenderness on palpation of lateral elbow"]
    assert "Patient reports pain is 4/10" in note["subjective"]["history_of_present_illness"]
    assert len(findings) == 2


def test_grounding_flags_invented_numbers():
    note = _empty_note()
    note["plan"] = ["Ibuprofen 400 mg BID", "Follow up in 2 weeks"]
    assert check_grounding(note, "take ibuprofen and follow up in two weeks") == ["400"]
