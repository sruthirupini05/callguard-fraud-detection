"""
Unit and Integration tests for CallGuard FastAPI backend.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_health_response_shape():
    """Health response must include required keys — tested without server startup."""
    from src.api.schemas import CallAnalysisResponse, BranchScores, ChannelInfo

    resp = CallAnalysisResponse(
        risk=0.91,
        risk_label="high",
        verdict="HIGH RISK",
        branch_scores=BranchScores(text=0.84, audio=0.89, fused=0.91),
        transcript="CALLER: share the OTP",
        flagged_phrases=["share the otp"],
        first_alert_seconds=48,
        channel_info=ChannelInfo(sample_rate=8000, codec="AMR-NB", packet_loss=0.03, wer=0.18),
        duration_seconds=32.0,
    )
    d = resp.model_dump()
    assert d["risk"] == 0.91
    assert d["risk_label"] == "high"
    assert d["verdict"] == "HIGH RISK"
    assert "text" in d["branch_scores"]
    assert d["channel_info"]["sample_rate"] == 8000


def test_schema_risk_label_valid():
    """Risk label must be one of the three allowed values."""
    from src.api.schemas import CallAnalysisResponse, BranchScores, ChannelInfo
    for label, risk in [("low", 0.1), ("moderate", 0.5), ("high", 0.9)]:
        resp = CallAnalysisResponse(
            risk=risk,
            risk_label=label,
            verdict="HIGH RISK" if risk >= 0.6 else "LOW RISK",
            branch_scores=BranchScores(text=risk, audio=risk, fused=risk),
            transcript="test",
            flagged_phrases=[],
            first_alert_seconds=None,
            channel_info=ChannelInfo(sample_rate=8000, codec="G.711", packet_loss=0.0, wer=0.05),
            duration_seconds=10.0,
        )
        assert resp.risk_label in ["low", "moderate", "high"]


def test_intent_classification_scam_call():
    """Intent classifier must flag credential request, urgency, and coercive threat on scam calls."""
    from src.models.intent_classifier import analyze_conversation_intents

    scam_text = (
        "CALLER: Sir, this is from your bank fraud cell. Your account will be blocked today "
        "unless you share the OTP immediately. Do not disconnect the call."
    )
    analysis = analyze_conversation_intents(scam_text)
    assert not analysis["is_anti_warning"]
    assert analysis["intents"]["credential_request"] > 0.5
    assert analysis["intents"]["urgency"] > 0.5
    assert analysis["intents"]["coercive_threat"] > 0.5
    assert analysis["scam_taxonomy"] == "Bank OTP / KYC Suspension Fraud"


def test_anti_warning_logic_on_advisory_call():
    """Anti-warning logic must detect educational bank notices (L04) and suppress false positives."""
    from src.models.intent_classifier import analyze_conversation_intents

    advisory_text = (
        "CALLER: Good morning. This is a security advisory from SBI. Please remember that "
        "the bank will NEVER ask for your OTP, PIN, or password over the phone. Never share your OTP with anyone."
    )
    analysis = analyze_conversation_intents(advisory_text)
    assert analysis["is_anti_warning"] is True
    assert len(analysis["warning_cues"]) > 0
    # Composite intent score must be strongly suppressed
    assert analysis["composite_intent_score"] < 0.20
    assert analysis["scam_taxonomy"] == "Legitimate Bank Advisory / Notice"


def test_calibration_and_uncertainty():
    """Calibrator must output probability, confidence intervals, and 3-tier verdict."""
    from src.models.calibration import ProbabilityCalibrator

    cal = ProbabilityCalibrator(temperature=1.25)
    
    # Low risk
    low = cal.compute_uncertainty(0.15)
    assert low["verdict"] == "LOW RISK"
    assert low["lower_bound"] <= 0.15 <= low["upper_bound"]

    # Uncertain range (40%-60%)
    ambig = cal.compute_uncertainty(0.52)
    assert ambig["verdict"] == "UNCERTAIN"
    assert ambig["is_uncertain"] is True

    # High risk
    high = cal.compute_uncertainty(0.92)
    assert high["verdict"] == "HIGH RISK"


def test_speaker_diarization():
    """Diarization must parse CALLER vs USER turns accurately."""
    from src.models.intent_classifier import parse_dialogue_turns

    dialogue = (
        "CALLER: Please read the 6-digit OTP.\n"
        "YOU: No, I will not share my OTP.\n"
        "CALLER: Sir, account block will happen immediately."
    )
    turns = parse_dialogue_turns(dialogue)
    assert len(turns["caller"]) == 2
    assert len(turns["user"]) == 1
    assert "Please read the 6-digit OTP." in turns["caller"]
    assert "No, I will not share my OTP." in turns["user"]
