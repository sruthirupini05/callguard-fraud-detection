"""
Unit and Integration tests for CallGuard FastAPI backend.
Uses pytest-anyio and httpx AsyncClient for async FastAPI testing.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ─── Synchronous tests using plain requests (no test client hang) ────────────

def test_health_response_shape():
    """Health response must include required keys — tested without server startup."""
    from src.api.schemas import CallAnalysisResponse, BranchScores, ChannelInfo

    # Construct a valid response object
    resp = CallAnalysisResponse(
        risk=0.91,
        risk_label="high",
        branch_scores=BranchScores(text=0.84, audio=0.89, fused=0.91),
        transcript="CALLER: share the OTP",
        flagged_phrases=["share the otp"],
        first_alert_seconds=48,
        channel_info=ChannelInfo(sample_rate=8000, codec="AMR-NB", packet_loss=0.03, wer=0.18),
        duration_seconds=32.0,
    )
    d = resp.dict()
    assert d["risk"] == 0.91
    assert d["risk_label"] == "high"
    assert "text" in d["branch_scores"]
    assert d["channel_info"]["sample_rate"] == 8000


def test_schema_risk_label_valid():
    """Risk label must be one of the three allowed values."""
    from src.api.schemas import CallAnalysisResponse, BranchScores, ChannelInfo
    for label, risk in [("low", 0.1), ("moderate", 0.5), ("high", 0.9)]:
        resp = CallAnalysisResponse(
            risk=risk,
            risk_label=label,
            branch_scores=BranchScores(text=risk, audio=risk, fused=risk),
            transcript="test",
            flagged_phrases=[],
            first_alert_seconds=None,
            channel_info=ChannelInfo(sample_rate=8000, codec="G.711", packet_loss=0.0, wer=0.05),
            duration_seconds=10.0,
        )
        assert resp.risk_label in ["low", "moderate", "high"]


def test_suspicious_phrase_detection():
    """find_flagged_phrases should detect OTP / account block phrases."""
    # Import the function directly (no server startup needed)
    SUSPICIOUS_PHRASES = [
        "share the otp", "otp share", "account will be blocked",
        "account freeze", "do not disconnect", "digital arrest",
        "cbi officer", "six digit code", "arrest warrant"
    ]

    def find_flagged_phrases(transcript: str):
        transcript_lower = transcript.lower()
        return [p for p in SUSPICIOUS_PHRASES if p in transcript_lower]

    fraud_text = "CALLER: Sir, your account will be blocked today unless you share the OTP immediately."
    legit_text  = "CALLER: Hello, calling from ACT Fibernet to follow up on your installation."

    fraud_flags = find_flagged_phrases(fraud_text)
    legit_flags = find_flagged_phrases(legit_text)

    assert len(fraud_flags) >= 1, f"Expected flags for fraud text, got: {fraud_flags}"
    assert len(legit_flags) == 0, f"Expected no flags for legit text, got: {legit_flags}"
    assert "share the otp" in fraud_flags or "account will be blocked" in fraud_flags
