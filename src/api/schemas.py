"""
Pydantic API request and response schemas for CallGuard.
Updated with:
- Calibrated probability & dynamic uncertainty intervals
- Multimodal Gate Weight trust distribution
- Semantic Intent breakdown & Scam Taxonomy
- Caller metadata and CLI spoofing diagnostics
- Feedback / False Positive reporting schema
"""

from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field


class BranchScores(BaseModel):
    text: float = Field(..., description="BiLSTM text branch fraud risk (0.0 to 1.0)")
    audio: float = Field(..., description="Acoustic branch fraud risk (0.0 to 1.0)")
    fused: float = Field(..., description="Multimodal fused fraud risk (0.0 to 1.0)")


class GateTrust(BaseModel):
    text_trust: float = Field(0.65, description="Learned gate weight assigned to text representation")
    audio_trust: float = Field(0.35, description="Learned gate weight assigned to acoustic representation")


class UncertaintyRange(BaseModel):
    lower: float = Field(..., description="Lower bound of confidence interval")
    upper: float = Field(..., description="Upper bound of confidence interval")
    margin: float = Field(..., description="Estimated uncertainty margin (half-width)")


class CallerMetadata(BaseModel):
    phone_number: str = Field("+91 98XXX XXX21", description="Reported or simulated caller number")
    spoof_risk: str = Field("Moderate", description="Heuristic CLI spoofing risk: 'Low', 'Moderate', 'High'")
    community_reports: int = Field(0, description="Crowdsourced scam report count in database")
    carrier_type: str = Field("VoIP Gateway / Cellular", description="Identified routing telecom carrier")


class ChannelInfo(BaseModel):
    sample_rate: int = Field(8000, description="Audio sampling rate in Hz")
    codec: str = Field("AMR-NB", description="Detected or simulated telephone codec")
    packet_loss: float = Field(0.03, description="VoIP/cellular frame drop rate")
    wer: Optional[float] = Field(None, description="Dynamic Word Error Rate against reference")


class CallAnalysisResponse(BaseModel):
    risk: float = Field(..., description="Calibrated fused fraud probability (0.0 to 1.0)")
    risk_label: str = Field(..., description="Verdict category: 'low', 'moderate', 'high'")
    verdict: str = Field("LOW RISK", description="Verdict: 'LOW RISK', 'UNCERTAIN', 'HIGH RISK'")
    uncertainty: UncertaintyRange = Field(default_factory=lambda: UncertaintyRange(lower=0.1, upper=0.2, margin=0.05))
    branch_scores: BranchScores
    gate_trust: GateTrust = Field(default_factory=lambda: GateTrust(text_trust=0.65, audio_trust=0.35))
    scam_type: str = Field("Social Engineering Threat", description="Scam taxonomy classification")
    intents: Dict[str, float] = Field(default_factory=dict, description="Semantic intent dimension scores")
    anti_warning_detected: bool = Field(False, description="True if call contains legitimate defensive advisory context")
    caller_metadata: CallerMetadata = Field(default_factory=CallerMetadata)
    transcript: str = Field(..., description="Transcribed conversation text")
    flagged_phrases: List[str] = Field(default_factory=list, description="Suspicious trigger keywords identified")
    first_alert_seconds: Optional[int] = Field(None, description="Earliest second fraud was reliably flagged")
    channel_info: ChannelInfo
    duration_seconds: float = Field(..., description="Total audio length analyzed")


class StreamingChunkMessage(BaseModel):
    event: str = Field("chunk_analysis", description="Streaming message type")
    current_second: int
    partial_transcript: str
    fused_risk: float
    uncertainty: Dict[str, float] = Field(default_factory=lambda: {"lower": 0.0, "upper": 0.0})
    verdict: str = Field("LOW RISK")
    gate_trust: Dict[str, float] = Field(default_factory=lambda: {"text_trust": 0.65, "audio_trust": 0.35})
    scam_type: str = Field("Analyzing...")
    intents: Dict[str, float] = Field(default_factory=dict)
    is_alert_active: bool
    flagged_phrases: List[str]
    trajectory: List[Dict[str, Any]] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    call_id: str
    feedback_type: str = Field(..., description="'false_positive' or 'confirmed_fraud'")
    notes: Optional[str] = None
    transcript_snippet: Optional[str] = None
    reported_risk: Optional[float] = None


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    call_id: str
    message: str
