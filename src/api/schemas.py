"""
Pydantic API request and response schemas for CallGuard.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class BranchScores(BaseModel):
    text: float = Field(..., description="BiLSTM text branch fraud risk (0.0 to 1.0)")
    audio: float = Field(..., description="CNN-BiGRU acoustic branch fraud risk (0.0 to 1.0)")
    fused: float = Field(..., description="Multimodal fused fraud risk (0.0 to 1.0)")


class ChannelInfo(BaseModel):
    sample_rate: int = Field(8000, description="Audio sampling rate in Hz")
    codec: str = Field("G711", description="Detected or simulated telephone codec")
    packet_loss: float = Field(0.03, description="VoIP/cellular frame drop rate")
    wer: Optional[float] = Field(None, description="Estimated Word Error Rate")


class CallAnalysisResponse(BaseModel):
    risk: float = Field(..., description="Fused fraud probability")
    risk_label: str = Field(..., description="Risk tier: 'low', 'moderate', 'high'")
    branch_scores: BranchScores
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
    is_alert_active: bool
    flagged_phrases: List[str]
