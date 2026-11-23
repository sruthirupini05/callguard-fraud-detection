// API client & types for CallGuard Research Workstation

export interface BranchScores {
  text: number;
  audio: number;
  fused: number;
}

export interface GateTrust {
  text_trust: number;
  audio_trust: number;
}

export interface UncertaintyRange {
  lower: number;
  upper: number;
  margin: number;
}

export interface CallerMetadata {
  phone_number: string;
  spoof_risk: string;
  community_reports: number;
  carrier_type: string;
}

export interface ChannelInfo {
  sample_rate: number;
  codec: string;
  packet_loss: number;
  wer: number | null;
}

export interface AudioTelemetry {
  original_sample_rate: number;
  original_channels: number;
  duration_seconds: number;
  speech_ratio: number;
  snr_db: number;
  noise_floor_db: number;
  clipping_ratio: number;
  dynamic_range_db: number;
  waveform_envelope: number[];
  vad_segments_count: number;
}

export interface AnalysisResult {
  risk: number;
  risk_label: 'low' | 'moderate' | 'high';
  verdict: 'LOW RISK' | 'UNCERTAIN' | 'HIGH RISK';
  uncertainty: UncertaintyRange;
  branch_scores: BranchScores;
  gate_trust: GateTrust;
  scam_type: string;
  intents: Record<string, number>;
  anti_warning_detected: boolean;
  caller_metadata: CallerMetadata;
  transcript: string;
  flagged_phrases: string[];
  first_alert_seconds: number | null;
  channel_info: ChannelInfo;
  duration_seconds: number;
  audio_telemetry?: AudioTelemetry;
}

export interface EarlyPrefixResult {
  prefix_seconds: number;
  evaluated_duration: number;
  risk: number;
  verdict: string;
  text_prob: number;
  audio_prob: number;
  fused_prob: number;
  detection_latency_ms: number;
  flagged_phrases: string[];
  scam_type: string;
}

export interface EarlyDetectionResponse {
  call_id: string;
  total_duration: number;
  prefix_evaluations: EarlyPrefixResult[];
  recommended_termination_second: number | null;
  time_saved_seconds: number;
}

export interface ChannelLabComparison {
  codec: string;
  snr_db: number;
  packet_loss: number;
  clean_snr_db: number;
  degraded_snr_db: number;
  measured_wer: number;
  clean_risk: number;
  degraded_risk: number;
  text_trust_shift: {
    clean_text_trust: number;
    clean_audio_trust: number;
    degraded_text_trust: number;
    degraded_audio_trust: number;
  };
  clean_waveform: number[];
  degraded_waveform: number[];
  clean_transcript: string;
  degraded_transcript: string;
}

export interface SampleCallInfo {
  call_id: string;
  filename: string;
  title: string;
  duration_seconds: number;
  category: string;
  scam_type: string;
  ground_truth_label: number;
  audio_url: string;
  reference_transcript: string;
}

export interface StreamChunk {
  event: string;
  current_second: number;
  partial_transcript: string;
  fused_risk: number;
  uncertainty?: { lower: number; upper: number };
  verdict?: string;
  gate_trust?: { text_trust: number; audio_trust: number };
  scam_type?: string;
  intents?: Record<string, number>;
  is_alert_active: boolean;
  flagged_phrases: string[];
  trajectory?: Array<{ t: number; risk: number; raw_risk: number }>;
}

export const API_BASE = 'http://127.0.0.1:8000';

export async function analyzeCall(
  file?: File,
  transcriptOverride?: string,
  codec = 'amr_nb',
  snrDb = 5.0,
  packetLoss = 0.03,
  phoneNumber = '+91 98XXX XXX21'
): Promise<AnalysisResult> {
  const form = new FormData();
  if (file) form.append('file', file);
  if (transcriptOverride) form.append('transcript_override', transcriptOverride);
  form.append('codec', codec);
  form.append('snr_db', String(snrDb));
  form.append('packet_loss', String(packetLoss));
  form.append('phone_number', phoneNumber);

  const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`API error ${res.status}: ${res.statusText}`);
  return res.json();
}

export async function getSampleCalls(): Promise<SampleCallInfo[]> {
  const res = await fetch(`${API_BASE}/sample_calls`);
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

export async function runEarlyDetection(callId: string, transcript?: string): Promise<EarlyDetectionResponse> {
  const form = new FormData();
  form.append('call_id', callId);
  if (transcript) form.append('transcript_override', transcript);

  const res = await fetch(`${API_BASE}/early_detection`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Early detection error ${res.status}`);
  return res.json();
}

export async function runChannelLab(
  transcript: string,
  codec = 'amr_nb',
  snrDb = 5.0,
  packetLoss = 0.03
): Promise<ChannelLabComparison> {
  const form = new FormData();
  form.append('transcript', transcript);
  form.append('codec', codec);
  form.append('snr_db', String(snrDb));
  form.append('packet_loss', String(packetLoss));

  const res = await fetch(`${API_BASE}/channel_lab`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Channel lab error ${res.status}`);
  return res.json();
}

export async function getEvaluationBenchmarks(): Promise<any> {
  const res = await fetch(`${API_BASE}/evaluation_benchmarks`);
  if (!res.ok) throw new Error(`Benchmarks error ${res.status}`);
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2500) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function submitFeedback(
  callId: string,
  feedbackType: 'false_positive' | 'confirmed_fraud',
  notes = '',
  transcriptSnippet = '',
  reportedRisk = 0.0
): Promise<{ status: string; message: string }> {
  const res = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      call_id: callId,
      feedback_type: feedbackType,
      notes,
      transcript_snippet: transcriptSnippet,
      reported_risk: reportedRisk
    })
  });
  if (!res.ok) throw new Error(`Feedback error ${res.status}`);
  return res.json();
}
