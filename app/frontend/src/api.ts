// API types for CallGuard backend

export interface BranchScores {
  text: number;
  audio: number;
  fused: number;
}

export interface ChannelInfo {
  sample_rate: number;
  codec: string;
  packet_loss: number;
  wer: number | null;
}

export interface AnalysisResult {
  risk: number;
  risk_label: 'low' | 'moderate' | 'high';
  branch_scores: BranchScores;
  transcript: string;
  flagged_phrases: string[];
  first_alert_seconds: number | null;
  channel_info: ChannelInfo;
  duration_seconds: number;
}

export interface StreamChunk {
  event: string;
  current_second: number;
  partial_transcript: string;
  fused_risk: number;
  is_alert_active: boolean;
  flagged_phrases: string[];
}

export const API_BASE = 'http://127.0.0.1:8000';

export async function analyzeCall(
  file?: File,
  transcriptOverride?: string,
  codec = 'amr_nb',
  snrDb = 5.0,
  packetLoss = 0.03
): Promise<AnalysisResult> {
  const form = new FormData();
  if (file) form.append('file', file);
  if (transcriptOverride) form.append('transcript_override', transcriptOverride);
  form.append('codec', codec);
  form.append('snr_db', String(snrDb));
  form.append('packet_loss', String(packetLoss));

  const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
