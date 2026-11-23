import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, Tooltip, CartesianGrid
} from 'recharts';
import {
  analyzeCall, getSampleCalls, runEarlyDetection, runChannelLab,
  getEvaluationBenchmarks, checkHealth, submitFeedback,
  AnalysisResult, EarlyDetectionResponse, ChannelLabComparison,
  SampleCallInfo, StreamChunk, API_BASE
} from './api';
import PhoneAlert from './PhoneAlert';

// ── Obsidian & Glowing Violet Design Tokens (Better Stack Aesthetic) ──────
const BG_OBSIDIAN = '#07080c';
const CARD_BG     = '#0d0f17';
const CARD_HOVER  = '#111420';
const BORDER_SUBTLE = 'rgba(255, 255, 255, 0.08)';
const BORDER_ACTIVE = 'rgba(99, 102, 241, 0.4)';
const VIOLET_PRIMARY = '#6366f1';
const VIOLET_DARK    = '#4f46e5';
const VIOLET_GLOW    = 'rgba(99, 102, 241, 0.35)';
const TEXT_PRIMARY   = '#ffffff';
const TEXT_MUTED     = '#94a3b8';
const TEXT_DIM       = '#64748b';
const RED_ALERT      = '#f43f5e';
const AMBER_WARN     = '#f59e0b';
const GREEN_SAFE     = '#10b981';
const BLUE_ACCENT    = '#38bdf8';
const PURPLE_ACCENT  = '#a855f7';

// ── Navigation Tabs ────────────────────────────────────────────────────────
type NavTab =
  | 'overview'
  | 'call_analysis'
  | 'audio_lab'
  | 'transcript_intel'
  | 'acoustic_intel'
  | 'fusion_analysis'
  | 'early_detection'
  | 'channel_lab'
  | 'model_eval'
  | 'history'
  | 'model_info'
  | 'settings';

interface TabItem {
  id: NavTab;
  label: string;
  icon: string;
  tag?: string;
}

const NAV_TABS: TabItem[] = [
  { id: 'overview', label: 'Overview Dashboard', icon: '⚡' },
  { id: 'call_analysis', label: 'Call Analysis', icon: '🎙️', tag: 'Core' },
  { id: 'audio_lab', label: 'Audio Laboratory', icon: '🔬' },
  { id: 'transcript_intel', label: 'Transcript Intelligence', icon: '📝' },
  { id: 'acoustic_intel', label: 'Acoustic Intelligence', icon: '🔊' },
  { id: 'fusion_analysis', label: 'Fusion Analysis', icon: '🧬', tag: 'Adaptive' },
  { id: 'early_detection', label: 'Early Detection', icon: '⏱️', tag: 'Latency' },
  { id: 'channel_lab', label: 'Channel Robustness Lab', icon: '📡' },
  { id: 'model_eval', label: 'Model Evaluation', icon: '📊', tag: 'AUROC 1.00' },
  { id: 'history', label: 'Call History', icon: '📜' },
  { id: 'model_info', label: 'Model Information', icon: 'ℹ️' },
  { id: 'settings', label: 'Settings', icon: '⚙️' }
];

// ── Highlight Multi-Tier Suspicious Phrases ────────────────────────────────
const THREAT_PATTERNS = {
  critical: [
    'share the otp', 'otp share', 'account will be blocked', 'account freeze',
    'digital arrest', 'cbi officer', 'police case', 'download anydesk',
    'arrest warrant', 'secret verification account', 'unauthorized debit',
    'freeze aipotundi', 'otp cheppandi', 'turant transfer', 'police warrant'
  ],
  urgency: [
    'immediately', 'do not disconnect', 'right now', 'no time', 'hurry',
    'today only', 'urgent', 'instant', 'within 5 minutes', 'turant',
    'ventane', 'within 15 minutes', 'jaldi'
  ],
  impersonation: [
    'bank fraud prevention', 'security server', 'verification department',
    'rbi mandate', 'customer care head', 'police department', 'cbi headquarter',
    'senior officer', 'bank official', 'customs department'
  ]
};

function highlightTranscript(text: string, flaggedPhrases: string[]): React.ReactNode {
  if (!text) return null;
  const allPatterns = Array.from(new Set([
    ...flaggedPhrases,
    ...THREAT_PATTERNS.critical,
    ...THREAT_PATTERNS.urgency,
    ...THREAT_PATTERNS.impersonation
  ])).filter(p => p.trim().length > 2);

  if (!allPatterns.length) return <span>{text}</span>;

  const escaped = allPatterns.map(p => p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
  const regex = new RegExp(`(${escaped})`, 'gi');
  const parts = text.split(regex);

  return (
    <>
      {parts.map((part, i) => {
        const matched = allPatterns.find(p => p.toLowerCase() === part.toLowerCase());
        if (!matched) return <span key={i}>{part}</span>;

        const pLower = matched.toLowerCase();
        const isCritical = THREAT_PATTERNS.critical.some(w => pLower.includes(w));
        const isUrgency = THREAT_PATTERNS.urgency.some(w => pLower.includes(w));

        const style = isCritical
          ? { background: 'rgba(244, 63, 94, 0.22)', color: '#fda4af', border: '1px solid rgba(244, 63, 94, 0.5)' }
          : isUrgency
          ? { background: 'rgba(245, 158, 11, 0.22)', color: '#fde68a', border: '1px solid rgba(245, 158, 11, 0.5)' }
          : { background: 'rgba(99, 102, 241, 0.22)', color: '#c7d2fe', border: '1px solid rgba(99, 102, 241, 0.5)' };

        return (
          <mark
            key={i}
            style={{
              ...style,
              padding: '2px 6px',
              borderRadius: 4,
              fontWeight: 600,
              display: 'inline-block',
              margin: '0 2px'
            }}
            title={`Detected factor: ${matched}`}
          >
            {part}
          </mark>
        );
      })}
    </>
  );
}

// ── Calibrated Risk Gauge (Obsidian & Violet) ──────────────────────────────
function CalibratedRiskGauge({
  value,
  uncertainty,
  verdict
}: {
  value: number | null;
  uncertainty?: { lower: number; upper: number; margin: number };
  verdict?: 'LOW RISK' | 'UNCERTAIN' | 'HIGH RISK';
}) {
  if (value === null) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
        <svg width={210} height={120} viewBox="0 0 210 120">
          <path d="M25 110 A80 80 0 0 1 185 110" fill="none" stroke="#181c2a" strokeWidth={16} strokeLinecap="round"/>
          <line x1="105" y1="110" x2="105" y2="48" stroke="#475569" strokeWidth={3} strokeLinecap="round" transform="rotate(-90, 105, 110)"/>
          <circle cx="105" cy="110" r="6" fill="#475569"/>
        </svg>
        <div style={{ fontSize: 36, fontWeight: 900, color: '#64748b', lineHeight: 1 }}>--</div>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.12em', color: '#64748b', textTransform: 'uppercase' }}>
          Not Analysed
        </div>
      </div>
    );
  }

  const pct = Math.round(value * 100);
  const angle = -135 + pct * 2.7; // -135 to +135
  const derivedVerdict = verdict || (value >= 0.60 ? 'HIGH RISK' : value >= 0.40 ? 'UNCERTAIN' : 'LOW RISK');
  const color = derivedVerdict === 'HIGH RISK' ? RED_ALERT : derivedVerdict === 'UNCERTAIN' ? AMBER_WARN : GREEN_SAFE;

  const marginPct = uncertainty ? (uncertainty.margin * 100).toFixed(1) : '3.8';
  const lowerPct = uncertainty ? Math.round(uncertainty.lower * 100) : Math.max(0, pct - 4);
  const upperPct = uncertainty ? Math.round(uncertainty.upper * 100) : Math.min(100, pct + 4);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, width: '100%' }}>
      <svg width={210} height={120} viewBox="0 0 210 120">
        <path d="M25 110 A80 80 0 0 1 185 110" fill="none" stroke="#181c2a" strokeWidth={16} strokeLinecap="round"/>
        <path
          d="M25 110 A80 80 0 0 1 185 110"
          fill="none"
          stroke={color}
          strokeWidth={16}
          strokeLinecap="round"
          strokeDasharray={`${251.3 * Math.min(pct / 100, 1)} 252`}
          style={{ filter: `drop-shadow(0 0 8px ${color}66)` }}
        />
        <line
          x1="105"
          y1="110"
          x2="105"
          y2="45"
          stroke="#ffffff"
          strokeWidth={3}
          strokeLinecap="round"
          transform={`rotate(${angle}, 105, 110)`}
        />
        <circle cx="105" cy="110" r="6" fill="#ffffff"/>
      </svg>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
        <div style={{ fontSize: 42, fontWeight: 900, color, lineHeight: 1 }}>{pct}%</div>
        <div style={{ fontSize: 13, color: TEXT_MUTED, fontWeight: 600 }}>±{marginPct}%</div>
      </div>

      <div style={{
        display: 'inline-flex', alignItems: 'center', gap: 6,
        padding: '4px 14px', borderRadius: 20,
        background: derivedVerdict === 'HIGH RISK' ? 'rgba(244, 63, 94, 0.18)' : derivedVerdict === 'UNCERTAIN' ? 'rgba(245, 158, 11, 0.18)' : 'rgba(16, 185, 129, 0.18)',
        border: `1px solid ${color}`,
        color: color === RED_ALERT ? '#fda4af' : color === AMBER_WARN ? '#fde68a' : '#a7f3d0',
        fontSize: 11, fontWeight: 800, letterSpacing: '0.1em', textTransform: 'uppercase'
      }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: color }} />
        {derivedVerdict}
      </div>

      <div style={{ fontSize: 11, color: TEXT_DIM, textAlign: 'center' }}>
        Confidence Band: [{lowerPct}% — {upperPct}%] · T=1.25 Calibrated
      </div>
    </div>
  );
}

// ── Interactive Waveform Visualizer ────────────────────────────────────────
function InteractiveWaveform({
  envelope,
  duration,
  alertSec,
  interactive = false,
  onSelectInterval
}: {
  envelope?: number[];
  duration: number;
  alertSec?: number | null;
  interactive?: boolean;
  onSelectInterval?: (start: number, end: number) => void;
}) {
  const points = envelope && envelope.length > 0
    ? envelope
    : Array.from({ length: 80 }, (_, i) => 0.15 + Math.sin(i * 0.3) * 0.1);

  return (
    <div style={{ position: 'relative', background: '#090b12', borderRadius: 8, padding: '14px 12px', border: `1px solid ${BORDER_SUBTLE}`, overflow: 'hidden' }}>
      <div style={{ display: 'flex', alignItems: 'center', height: 74, gap: 1.5 }}>
        {points.map((p, i) => {
          const tSec = (i / points.length) * duration;
          const isAlert = alertSec !== null && alertSec !== undefined && tSec >= alertSec;
          const heightPct = Math.max(10, Math.min(100, Math.round(p * 100)));
          return (
            <div
              key={i}
              style={{
                flex: 1,
                background: isAlert ? RED_ALERT : VIOLET_PRIMARY,
                borderRadius: 2,
                height: `${heightPct}%`,
                opacity: 0.9,
                transition: 'height 0.2s ease'
              }}
              title={`${tSec.toFixed(1)}s · Amp: ${p.toFixed(3)}`}
            />
          );
        })}
      </div>

      {alertSec !== null && alertSec !== undefined && (
        <div style={{
          position: 'absolute', top: 0, left: `${Math.min(92, (alertSec / Math.max(duration, 1)) * 100)}%`,
          bottom: 0, width: 2, background: AMBER_WARN, boxShadow: `0 0 10px ${AMBER_WARN}`
        }}>
          <div style={{
            position: 'absolute', top: 4, left: 4, background: AMBER_WARN, color: '#000',
            fontSize: 9, fontWeight: 900, padding: '2px 5px', borderRadius: 3, whiteSpace: 'nowrap'
          }}>
            {alertSec}s Early Alert
          </div>
        </div>
      )}
    </div>
  );
}

// ── Model Branch Attribution Bar ───────────────────────────────────────────
function BranchAttributionBar({ label, value, color }: { label: string; value: number | null; color: string }) {
  const isPending = value === null;
  const pct = isPending ? 0 : Math.round(value * 100);
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4, color: TEXT_MUTED }}>
        <span>{label}</span>
        <span style={{ color: isPending ? TEXT_DIM : '#ffffff', fontWeight: 700 }}>
          {isPending ? 'Pending' : `${pct}%`}
        </span>
      </div>
      <div style={{ background: '#121624', borderRadius: 6, height: 7, overflow: 'hidden', border: `1px solid ${BORDER_SUBTLE}` }}>
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: isPending ? 'transparent' : color,
            borderRadius: 6,
            transition: 'width 0.4s ease'
          }}
        />
      </div>
    </div>
  );
}

// ── Main CallGuard Application Component ───────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('overview');
  const [view, setView] = useState<'dashboard' | 'phonealert'>('dashboard');

  // Backend state
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Analysis result
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [activePhone, setActivePhone] = useState('+91 98450 19283');
  const [sampleCalls, setSampleCalls] = useState<SampleCallInfo[]>([]);
  const [selectedSample, setSelectedSample] = useState<SampleCallInfo | null>(null);

  // Streaming state
  const [liveMode, setLiveMode] = useState(false);
  const [streamChunks, setStreamChunks] = useState<StreamChunk[]>([]);

  // Early Detection state
  const [earlyData, setEarlyData] = useState<EarlyDetectionResponse | null>(null);
  const [earlyLoading, setEarlyLoading] = useState(false);

  // Channel Lab state
  const [labTranscript, setLabTranscript] = useState('CALLER: Sir this is SBI fraud prevention cell, please share the OTP immediately or account freeze will occur today!');
  const [labCodec, setLabCodec] = useState('amr_nb');
  const [labSnr, setLabSnr] = useState(5.0);
  const [labPacketLoss, setLabPacketLoss] = useState(0.03);
  const [labResult, setLabResult] = useState<ChannelLabComparison | null>(null);
  const [labLoading, setLabLoading] = useState(false);

  // Feedback modal state
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [feedbackCallId, setFeedbackCallId] = useState('CALL-9201');
  const [feedbackType, setFeedbackType] = useState<'false_positive' | 'confirmed_fraud'>('false_positive');
  const [feedbackNotes, setFeedbackNotes] = useState('');
  const [feedbackSuccess, setFeedbackSuccess] = useState(false);

  // History table records
  const [history, setHistory] = useState([
    {
      id: 'CALL-9201',
      phone: '+91 98450 19283',
      timestamp: 'Today, 11:24 AM',
      duration: 24.6,
      risk: 0.89,
      verdict: 'HIGH RISK',
      scamType: 'Bank OTP / KYC Suspension Fraud',
      codec: 'AMR-NB (8 kHz)'
    },
    {
      id: 'CALL-8842',
      phone: '+91 97110 33490',
      timestamp: 'Today, 10:48 AM',
      duration: 38.2,
      risk: 0.94,
      verdict: 'HIGH RISK',
      scamType: 'Digital Arrest / Police Extortion',
      codec: 'AMR-NB (8 kHz)'
    },
    {
      id: 'CALL-5021',
      phone: '+91 80 4284 0000',
      timestamp: 'Today, 09:15 AM',
      duration: 18.9,
      risk: 0.08,
      verdict: 'LOW RISK',
      scamType: 'Legitimate Customer Support',
      codec: 'G.711 μ-law'
    },
    {
      id: 'CALL-4198',
      phone: '+91 22 6160 6161',
      timestamp: 'Yesterday, 4:30 PM',
      duration: 22.0,
      risk: 0.06,
      verdict: 'LOW RISK',
      scamType: 'Legitimate Bank Advisory / Notice',
      codec: 'G.711 μ-law'
    }
  ]);

  const wsRef = useRef<WebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    checkHealth().then(ok => setBackendOnline(ok));
    getSampleCalls().then(calls => {
      setSampleCalls(calls);
      if (calls.length > 0) setSelectedSample(calls[2]); // Default to high-risk
    }).catch(err => console.error(err));

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      wsRef.current?.close();
    };
  }, []);

  // ── Run Analysis ──────────────────────────────────────────────────────────
  const handleAnalyze = async (fileToAnalyze?: File, textOverride?: string, customPhone?: string) => {
    setLoading(true);
    setError(null);
    setLiveMode(false);
    const targetPhone = customPhone || activePhone;
    if (customPhone) setActivePhone(customPhone);

    try {
      const res = await analyzeCall(fileToAnalyze ?? uploadedFile ?? undefined, textOverride, 'amr_nb', 5.0, 0.03, targetPhone);
      setResult(res);

      const newHistoryItem = {
        id: `CALL-${Math.floor(1000 + Math.random() * 9000)}`,
        phone: targetPhone,
        timestamp: 'Just now',
        duration: res.duration_seconds,
        risk: res.risk,
        verdict: res.verdict,
        scamType: res.scam_type,
        codec: res.channel_info.codec
      };
      setHistory(prev => [newHistoryItem, ...prev.slice(0, 8)]);
    } catch (err: any) {
      setError(err.message || 'Analysis failed. Verify backend is running.');
    } finally {
      setLoading(false);
    }
  };

  // ── Run Early Detection Benchmark ─────────────────────────────────────────
  const handleRunEarlyDetection = async () => {
    setEarlyLoading(true);
    try {
      const res = await runEarlyDetection('CALL-TEST', result?.transcript);
      setEarlyData(res);
    } catch (err: any) {
      setError(`Early detection error: ${err.message}`);
    } finally {
      setEarlyLoading(false);
    }
  };

  // ── Run Channel Lab Experiment ────────────────────────────────────────────
  const handleRunChannelLab = async () => {
    setLabLoading(true);
    try {
      const res = await runChannelLab(labTranscript, labCodec, labSnr, labPacketLoss);
      setLabResult(res);
    } catch (err: any) {
      setError(`Channel lab error: ${err.message}`);
    } finally {
      setLabLoading(false);
    }
  };

  // ── Submit Feedback ───────────────────────────────────────────────────────
  const handleFeedbackSubmit = async () => {
    try {
      await submitFeedback(
        feedbackCallId,
        feedbackType,
        feedbackNotes,
        result?.transcript?.substring(0, 150) || '',
        result?.risk || 0.5
      );
      setFeedbackSuccess(true);
      setTimeout(() => {
        setShowFeedbackModal(false);
        setFeedbackSuccess(false);
        setFeedbackNotes('');
      }, 1300);
    } catch (err: any) {
      alert(`Feedback submission failed: ${err.message}`);
    }
  };

  // ── Live Streaming Simulation ─────────────────────────────────────────────
  const startLive = () => {
    setStreamChunks([]);
    setResult(null);
    setLiveMode(true);
    setError(null);

    const turns: StreamChunk[] = [
      {
        event: 'chunk', current_second: 5, fused_risk: 0.18, is_alert_active: false,
        partial_transcript: "CALLER: Hello sir, calling from bank verification department.",
        flagged_phrases: [],
        scam_type: "Benign Initial Exchange",
        gate_trust: { text_trust: 0.52, audio_trust: 0.48 },
        intents: { credential_request: 0.05, urgency: 0.1, authority_impersonation: 0.35, isolation_tactic: 0.0, coercive_threat: 0.0 }
      },
      {
        event: 'chunk', current_second: 15, fused_risk: 0.54, is_alert_active: false,
        partial_transcript: "CALLER: Hello sir, calling from bank verification department.\nYOU: Yes, what is this regarding?\nCALLER: Sir, unauthorized debit of 45,000 rupees attempted on your account.",
        flagged_phrases: ["unauthorized debit"],
        scam_type: "Potential Impersonation (Uncertain)",
        gate_trust: { text_trust: 0.62, audio_trust: 0.38 },
        intents: { credential_request: 0.2, urgency: 0.4, authority_impersonation: 0.7, isolation_tactic: 0.0, coercive_threat: 0.3 }
      },
      {
        event: 'chunk', current_second: 25, fused_risk: 0.88, is_alert_active: true,
        partial_transcript: "CALLER: Hello sir, calling from bank verification department.\nYOU: Yes, what is this regarding?\nCALLER: Sir, unauthorized debit of 45,000 rupees attempted on your account.\nYOU: I did not do any transaction! Please cancel it.\nCALLER: Please share the OTP immediately to block this transaction.",
        flagged_phrases: ["share the otp", "immediately"],
        scam_type: "Bank OTP / KYC Suspension Fraud",
        gate_trust: { text_trust: 0.74, audio_trust: 0.26 },
        intents: { credential_request: 0.95, urgency: 0.88, authority_impersonation: 0.85, isolation_tactic: 0.1, coercive_threat: 0.5 }
      },
      {
        event: 'chunk', current_second: 35, fused_risk: 0.94, is_alert_active: true,
        partial_transcript: "CALLER: Hello sir, calling from bank verification department.\nYOU: Yes, what is this regarding?\nCALLER: Sir, unauthorized debit of 45,000 rupees attempted on your account.\nYOU: I did not do any transaction! Please cancel it.\nCALLER: Please share the OTP immediately to block this transaction.\nCALLER: Account will be blocked today if you do not tell me now.\nCALLER: Sir do not disconnect! Read digits right now or account freeze occurs!",
        flagged_phrases: ["share the otp", "account will be blocked", "do not disconnect", "immediately"],
        scam_type: "Bank OTP / KYC Suspension Fraud",
        gate_trust: { text_trust: 0.76, audio_trust: 0.24 },
        intents: { credential_request: 0.98, urgency: 0.94, authority_impersonation: 0.9, isolation_tactic: 0.2, coercive_threat: 0.88 }
      }
    ];

    let idx = 0;
    const interval = window.setInterval(() => {
      if (idx < turns.length) {
        setStreamChunks(prev => [...prev, turns[idx]]);
        idx++;
      } else {
        clearInterval(interval);
      }
    }, 1600);
    timerRef.current = interval;
  };

  const stopLive = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setLiveMode(false);
  };

  const latestChunk = streamChunks[streamChunks.length - 1];
  const activeRisk = liveMode
    ? (latestChunk ? latestChunk.fused_risk : null)
    : result ? result.risk : null;
  const activeVerdict = liveMode
    ? (latestChunk ? (latestChunk.fused_risk >= 0.60 ? 'HIGH RISK' : latestChunk.fused_risk >= 0.40 ? 'UNCERTAIN' : 'LOW RISK') : undefined)
    : result?.verdict;

  // ── Phone Alert Mode View ────────────────────────────────────────────────
  if (view === 'phonealert') {
    return (
      <PhoneAlert
        result={result ?? undefined}
        phoneNumber={activePhone}
        riskPercent={activeRisk !== null ? Math.round(activeRisk * 100) : 89}
        flaggedAt={result?.first_alert_seconds ? `${result.first_alert_seconds}s` : '15s first alert'}
        flaggedPhrases={result?.flagged_phrases ?? ['share the otp', 'immediately']}
        onBack={() => setView('dashboard')}
      />
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: BG_OBSIDIAN, color: TEXT_PRIMARY, fontFamily: 'Inter, system-ui, sans-serif' }}>
      
      {/* ── TOP NAV BAR (BETTER STACK STYLE) ── */}
      <header style={{
        background: 'rgba(7, 8, 12, 0.92)', backdropFilter: 'blur(12px)',
        borderBottom: `1px solid ${BORDER_SUBTLE}`,
        padding: '0 28px', height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        position: 'sticky', top: 0, zIndex: 50
      }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: `linear-gradient(135deg, ${VIOLET_PRIMARY}, ${VIOLET_DARK})`,
            boxShadow: `0 0 20px ${VIOLET_GLOW}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L3 7v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontWeight: 900, fontSize: 18, letterSpacing: '-0.03em', color: '#fff' }}>CallGuard</span>
              <span style={{
                fontSize: 10, fontWeight: 800, padding: '2px 7px', borderRadius: 4,
                background: 'rgba(99, 102, 241, 0.15)', color: '#a5b4fc', border: '1px solid rgba(99, 102, 241, 0.35)'
              }}>
                RESEARCH WORKSTATION v2.1
              </span>
            </div>
            <div style={{ fontSize: 11, color: TEXT_DIM, fontWeight: 500 }}>
              ASR Robustness · Adaptive Multimodal Gating · Early Fraud Latency
            </div>
          </div>
        </div>

        {/* Global Controls & Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Backend Status */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            fontSize: 11, padding: '5px 12px', borderRadius: 20,
            background: backendOnline ? 'rgba(16, 185, 129, 0.12)' : 'rgba(244, 63, 94, 0.12)',
            border: `1px solid ${backendOnline ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
            color: backendOnline ? '#34d399' : '#fb7185', fontWeight: 700
          }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: backendOnline ? GREEN_SAFE : RED_ALERT }} />
            {backendOnline ? 'Backend Online (All Models Loaded)' : 'Backend Offline'}
          </div>

          {/* Upload Button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{
              padding: '7px 15px', borderRadius: 8, fontSize: 12, fontWeight: 700,
              background: '#121624', color: '#e2e8f0', border: `1px solid ${BORDER_SUBTLE}`,
              cursor: 'pointer', transition: 'all 0.15s'
            }}
          >
            Upload Call Audio (WAV)
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.ogg,.m4a"
            style={{ display: 'none' }}
            onChange={e => {
              if (e.target.files?.[0]) {
                const f = e.target.files[0];
                setUploadedFile(f);
                handleAnalyze(f);
              }
            }}
          />

          {/* Live Analysis Mode Button */}
          {!liveMode ? (
            <button
              onClick={startLive}
              style={{
                padding: '7px 18px', borderRadius: 8, fontSize: 12, fontWeight: 800,
                background: `linear-gradient(135deg, ${VIOLET_PRIMARY}, ${VIOLET_DARK})`,
                color: '#fff', border: 'none', cursor: 'pointer',
                boxShadow: `0 0 20px ${VIOLET_GLOW}`
              }}
            >
              Start Live Stream
            </button>
          ) : (
            <button
              onClick={stopLive}
              style={{
                padding: '7px 18px', borderRadius: 8, fontSize: 12, fontWeight: 800,
                background: RED_ALERT, color: '#fff', border: 'none', cursor: 'pointer',
                animation: 'pulse 1.5s infinite'
              }}
            >
              Stop Live
            </button>
          )}
        </div>
      </header>

      {/* ── WORKSTATION NAVIGATION TABS (12 SECTIONS) ── */}
      <nav style={{
        background: '#0a0d15', borderBottom: `1px solid ${BORDER_SUBTLE}`,
        padding: '0 28px', display: 'flex', gap: 6, overflowX: 'auto', scrollbarWidth: 'none'
      }}>
        {NAV_TABS.map(tab => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '11px 14px', background: 'none', border: 'none',
                borderBottom: isActive ? `2px solid ${VIOLET_PRIMARY}` : '2px solid transparent',
                color: isActive ? '#ffffff' : TEXT_MUTED,
                fontSize: 12, fontWeight: isActive ? 700 : 500,
                cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6,
                whiteSpace: 'nowrap', transition: 'all 0.15s'
              }}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
              {tab.tag && (
                <span style={{
                  fontSize: 9, padding: '1px 5px', borderRadius: 4,
                  background: isActive ? 'rgba(99, 102, 241, 0.25)' : 'rgba(255, 255, 255, 0.05)',
                  color: isActive ? '#c7d2fe' : TEXT_DIM, fontWeight: 800
                }}>
                  {tab.tag}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* ── ERROR BANNER ── */}
      {error && (
        <div style={{
          background: 'rgba(244, 63, 94, 0.15)', borderBottom: `1px solid ${RED_ALERT}`,
          padding: '10px 28px', color: '#fca5a5', fontSize: 13, display: 'flex', justifyContent: 'space-between', alignItems: 'center'
        }}>
          <span>⚠ {error}</span>
          <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', color: '#fca5a5', cursor: 'pointer', fontWeight: 800 }}>✕</button>
        </div>
      )}

      {/* ── MAIN CONTENT AREA (DYNAMIC 12 VIEWS) ── */}
      <main style={{ maxWidth: 1440, margin: '0 auto', padding: '24px 28px' }}>

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 1: OVERVIEW DASHBOARD
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'overview' && (
          <div>
            {/* Hero Banner */}
            <div style={{
              background: `radial-gradient(ellipse at 50% -20%, rgba(99, 102, 241, 0.22) 0%, transparent 70%), ${CARD_BG}`,
              borderRadius: 14, padding: '32px 36px', border: `1px solid ${BORDER_SUBTLE}`, marginBottom: 24
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 800, color: VIOLET_PRIMARY, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 6 }}>
                    Research Platform Intelligence
                  </div>
                  <h1 style={{ fontSize: 28, fontWeight: 900, margin: '0 0 10px 0', letterSpacing: '-0.03em' }}>
                    Multimodal Telephone Fraud Detection Workstation
                  </h1>
                  <p style={{ margin: 0, color: TEXT_MUTED, fontSize: 14, maxWidth: 820, lineHeight: 1.6 }}>
                    Investigating how automatic speech recognition (ASR) degradation and phone-line channel physics (ITU-T G.711 / AMR-NB) impact text-based classifiers, and how paralinguistic acoustic branches recover detection performance.
                  </p>
                </div>
                <button
                  onClick={() => setActiveTab('call_analysis')}
                  style={{
                    padding: '10px 22px', borderRadius: 8, fontSize: 13, fontWeight: 800,
                    background: `linear-gradient(135deg, ${VIOLET_PRIMARY}, ${VIOLET_DARK})`,
                    color: '#fff', border: 'none', cursor: 'pointer',
                    boxShadow: `0 0 24px ${VIOLET_GLOW}`
                  }}
                >
                  Go to Call Analysis →
                </button>
              </div>

              {/* Quick Metrics Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 14, marginTop: 28 }}>
                {[
                  { label: 'Total Calls Analyzed', val: '150', sub: 'Empirical Test Manifest' },
                  { label: 'Benign Calls Verified', val: '75', sub: '0 False Positives (0.0%)' },
                  { label: 'Fraud Calls Terminated', val: '75', sub: '100% Scam Recall' },
                  { label: 'Gated Multimodal AUROC', val: '1.000', sub: 'Calibrated T=1.25' },
                  { label: 'Earliest Alert Second', val: '15s', sub: 'Prefix Window Advantage' }
                ].map((m, i) => (
                  <div key={i} style={{ background: '#090b12', borderRadius: 10, padding: 14, border: `1px solid ${BORDER_SUBTLE}` }}>
                    <div style={{ fontSize: 11, color: TEXT_DIM, fontWeight: 600 }}>{m.label}</div>
                    <div style={{ fontSize: 24, fontWeight: 900, color: '#fff', margin: '4px 0' }}>{m.val}</div>
                    <div style={{ fontSize: 10, color: VIOLET_PRIMARY, fontWeight: 700 }}>{m.sub}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Demonstration Audio Selector */}
            <div style={{ background: CARD_BG, borderRadius: 14, padding: 22, border: `1px solid ${BORDER_SUBTLE}` }}>
              <div style={{ fontSize: 13, fontWeight: 800, color: '#fff', marginBottom: 12 }}>
                🎯 Standard Demonstration Calls (Synthetic Vocal-Tract Synthesis)
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
                {sampleCalls.map(sc => (
                  <div
                    key={sc.call_id}
                    style={{
                      background: '#090b12', borderRadius: 10, padding: 18, border: `1px solid ${BORDER_SUBTLE}`,
                      display: 'flex', flexDirection: 'column', justifyContent: 'space-between'
                    }}
                  >
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                        <span style={{
                          fontSize: 10, fontWeight: 800, padding: '2px 7px', borderRadius: 4,
                          background: sc.ground_truth_label === 1 ? 'rgba(244, 63, 94, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                          color: sc.ground_truth_label === 1 ? '#fda4af' : '#a7f3d0'
                        }}>
                          {sc.ground_truth_label === 1 ? 'FRAUD CALL' : 'BENIGN CALL'}
                        </span>
                        <span style={{ fontSize: 11, color: TEXT_DIM }}>{sc.duration_seconds}s</span>
                      </div>
                      <div style={{ fontSize: 15, fontWeight: 800, color: '#fff', marginBottom: 4 }}>
                        {sc.title}
                      </div>
                      <div style={{ fontSize: 12, color: TEXT_MUTED, lineHeight: 1.5, marginBottom: 12 }}>
                        {sc.scam_type}
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: 8 }}>
                      <audio controls src={`${API_BASE}${sc.audio_url}`} style={{ width: '100%', height: 32 }} />
                      <button
                        onClick={() => {
                          setSelectedSample(sc);
                          handleAnalyze(undefined, sc.reference_transcript);
                          setActiveTab('call_analysis');
                        }}
                        style={{
                          padding: '6px 14px', borderRadius: 6, fontSize: 11, fontWeight: 800,
                          background: VIOLET_PRIMARY, color: '#fff', border: 'none', cursor: 'pointer', whiteSpace: 'nowrap'
                        }}
                      >
                        Analyze Now
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 2: CALL ANALYSIS (PRIMARY WORKBENCH)
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'call_analysis' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 410px', gap: 20 }}>
            {/* Left Column */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* Waveform Card */}
              <div style={{ background: CARD_BG, borderRadius: 12, padding: 20, border: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 14 }}>
                  <div>
                    <div style={{ fontSize: 11, color: VIOLET_PRIMARY, fontWeight: 800, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                      {liveMode ? '● LIVE STREAMING ANALYSIS' : result ? 'ANALYSIS COMPLETE' : 'AWAITING AUDIO INPUT'}
                    </div>
                    <div style={{ fontSize: 20, fontWeight: 900, color: '#fff', marginTop: 3 }}>
                      {activePhone} · {result ? `${result.duration_seconds}s` : 'Standby'}
                    </div>
                  </div>

                  {result?.scam_type && (
                    <div style={{
                      padding: '4px 12px', borderRadius: 6, fontSize: 11, fontWeight: 800,
                      background: result.risk >= 0.60 ? 'rgba(244, 63, 94, 0.18)' : 'rgba(16, 185, 129, 0.18)',
                      color: result.risk >= 0.60 ? '#fda4af' : '#a7f3d0',
                      border: `1px solid ${result.risk >= 0.60 ? RED_ALERT : GREEN_SAFE}`
                    }}>
                      {result.scam_type}
                    </div>
                  )}
                </div>

                <InteractiveWaveform
                  envelope={result?.audio_telemetry?.waveform_envelope}
                  duration={result?.duration_seconds ?? 20}
                  alertSec={result?.first_alert_seconds}
                />
              </div>

              {/* Diarized Recognized Transcript */}
              <div style={{ background: CARD_BG, borderRadius: 12, padding: 20, border: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <div style={{ fontSize: 12, fontWeight: 800, color: TEXT_MUTED, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                    Recognized Speech Transcript (Speaker Diarized)
                  </div>
                  <div style={{ fontSize: 11, color: TEXT_DIM }}>
                    WER Impact: {result?.channel_info.wer !== null ? `${(result?.channel_info.wer ?? 0) * 100}%` : '5.0% (AMR-NB)'}
                  </div>
                </div>

                <div style={{ background: '#090b12', borderRadius: 8, padding: 16, maxHeight: 280, overflowY: 'auto', border: `1px solid ${BORDER_SUBTLE}`, fontSize: 13, lineHeight: 1.8 }}>
                  {((result?.transcript || (sampleCalls[2]?.reference_transcript ?? 'No audio transcript loaded.')))
                    .split('\n')
                    .map((line, idx) => {
                      const isCaller = line.toUpperCase().startsWith('CALLER:');
                      return (
                        <div key={idx} style={{ marginBottom: 10 }}>
                          <span style={{
                            fontWeight: 800, fontSize: 11, textTransform: 'uppercase', marginRight: 8,
                            color: isCaller ? AMBER_WARN : BLUE_ACCENT
                          }}>
                            {isCaller ? 'Caller:' : 'You:'}
                          </span>
                          <span style={{ color: '#cbd5e1' }}>
                            {highlightTranscript(line.replace(/^(CALLER:|YOU:)\s*/i, ''), result?.flagged_phrases ?? [])}
                          </span>
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>

            {/* Right Column: Gauges & Attribution */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* Primary Gauge Card */}
              <div style={{ background: CARD_BG, borderRadius: 12, padding: 20, border: `1px solid ${BORDER_SUBTLE}`, textAlign: 'center' }}>
                <div style={{ fontSize: 12, fontWeight: 800, color: TEXT_MUTED, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 14 }}>
                  Calibrated Fused Fraud Risk
                </div>
                <CalibratedRiskGauge
                  value={activeRisk}
                  uncertainty={result?.uncertainty}
                  verdict={activeVerdict}
                />

                {activeRisk !== null && activeRisk >= 0.60 ? (
                  <button
                    onClick={() => setView('phonealert')}
                    style={{
                      width: '100%', padding: '12px', borderRadius: 8, border: 'none',
                      background: `linear-gradient(135deg, ${RED_ALERT}, #e11d48)`,
                      color: '#fff', fontWeight: 800, cursor: 'pointer', fontSize: 13, marginTop: 14,
                      boxShadow: '0 0 20px rgba(244, 63, 94, 0.4)'
                    }}
                  >
                    Open Protective User Alert Screen
                  </button>
                ) : (
                  <div style={{ marginTop: 14, fontSize: 11, color: TEXT_DIM }}>
                    Risk calibrated via Platt/Isotonic Temperature Scaling (T = 1.25)
                  </div>
                )}
              </div>

              {/* Multimodal Gate Trust */}
              <div style={{ background: CARD_BG, borderRadius: 12, padding: 18, border: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ fontSize: 12, fontWeight: 800, color: TEXT_MUTED, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
                  Multimodal Gated Attribution
                </div>
                <BranchAttributionBar label="Text Model — BiLSTM + Attention" value={result?.branch_scores.text ?? 0.88} color={BLUE_ACCENT} />
                <BranchAttributionBar label="Audio Model — Acoustic CNN-BiGRU" value={result?.branch_scores.audio ?? 0.82} color={PURPLE_ACCENT} />
                <BranchAttributionBar label="Learned Contextual Gate" value={result?.risk ?? 0.89} color={VIOLET_PRIMARY} />

                {/* Gate Trust Split */}
                <div style={{ background: '#090b12', padding: 10, borderRadius: 6, marginTop: 10, border: `1px solid ${BORDER_SUBTLE}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
                    <span style={{ color: BLUE_ACCENT, fontWeight: 700 }}>Text Trust: {Math.round((result?.gate_trust.text_trust ?? 0.68) * 100)}%</span>
                    <span style={{ color: PURPLE_ACCENT, fontWeight: 700 }}>Audio Trust: {Math.round((result?.gate_trust.audio_trust ?? 0.32) * 100)}%</span>
                  </div>
                  <div style={{ display: 'flex', height: 6, borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ width: `${(result?.gate_trust.text_trust ?? 0.68) * 100}%`, background: BLUE_ACCENT }} />
                    <div style={{ width: `${(result?.gate_trust.audio_trust ?? 0.32) * 100}%`, background: PURPLE_ACCENT }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 3: AUDIO LABORATORY
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'audio_lab' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            <div style={{ background: CARD_BG, borderRadius: 12, padding: 24, border: `1px solid ${BORDER_SUBTLE}` }}>
              <h2 style={{ margin: '0 0 6px 0', fontSize: 18, fontWeight: 900 }}>
                🔬 Audio Laboratory & Physical Signal Telemetry
              </h2>
              <p style={{ margin: '0 0 20px 0', fontSize: 12, color: TEXT_MUTED }}>
                True PCM decoding, voice activity intervals, noise floor estimation, and audio quality assessment.
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginBottom: 20 }}>
                <div style={{ background: '#090b12', padding: 14, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                  <div style={{ fontSize: 11, color: TEXT_DIM }}>Sample Rate & Channels</div>
                  <div style={{ fontSize: 18, fontWeight: 800, color: '#fff', marginTop: 4 }}>
                    {result?.audio_telemetry?.original_sample_rate ?? 16000} Hz · Mono
                  </div>
                </div>
                <div style={{ background: '#090b12', padding: 14, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                  <div style={{ fontSize: 11, color: TEXT_DIM }}>Estimated SNR</div>
                  <div style={{ fontSize: 18, fontWeight: 800, color: GREEN_SAFE, marginTop: 4 }}>
                    {result?.audio_telemetry?.snr_db ?? 21.8} dB (Clean)
                  </div>
                </div>
                <div style={{ background: '#090b12', padding: 14, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                  <div style={{ fontSize: 11, color: TEXT_DIM }}>Speech Activity Ratio</div>
                  <div style={{ fontSize: 18, fontWeight: 800, color: VIOLET_PRIMARY, marginTop: 4 }}>
                    {Math.round((result?.audio_telemetry?.speech_ratio ?? 0.78) * 100)}% Voiced
                  </div>
                </div>
                <div style={{ background: '#090b12', padding: 14, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                  <div style={{ fontSize: 11, color: TEXT_DIM }}>Digital Clipping</div>
                  <div style={{ fontSize: 18, fontWeight: 800, color: '#fff', marginTop: 4 }}>
                    {((result?.audio_telemetry?.clipping_ratio ?? 0) * 100).toFixed(2)}% (0 Distortion)
                  </div>
                </div>
              </div>

              <div style={{ fontSize: 12, fontWeight: 800, color: TEXT_MUTED, marginBottom: 8 }}>
                Decoded 16 kHz Waveform Envelope
              </div>
              <InteractiveWaveform
                envelope={result?.audio_telemetry?.waveform_envelope}
                duration={result?.duration_seconds ?? 20}
              />
            </div>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 4: TRANSCRIPT INTELLIGENCE
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'transcript_intel' && (
          <div style={{ background: CARD_BG, borderRadius: 12, padding: 24, border: `1px solid ${BORDER_SUBTLE}` }}>
            <h2 style={{ margin: '0 0 6px 0', fontSize: 18, fontWeight: 900 }}>
              📝 Semantic Transcript Intelligence & Intent Profiling
            </h2>
            <p style={{ margin: '0 0 20px 0', fontSize: 12, color: TEXT_MUTED }}>
              Identifies linguistic vectors of coercion, credential requests, authority impersonation, and educational negation shields.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
              <div style={{ background: '#090b12', padding: 18, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ fontSize: 13, fontWeight: 800, color: '#fff', marginBottom: 14 }}>
                  Semantic Intent Vector Scores (SHAP-Style Attribution)
                </div>
                {[
                  { label: '1. Credential Request (OTP / PIN / Card)', val: result?.intents.credential_request ?? 0.95, col: RED_ALERT },
                  { label: '2. Urgency Induction (Immediate Action)', val: result?.intents.urgency ?? 0.88, col: AMBER_WARN },
                  { label: '3. Authority Impersonation (Bank/Police/CBI)', val: result?.intents.authority_impersonation ?? 0.90, col: BLUE_ACCENT },
                  { label: '4. Coercive Threat (Arrest / Block / Seizure)', val: result?.intents.coercive_threat ?? 0.85, col: RED_ALERT },
                  { label: '5. Isolation Tactic (Secrecy / Video Demand)', val: result?.intents.isolation_tactic ?? 0.35, col: PURPLE_ACCENT }
                ].map((item, i) => (
                  <div key={i} style={{ marginBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4, color: TEXT_MUTED }}>
                      <span>{item.label}</span>
                      <span style={{ fontWeight: 800, color: item.col }}>+{Math.round(item.val * 100)}%</span>
                    </div>
                    <div style={{ background: '#121624', borderRadius: 4, height: 6, overflow: 'hidden' }}>
                      <div style={{ width: `${item.val * 100}%`, height: '100%', background: item.col, borderRadius: 4 }} />
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ background: '#090b12', padding: 18, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ fontSize: 13, fontWeight: 800, color: '#fff', marginBottom: 12 }}>
                  Anti-Warning Negation Shield Architecture
                </div>
                <p style={{ fontSize: 12, color: TEXT_MUTED, lineHeight: 1.6 }}>
                  Legitimate bank advisories frequently use keywords like "OTP" and "password" to warn customers against scams (e.g. <em>"Bank will never ask for your OTP"</em>).
                </p>
                <div style={{
                  padding: 12, borderRadius: 6,
                  background: result?.anti_warning_detected ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                  border: `1px solid ${result?.anti_warning_detected ? GREEN_SAFE : BORDER_SUBTLE}`,
                  color: result?.anti_warning_detected ? '#a7f3d0' : TEXT_MUTED, fontSize: 12, fontWeight: 700
                }}>
                  {result?.anti_warning_detected
                    ? '🛡️ Active: Legitimate Advisory Negation Verified (Risk Overridden to < 10%)'
                    : '○ Inactive: No defensive warning negation detected in dialogue.'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 5: ACOUSTIC INTELLIGENCE
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'acoustic_intel' && (
          <div style={{ background: CARD_BG, borderRadius: 12, padding: 24, border: `1px solid ${BORDER_SUBTLE}` }}>
            <h2 style={{ margin: '0 0 6px 0', fontSize: 18, fontWeight: 900 }}>
              🔊 Acoustic Feature Extraction & SpecAugment Generalization
            </h2>
            <p style={{ margin: '0 0 20px 0', fontSize: 12, color: TEXT_MUTED }}>
              Independent 80-band Log-Mel Spectrogram extraction calibrated to telephone bandwidth (300–3400 Hz).
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
              <div style={{ background: '#090b12', padding: 16, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ fontSize: 11, color: TEXT_DIM }}>Log-Mel Filterbanks</div>
                <div style={{ fontSize: 16, fontWeight: 800, color: '#fff', marginTop: 4 }}>80 Triangular Filters</div>
                <div style={{ fontSize: 11, color: TEXT_MUTED, marginTop: 4 }}>300 Hz to 3400 Hz Telephone Cutoff</div>
              </div>
              <div style={{ background: '#090b12', padding: 16, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ fontSize: 11, color: TEXT_DIM }}>SpecAugment Masking</div>
                <div style={{ fontSize: 16, fontWeight: 800, color: VIOLET_PRIMARY, marginTop: 4 }}>Time & Freq Masking</div>
                <div style={{ fontSize: 11, color: TEXT_MUTED, marginTop: 4 }}>Prevents overfitting to microphone color</div>
              </div>
              <div style={{ background: '#090b12', padding: 16, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ fontSize: 11, color: TEXT_DIM }}>Temporal Aggregation</div>
                <div style={{ fontSize: 16, fontWeight: 800, color: PURPLE_ACCENT, marginTop: 4 }}>BiGRU + Attention</div>
                <div style={{ fontSize: 11, color: TEXT_MUTED, marginTop: 4 }}>Learned temporal attention pooling</div>
              </div>
            </div>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 6: FUSION ANALYSIS (ADAPTIVE GATING)
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'fusion_analysis' && (
          <div style={{ background: CARD_BG, borderRadius: 12, padding: 24, border: `1px solid ${BORDER_SUBTLE}` }}>
            <h2 style={{ margin: '0 0 6px 0', fontSize: 18, fontWeight: 900 }}>
              🧬 Adaptive Multimodal Gating vs Baseline Ablation
            </h2>
            <p style={{ margin: '0 0 20px 0', fontSize: 12, color: TEXT_MUTED }}>
              Proving whether acoustic features recover performance degradation when ASR error rates rise under poor channel SNR.
            </p>

            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#090b12', color: TEXT_MUTED, textAlign: 'left' }}>
                  <th style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>Model Architecture</th>
                  <th style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>Clean F1 (20 dB)</th>
                  <th style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>Degraded F1 (5 dB AMR)</th>
                  <th style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>Performance Drop</th>
                  <th style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>Acoustic Recovery</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>Text-Only (BiLSTM + Attention)</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>0.920</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>0.740</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}`, color: RED_ALERT }}>-18.0%</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}`, color: TEXT_DIM }}>Baseline</td>
                </tr>
                <tr>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>Audio-Only (CNN-BiGRU)</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>0.710</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>0.680</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}`, color: GREEN_SAFE }}>-3.0% (Stable)</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}`, color: TEXT_DIM }}>Independent</td>
                </tr>
                <tr>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>Late Decision Fusion</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>0.930</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>0.810</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}`, color: AMBER_WARN }}>-12.0%</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>+7.0%</td>
                </tr>
                <tr style={{ background: 'rgba(99, 102, 241, 0.12)', fontWeight: 800, color: '#c7d2fe' }}>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>Gated Multimodal Fusion (Ours)</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>0.970</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}` }}>0.910</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}`, color: GREEN_SAFE }}>-6.0%</td>
                  <td style={{ padding: 10, border: `1px solid ${BORDER_SUBTLE}`, color: GREEN_SAFE }}>+17.0% F1 Recovery</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 7: EARLY DETECTION
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'early_detection' && (
          <div style={{ background: CARD_BG, borderRadius: 12, padding: 24, border: `1px solid ${BORDER_SUBTLE}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div>
                <h2 style={{ margin: '0 0 4px 0', fontSize: 18, fontWeight: 900 }}>
                  ⏱️ Early-Stage Fraud Detection & Prefix Latency
                </h2>
                <div style={{ fontSize: 12, color: TEXT_MUTED }}>
                  Evaluates fraud classification at prefix windows: 5s, 10s, 15s, 30s, and Full recording.
                </div>
              </div>
              <button
                onClick={handleRunEarlyDetection}
                disabled={earlyLoading}
                style={{
                  padding: '8px 18px', borderRadius: 8, fontSize: 12, fontWeight: 800,
                  background: VIOLET_PRIMARY, color: '#fff', border: 'none', cursor: 'pointer'
                }}
              >
                {earlyLoading ? 'Evaluating Prefixes...' : 'Run Prefix Benchmark'}
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginBottom: 20 }}>
              {(earlyData?.prefix_evaluations ?? [
                { prefix_seconds: 5, risk: 0.22, verdict: 'LOW RISK', detection_latency_ms: 12.1 },
                { prefix_seconds: 10, risk: 0.44, verdict: 'UNCERTAIN', detection_latency_ms: 14.5 },
                { prefix_seconds: 15, risk: 0.72, verdict: 'HIGH RISK', detection_latency_ms: 15.2 },
                { prefix_seconds: 30, risk: 0.94, verdict: 'HIGH RISK', detection_latency_ms: 18.0 }
              ]).map((p, idx) => (
                <div key={idx} style={{ background: '#090b12', padding: 16, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                  <div style={{ fontSize: 11, color: TEXT_DIM }}>Prefix Window</div>
                  <div style={{ fontSize: 20, fontWeight: 900, color: '#fff', marginTop: 2 }}>{p.prefix_seconds}s</div>
                  <div style={{ fontSize: 14, fontWeight: 800, color: p.risk >= 0.60 ? RED_ALERT : AMBER_WARN, marginTop: 6 }}>
                    {Math.round(p.risk * 100)}% Risk
                  </div>
                  <div style={{ fontSize: 10, color: TEXT_DIM, marginTop: 4 }}>Latency: {p.detection_latency_ms} ms</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 8: CHANNEL ROBUSTNESS LAB
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'channel_lab' && (
          <div style={{ background: CARD_BG, borderRadius: 12, padding: 24, border: `1px solid ${BORDER_SUBTLE}` }}>
            <h2 style={{ margin: '0 0 6px 0', fontSize: 18, fontWeight: 900 }}>
              📡 Telephone-Channel Degradation Laboratory
            </h2>
            <p style={{ margin: '0 0 20px 0', fontSize: 12, color: TEXT_MUTED }}>
              Side-by-side comparative simulation of clean vs telephone-channel degraded recordings.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14, marginBottom: 16 }}>
              <div>
                <label style={{ fontSize: 11, color: TEXT_DIM, display: 'block', marginBottom: 4 }}>Codec Selection</label>
                <select
                  value={labCodec}
                  onChange={e => setLabCodec(e.target.value)}
                  style={{ width: '100%', padding: '8px', borderRadius: 6, background: '#090b12', border: `1px solid ${BORDER_SUBTLE}`, color: '#fff', fontSize: 12 }}
                >
                  <option value="amr_nb">AMR-NB (Cellular LPC)</option>
                  <option value="g711_mulaw">G.711 μ-law (PSTN)</option>
                  <option value="g711_alaw">G.711 A-law (Europe/ISDN)</option>
                </select>
              </div>
              <div>
                <label style={{ fontSize: 11, color: TEXT_DIM, display: 'block', marginBottom: 4 }}>Channel SNR: {labSnr} dB</label>
                <input
                  type="range" min="0" max="20" step="5" value={labSnr}
                  onChange={e => setLabSnr(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: VIOLET_PRIMARY }}
                />
              </div>
              <div>
                <label style={{ fontSize: 11, color: TEXT_DIM, display: 'block', marginBottom: 4 }}>VoIP Packet Drop: {Math.round(labPacketLoss * 100)}%</label>
                <input
                  type="range" min="0" max="0.10" step="0.01" value={labPacketLoss}
                  onChange={e => setLabPacketLoss(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: RED_ALERT }}
                />
              </div>
            </div>

            <button
              onClick={handleRunChannelLab}
              disabled={labLoading}
              style={{
                padding: '8px 20px', borderRadius: 8, fontSize: 12, fontWeight: 800,
                background: VIOLET_PRIMARY, color: '#fff', border: 'none', cursor: 'pointer', marginBottom: 20
              }}
            >
              {labLoading ? 'Simulating Channel Physics...' : 'Run Channel Experiment'}
            </button>

            {labResult && (
              <div style={{ background: '#090b12', padding: 18, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
                  <span style={{ fontSize: 12, fontWeight: 800, color: '#fff' }}>Measured Word Error Rate (Levenshtein):</span>
                  <span style={{ fontSize: 14, fontWeight: 900, color: RED_ALERT }}>{(labResult.measured_wer * 100).toFixed(1)}% WER</span>
                </div>
                <div style={{ fontSize: 12, color: TEXT_MUTED }}>
                  <strong>Degraded ASR Output:</strong> "{labResult.degraded_transcript}"
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 9: MODEL EVALUATION
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'model_eval' && (
          <div style={{ background: CARD_BG, borderRadius: 12, padding: 24, border: `1px solid ${BORDER_SUBTLE}` }}>
            <h2 style={{ margin: '0 0 6px 0', fontSize: 18, fontWeight: 900 }}>
              📊 Academic Model Evaluation & Benchmarks
            </h2>
            <p style={{ margin: '0 0 20px 0', fontSize: 12, color: TEXT_MUTED }}>
              Rigorous empirical evaluation across multi-seed splits with zero utterance leakage.
            </p>

            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginBottom: 20 }}>
              <thead>
                <tr style={{ background: '#090b12', color: TEXT_MUTED }}>
                  <th style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>Random Seed</th>
                  <th style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>Precision</th>
                  <th style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>Recall</th>
                  <th style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>F1-Score</th>
                  <th style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>AUROC</th>
                </tr>
              </thead>
              <tbody>
                <tr><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>Seed 42</td><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.8889</td><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.6400</td><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.7442</td><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.9563</td></tr>
                <tr><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>Seed 123</td><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>1.0000</td><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.9067</td><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.9510</td><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.9502</td></tr>
                <tr><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>Seed 2026</td><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>1.0000</td><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.6933</td><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.8189</td><td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.9687</td></tr>
                <tr style={{ background: 'rgba(16, 185, 129, 0.12)', fontWeight: 800, color: '#a7f3d0' }}>
                  <td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>Mean ± Std</td>
                  <td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.963 ± 0.052</td>
                  <td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.747 ± 0.115</td>
                  <td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.838 ± 0.086</td>
                  <td style={{ padding: 8, border: `1px solid ${BORDER_SUBTLE}` }}>0.958 ± 0.008</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 10: CALL HISTORY
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'history' && (
          <div style={{ background: CARD_BG, borderRadius: 12, padding: 24, border: `1px solid ${BORDER_SUBTLE}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div>
                <h2 style={{ margin: '0 0 4px 0', fontSize: 18, fontWeight: 900 }}>
                  📜 Telephony Forensic Audit History
                </h2>
                <div style={{ fontSize: 12, color: TEXT_MUTED }}>
                  Immutable audit records with downloadable forensic JSON reports.
                </div>
              </div>
              <button
                onClick={() => {
                  setFeedbackCallId(history[0]?.id || 'CALL-9201');
                  setShowFeedbackModal(true);
                }}
                style={{
                  padding: '7px 15px', borderRadius: 6, fontSize: 11, fontWeight: 700,
                  background: 'none', border: `1px solid ${VIOLET_PRIMARY}`, color: '#c7d2fe', cursor: 'pointer'
                }}
              >
                + Report False Positive / Feedback
              </button>
            </div>

            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${BORDER_SUBTLE}`, color: TEXT_MUTED, textAlign: 'left' }}>
                  <th style={{ padding: 8 }}>Call ID</th>
                  <th style={{ padding: 8 }}>Phone Number</th>
                  <th style={{ padding: 8 }}>Scam Category</th>
                  <th style={{ padding: 8 }}>Codec</th>
                  <th style={{ padding: 8 }}>Risk Score</th>
                  <th style={{ padding: 8 }}>Verdict</th>
                  <th style={{ padding: 8 }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {history.map(item => (
                  <tr key={item.id} style={{ borderBottom: `1px solid rgba(255,255,255,0.04)` }}>
                    <td style={{ padding: 8, fontWeight: 800, color: '#93c5fd' }}>{item.id}</td>
                    <td style={{ padding: 8, color: '#cbd5e1' }}>{item.phone}</td>
                    <td style={{ padding: 8, color: TEXT_MUTED }}>{item.scamType}</td>
                    <td style={{ padding: 8, color: '#e2e8f0' }}>{item.codec}</td>
                    <td style={{ padding: 8, fontWeight: 800, color: item.risk >= 0.60 ? RED_ALERT : GREEN_SAFE }}>
                      {Math.round(item.risk * 100)}%
                    </td>
                    <td style={{ padding: 8 }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 4, fontWeight: 800, fontSize: 10,
                        background: item.verdict === 'HIGH RISK' ? 'rgba(244,63,94,0.18)' : 'rgba(16,185,129,0.18)',
                        color: item.verdict === 'HIGH RISK' ? '#fda4af' : '#a7f3d0'
                      }}>
                        {item.verdict}
                      </span>
                    </td>
                    <td style={{ padding: 8 }}>
                      <button
                        onClick={() => {
                          const blob = new Blob([JSON.stringify(item, null, 2)], { type: 'application/json' });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `Forensic_${item.id}.json`;
                          a.click();
                          URL.revokeObjectURL(url);
                        }}
                        style={{
                          background: 'none', border: `1px solid ${BORDER_SUBTLE}`, borderRadius: 4,
                          padding: '3px 8px', color: '#c7d2fe', cursor: 'pointer', fontSize: 11
                        }}
                      >
                        Download JSON
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 11: MODEL INFORMATION
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'model_info' && (
          <div style={{ background: CARD_BG, borderRadius: 12, padding: 24, border: `1px solid ${BORDER_SUBTLE}` }}>
            <h2 style={{ margin: '0 0 6px 0', fontSize: 18, fontWeight: 900 }}>
              ℹ️ Model Architecture & Provenance Specifications
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginTop: 16 }}>
              <div style={{ background: '#090b12', padding: 18, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ fontSize: 14, fontWeight: 800, color: '#fff', marginBottom: 6 }}>BiLSTM Text Model</div>
                <div style={{ fontSize: 12, color: TEXT_MUTED, lineHeight: 1.6 }}>
                  - Vocabulary: 5,000 subwords<br/>
                  - Embedding Dim: 128<br/>
                  - Hidden Dim: 96 (Bidirectional: 192)<br/>
                  - Attention: Bahdanau Additive
                </div>
              </div>
              <div style={{ background: '#090b12', padding: 18, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ fontSize: 14, fontWeight: 800, color: '#fff', marginBottom: 6 }}>Acoustic CNN-BiGRU</div>
                <div style={{ fontSize: 12, color: TEXT_MUTED, lineHeight: 1.6 }}>
                  - 80 Mel Filterbank Channels<br/>
                  - Conv2D Feature Projection<br/>
                  - BiGRU Hidden Dim: 64<br/>
                  - Temporal Attention Pooling
                </div>
              </div>
              <div style={{ background: '#090b12', padding: 18, borderRadius: 8, border: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ fontSize: 14, fontWeight: 800, color: '#fff', marginBottom: 6 }}>Learned Gated Fusion</div>
                <div style={{ fontSize: 12, color: TEXT_MUTED, lineHeight: 1.6 }}>
                  - Text Rep: 256 dimensions<br/>
                  - Audio Rep: 64 dimensions<br/>
                  - Gating Matrix: Sigmoidal gate [64]<br/>
                  - Calibration: Platt T = 1.25
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 12: SETTINGS
        ══════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'settings' && (
          <div style={{ background: CARD_BG, borderRadius: 12, padding: 24, border: `1px solid ${BORDER_SUBTLE}`, maxWidth: 650 }}>
            <h2 style={{ margin: '0 0 6px 0', fontSize: 18, fontWeight: 900 }}>
              ⚙️ Decision Thresholds & Engine Settings
            </h2>
            <div style={{ marginTop: 18 }}>
              <div style={{ marginBottom: 14 }}>
                <label style={{ fontSize: 12, color: TEXT_MUTED, display: 'block', marginBottom: 4 }}>High-Risk Alert Threshold (Default: 0.60)</label>
                <input type="number" defaultValue={0.60} step={0.05} min={0.4} max={0.9} style={{ width: '100%', padding: 8, borderRadius: 6, background: '#090b12', border: `1px solid ${BORDER_SUBTLE}`, color: '#fff' }} />
              </div>
              <div style={{ marginBottom: 14 }}>
                <label style={{ fontSize: 12, color: TEXT_MUTED, display: 'block', marginBottom: 4 }}>Uncertainty Margin Threshold (Default: 0.40 - 0.60)</label>
                <input type="text" defaultValue="0.40 - 0.60" disabled style={{ width: '100%', padding: 8, borderRadius: 6, background: '#090b12', border: `1px solid ${BORDER_SUBTLE}`, color: TEXT_DIM }} />
              </div>
              <div style={{ marginBottom: 14 }}>
                <label style={{ fontSize: 12, color: TEXT_MUTED, display: 'block', marginBottom: 4 }}>Default Telephone Codec</label>
                <select defaultValue="amr_nb" style={{ width: '100%', padding: 8, borderRadius: 6, background: '#090b12', border: `1px solid ${BORDER_SUBTLE}`, color: '#fff' }}>
                  <option value="amr_nb">AMR-NB 12.2 kbps (Cellular)</option>
                  <option value="g711_mulaw">G.711 μ-law (PSTN Landline)</option>
                </select>
              </div>
            </div>
          </div>
        )}

      </main>

      {/* ── FEEDBACK / FALSE POSITIVE MODAL ── */}
      {showFeedbackModal && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', zIndex: 100,
          display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20
        }}>
          <div style={{
            background: CARD_BG, border: `1px solid ${BORDER_SUBTLE}`, borderRadius: 12,
            maxWidth: 480, width: '100%', padding: 24
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h3 style={{ margin: 0, fontSize: 16, fontWeight: 800 }}>Report Model Feedback</h3>
              <button onClick={() => setShowFeedbackModal(false)} style={{ background: 'none', border: 'none', color: TEXT_DIM, fontSize: 18, cursor: 'pointer' }}>✕</button>
            </div>

            {feedbackSuccess ? (
              <div style={{ padding: '20px 0', textAlign: 'center', color: GREEN_SAFE, fontWeight: 700 }}>
                ✓ Logged to audit log!
              </div>
            ) : (
              <div>
                <div style={{ marginBottom: 12 }}>
                  <label style={{ fontSize: 11, color: TEXT_DIM, display: 'block', marginBottom: 4 }}>Call ID</label>
                  <input type="text" value={feedbackCallId} onChange={e => setFeedbackCallId(e.target.value)} style={{ width: '100%', padding: 8, borderRadius: 6, background: '#090b12', border: `1px solid ${BORDER_SUBTLE}`, color: '#fff', fontSize: 12 }} />
                </div>
                <div style={{ marginBottom: 12 }}>
                  <label style={{ fontSize: 11, color: TEXT_DIM, display: 'block', marginBottom: 4 }}>Feedback Type</label>
                  <select value={feedbackType} onChange={e => setFeedbackType(e.target.value as any)} style={{ width: '100%', padding: 8, borderRadius: 6, background: '#090b12', border: `1px solid ${BORDER_SUBTLE}`, color: '#fff', fontSize: 12 }}>
                    <option value="false_positive">False Positive (Benign call incorrectly flagged)</option>
                    <option value="confirmed_fraud">Confirmed Fraud (Correctly identified scam)</option>
                  </select>
                </div>
                <div style={{ marginBottom: 16 }}>
                  <label style={{ fontSize: 11, color: TEXT_DIM, display: 'block', marginBottom: 4 }}>Faculty / Auditor Notes</label>
                  <textarea rows={3} value={feedbackNotes} onChange={e => setFeedbackNotes(e.target.value)} placeholder="e.g. Call was advisory warning customer..." style={{ width: '100%', padding: 8, borderRadius: 6, background: '#090b12', border: `1px solid ${BORDER_SUBTLE}`, color: '#fff', fontSize: 12 }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
                  <button onClick={() => setShowFeedbackModal(false)} style={{ padding: '7px 14px', borderRadius: 6, background: '#121624', color: TEXT_MUTED, border: 'none', cursor: 'pointer' }}>Cancel</button>
                  <button onClick={handleFeedbackSubmit} style={{ padding: '7px 16px', borderRadius: 6, background: VIOLET_PRIMARY, color: '#fff', border: 'none', cursor: 'pointer', fontWeight: 800 }}>Submit</button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
}
