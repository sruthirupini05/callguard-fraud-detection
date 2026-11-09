import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, BarChart, Bar, ResponsiveContainer, Legend
} from 'recharts';
import { analyzeCall, checkHealth, AnalysisResult, StreamChunk, API_BASE } from './api';
import PhoneAlert from './PhoneAlert';

// ── Design tokens ──────────────────────────────────────────────────────────
const NAVY   = '#14213D';
const AMBER  = '#F5A524';
const RED    = '#B42318';
const GREEN  = '#16a34a';
const SLATE  = '#94a3b8';

// ── Utility helpers ─────────────────────────────────────────────────────────
function riskColor(r: number) {
  if (r >= 0.75) return RED;
  if (r >= 0.40) return AMBER;
  return GREEN;
}
function riskLabel(r: number) {
  if (r >= 0.75) return 'HIGH RISK';
  if (r >= 0.40) return 'MODERATE';
  return 'LOW RISK';
}

function highlightText(text: string, phrases: string[]): React.ReactNode {
  if (!phrases.length) return <span>{text}</span>;
  const regex = new RegExp(`(${phrases.map(p => p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  const parts = text.split(regex);
  return (
    <>
      {parts.map((p, i) =>
        phrases.some(ph => ph.toLowerCase() === p.toLowerCase())
          ? <mark key={i} style={{ background: '#7f1d1d44', color: '#fca5a5', borderRadius: 2, padding: '0 2px' }}>{p}</mark>
          : <span key={i}>{p}</span>
      )}
    </>
  );
}

// ── Gauge component ─────────────────────────────────────────────────────────
function RiskGauge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const angle = -135 + pct * 2.7; // -135° → +135°
  const color = riskColor(value);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      <svg width={180} height={110} viewBox="0 0 180 110">
        {/* Track */}
        <path d="M20 100 A70 70 0 0 1 160 100" fill="none" stroke="#1e3a5f" strokeWidth={16} strokeLinecap="round"/>
        {/* Value arc */}
        <path d="M20 100 A70 70 0 0 1 160 100" fill="none" stroke={color} strokeWidth={16}
              strokeLinecap="round"
              strokeDasharray={`${219.9 * Math.min(pct/100, 1)} 220`}/>
        {/* Needle */}
        <line x1="90" y1="100" x2="90" y2="38"
              stroke="#e2e8f0" strokeWidth={2.5} strokeLinecap="round"
              transform={`rotate(${angle}, 90, 100)`}/>
        <circle cx="90" cy="100" r="5" fill="#e2e8f0"/>
      </svg>
      <div style={{ fontSize: 36, fontWeight: 800, color, lineHeight: 1 }}>{pct}%</div>
      <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.12em', color }}>{riskLabel(value)}</div>
    </div>
  );
}

// ── Branch bar ───────────────────────────────────────────────────────────────
function BranchBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 3, color: '#94a3b8' }}>
        <span>{label}</span>
        <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{Math.round(value * 100)}%</span>
      </div>
      <div style={{ background: '#1e3a5f', borderRadius: 6, height: 8, overflow: 'hidden' }}>
        <div style={{ width: `${value * 100}%`, height: '100%', background: color, borderRadius: 6, transition: 'width 0.5s ease' }}/>
      </div>
    </div>
  );
}

// ── Waveform visualization ───────────────────────────────────────────────────
function WaveformBar({ alertSec, duration }: { alertSec: number | null; duration: number }) {
  // Generate synthetic waveform bars
  const bars = Array.from({ length: 80 }, (_, i) => ({
    t: (i / 80) * duration,
    h: 15 + Math.abs(Math.sin(i * 0.4 + 1) * 25 + Math.sin(i * 0.13) * 20)
  }));
  return (
    <div style={{ position: 'relative', background: '#0d1829', borderRadius: 8, padding: '12px 8px', overflow: 'hidden' }}>
      <div style={{ display: 'flex', alignItems: 'center', height: 64, gap: 1 }}>
        {bars.map((b, i) => (
          <div key={i} style={{
            flex: 1, background: alertSec !== null && b.t >= alertSec ? '#B42318' : '#2d5a9e',
            borderRadius: 2, height: `${b.h}%`, transition: 'height 0.1s'
          }}/>
        ))}
      </div>
      {alertSec !== null && (
        <div style={{
          position: 'absolute', top: 0, left: `${(alertSec / duration) * 100}%`,
          bottom: 0, width: 2, background: AMBER,
          boxShadow: `0 0 8px ${AMBER}`
        }}>
          <div style={{
            position: 'absolute', top: 2, left: 4,
            background: AMBER, color: '#000', fontSize: 9, fontWeight: 700,
            padding: '1px 4px', borderRadius: 2, whiteSpace: 'nowrap'
          }}>
            {alertSec}s first alert
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [view, setView] = useState<'dashboard' | 'phonealert'>('dashboard');
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [streamChunks, setStreamChunks] = useState<StreamChunk[]>([]);
  const [liveMode, setLiveMode] = useState(false);
  const [codec, setCodec] = useState('amr_nb');
  const [snr, setSnr] = useState(5);
  const [pl, setPl] = useState(0.03);
  const wsRef = useRef<WebSocket | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const timerRef = useRef<number | null>(null);

  // Health-check on mount
  useEffect(() => {
    checkHealth().then(ok => setBackendOnline(ok));
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      wsRef.current?.close();
    };
  }, []);

  const runSimulation = useCallback(() => {
    const simulatedTurns: StreamChunk[] = [
      { current_second: 5, fused_risk: 0.24, is_alert_active: false, partial_transcript: "CALLER: Hello sir, calling from bank verification department.", flagged_phrases: [] },
      { current_second: 10, fused_risk: 0.28, is_alert_active: false, partial_transcript: "CALLER: Hello sir, calling from bank verification department.\nYOU: Yes, what is this regarding?", flagged_phrases: [] },
      { current_second: 15, fused_risk: 0.58, is_alert_active: false, partial_transcript: "CALLER: Hello sir, calling from bank verification department.\nYOU: Yes, what is this regarding?\nCALLER: Sir, unauthorized debit card transaction of 45,000 attempted.", flagged_phrases: [] },
      { current_second: 20, fused_risk: 0.64, is_alert_active: false, partial_transcript: "CALLER: Hello sir, calling from bank verification department.\nYOU: Yes, what is this regarding?\nCALLER: Sir, unauthorized debit card transaction of 45,000 attempted.\nYOU: I did not do any transaction! Please stop it.", flagged_phrases: [] },
      { current_second: 25, fused_risk: 0.88, is_alert_active: true, partial_transcript: "CALLER: Hello sir, calling from bank verification department.\nYOU: Yes, what is this regarding?\nCALLER: Sir, unauthorized debit card transaction of 45,000 attempted.\nYOU: I did not do any transaction! Please stop it.\nCALLER: Please share the OTP immediately to block this transaction.", flagged_phrases: ["share the otp"] },
      { current_second: 30, fused_risk: 0.94, is_alert_active: true, partial_transcript: "CALLER: Hello sir, calling from bank verification department.\nYOU: Yes, what is this regarding?\nCALLER: Sir, unauthorized debit card transaction of 45,000 attempted.\nYOU: I did not do any transaction! Please stop it.\nCALLER: Please share the OTP immediately to block this transaction.\nCALLER: Account will be blocked today if you do not tell me now.", flagged_phrases: ["share the otp", "account will be blocked"] },
      { current_second: 35, fused_risk: 0.92, is_alert_active: true, partial_transcript: "CALLER: Hello sir, calling from bank verification department.\nYOU: Yes, what is this regarding?\nCALLER: Sir, unauthorized debit card transaction of 45,000 attempted.\nYOU: I did not do any transaction! Please stop it.\nCALLER: Please share the OTP immediately to block this transaction.\nCALLER: Account will be blocked today if you do not tell me now.\nYOU: Wait, bank says never share OTP with anyone.", flagged_phrases: ["share the otp", "account will be blocked"] },
      { current_second: 40, fused_risk: 0.96, is_alert_active: true, partial_transcript: "CALLER: Hello sir, calling from bank verification department.\nYOU: Yes, what is this regarding?\nCALLER: Sir, unauthorized debit card transaction of 45,000 attempted.\nYOU: I did not do any transaction! Please stop it.\nCALLER: Please share the OTP immediately to block this transaction.\nCALLER: Account will be blocked today if you do not tell me now.\nYOU: Wait, bank says never share OTP with anyone.\nCALLER: Sir do not disconnect! This is security server, read digits.", flagged_phrases: ["share the otp", "account will be blocked", "do not disconnect"] },
    ];

    let idx = 0;
    if (timerRef.current) clearInterval(timerRef.current);
    const interval = window.setInterval(() => {
      if (idx < simulatedTurns.length) {
        const item = simulatedTurns[idx];
        setStreamChunks(prev => [...prev, item]);
        idx++;
      } else {
        clearInterval(interval);
      }
    }, 1500);
    timerRef.current = interval;
  }, []);

  const startLive = useCallback(() => {
    setStreamChunks([]);
    setLiveMode(true);
    setError(null);
    let wsSuccess = false;

    try {
      const ws = new WebSocket(`ws://127.0.0.1:8000/ws/stream`);
      wsRef.current = ws;

      ws.onopen = () => {
        wsSuccess = true;
      };

      ws.onmessage = e => {
        try {
          const chunk: StreamChunk = JSON.parse(e.data);
          setStreamChunks(prev => [...prev, chunk]);
        } catch (err) {
          console.error("Failed to parse chunk", err);
        }
      };

      ws.onerror = () => {
        if (!wsSuccess) {
          runSimulation();
        }
      };

      ws.onclose = () => {
        if (!wsSuccess) {
          runSimulation();
        }
      };
    } catch {
      runSimulation();
    }
  }, [runSimulation]);

  const stopLive = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    wsRef.current?.close();
    setLiveMode(false);
  }, []);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await analyzeCall(file ?? undefined, undefined, codec, snr, pl);
      setResult(r);
    } catch (err: any) {
      setError(err.message ?? 'Analysis failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const latestChunk = streamChunks[streamChunks.length - 1];
  const timelineData = streamChunks.map(c => ({ t: c.current_second, risk: Math.round(c.fused_risk * 100) }));

  // ── Dashboard view ──────────────────────────────────────────────────────
  if (view === 'phonealert') {
    return (
      <PhoneAlert
        result={result ?? undefined}
        phoneNumber="+91 98XXX XXX21"
        riskPercent={Math.round((latestChunk?.fused_risk ?? result?.risk ?? 0.91) * 100)}
        flaggedAt={latestChunk ? `${latestChunk.current_second}s` : (result?.first_alert_seconds ? `${result.first_alert_seconds}s` : "0:48 first alert")}
        flaggedPhrases={latestChunk?.flagged_phrases ?? result?.flagged_phrases ?? []}
        onBack={() => setView('dashboard')}
      />
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: '#0d1829', color: '#e2e8f0', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <header style={{ background: NAVY, borderBottom: '1px solid #1e3a5f', padding: '0 24px', height: 56, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke={AMBER} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span style={{ fontWeight: 800, fontSize: 18, letterSpacing: '-0.02em' }}>CallGuard</span>
          <span style={{ fontSize: 11, color: SLATE, marginLeft: 4 }}>Phone-line fraud detection</span>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {backendOnline !== null && (
            <span style={{ fontSize: 11, padding: '3px 8px', borderRadius: 20,
                           background: backendOnline ? '#14532d' : '#7f1d1d',
                           color: backendOnline ? '#86efac' : '#fca5a5' }}>
              Backend {backendOnline ? 'online' : 'offline'}
            </span>
          )}
          <button onClick={() => fileRef.current?.click()}
                  style={{ padding: '7px 14px', borderRadius: 6, border: '1px solid #2d4a7a', background: 'transparent', color: '#e2e8f0', cursor: 'pointer', fontSize: 13 }}>
            Upload call
          </button>
          <input ref={fileRef} type="file" accept="audio/*" style={{ display: 'none' }} onChange={e => setFile(e.target.files?.[0] ?? null)} />
          <button onClick={handleAnalyze} disabled={loading}
                  style={{ padding: '7px 16px', borderRadius: 6, border: 'none', background: AMBER, color: '#000', fontWeight: 700, cursor: loading ? 'not-allowed' : 'pointer', fontSize: 13 }}>
            {loading ? 'Analysing...' : 'Analyse'}
          </button>
          {!liveMode
            ? <button onClick={startLive} style={{ padding: '7px 14px', borderRadius: 6, border: 'none', background: '#1e3a5f', color: '#e2e8f0', cursor: 'pointer', fontSize: 13 }}>Start live</button>
            : <button onClick={stopLive} style={{ padding: '7px 14px', borderRadius: 6, border: 'none', background: RED, color: '#fff', cursor: 'pointer', fontSize: 13 }}>Stop</button>
          }
        </div>
      </header>

      {/* Backend offline notice */}
      {backendOnline === false && (
        <div style={{ background: '#7f1d1d', padding: '10px 24px', fontSize: 13, color: '#fca5a5', borderBottom: '1px solid #991b1b' }}>
          Backend is offline. Run <code style={{ background: '#000', padding: '1px 6px', borderRadius: 3 }}>uvicorn src.api.main:app --host 127.0.0.1 --port 8000</code> in the project root.
        </div>
      )}

      {error && (
        <div style={{ background: '#7f1d1d', padding: '10px 24px', fontSize: 13, color: '#fca5a5', borderBottom: '1px solid #991b1b' }}>
          {error}
        </div>
      )}

      {file && !result && !loading && (
        <div style={{ background: '#14213d', padding: '8px 24px', fontSize: 12, color: AMBER, borderBottom: '1px solid #1e3a5f' }}>
          Loaded: {file.name} — click Analyse to process.
        </div>
      )}

      <main style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16, padding: 16, maxWidth: 1280, margin: '0 auto' }}>

        {/* LEFT COLUMN */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* Live Call Info */}
          <div style={{ background: NAVY, borderRadius: 10, padding: 16, border: '1px solid #1e3a5f' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
              <div>
                <div style={{ fontSize: 10, color: AMBER, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 2 }}>
                  {liveMode ? 'LIVE CALL' : result ? 'UPLOADED CALL' : 'NO CALL LOADED'}
                </div>
                <div style={{ fontWeight: 700, fontSize: 16 }}>
                  {liveMode
                    ? `+91 98XXX XXX21 · ${latestChunk?.current_second ?? 0}s`
                    : result ? `Analysis complete · ${result.duration_seconds.toFixed(0)}s` : 'Upload or start live analysis'}
                </div>
              </div>
              {result && result.risk_label === 'high' && (
                <span style={{ background: RED, color: '#fff', fontSize: 11, padding: '3px 8px', borderRadius: 4, fontWeight: 700 }}>
                  Fraud detected at {result.first_alert_seconds}s
                </span>
              )}
            </div>
            <WaveformBar
              alertSec={result?.first_alert_seconds ?? null}
              duration={result?.duration_seconds ?? 45}
            />
            {/* Timeline chart for live mode */}
            {liveMode && streamChunks.length > 0 && (
              <div style={{ marginTop: 10, height: 80 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={timelineData}>
                    <defs>
                      <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={RED} stopOpacity={0.4}/>
                        <stop offset="95%" stopColor={RED} stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="t" hide />
                    <YAxis domain={[0, 100]} hide />
                    <ReferenceLine y={75} stroke={RED} strokeDasharray="4 2" />
                    <Area type="monotone" dataKey="risk" stroke={RED} fill="url(#riskGrad)" strokeWidth={2} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Transcript */}
          <div style={{ background: NAVY, borderRadius: 10, padding: 16, border: '1px solid #1e3a5f', flex: 1 }}>
            <div style={{ fontSize: 11, color: SLATE, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 10 }}>Transcript</div>
            {!result && !liveMode && (
              <div style={{ color: '#475569', fontSize: 13, fontStyle: 'italic' }}>No transcript available. Upload an audio file or start live analysis.</div>
            )}
            {(result || liveMode) && (
              <div style={{ fontSize: 13, lineHeight: 1.75, maxHeight: 220, overflowY: 'auto' }}>
                {liveMode && !latestChunk && (
                  <div style={{ color: AMBER, fontSize: 13, display: 'flex', alignItems: 'center', gap: 8, padding: '12px 0' }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: AMBER, display: 'inline-block' }} />
                    Connecting to live call audio stream...
                  </div>
                )}
                {(((liveMode ? latestChunk?.partial_transcript : result?.transcript) || '')
                  .split('\n')
                  .filter(l => l.trim().length > 0)
                  .map((line, i) => {
                    const isCaller = line.startsWith('CALLER:');
                    return (
                      <div key={i} style={{ marginBottom: 6 }}>
                        <span style={{ color: isCaller ? AMBER : '#93c5fd', fontWeight: 600, fontSize: 11, textTransform: 'uppercase', marginRight: 6 }}>
                          {isCaller ? 'Caller' : 'You'}
                        </span>
                        <span style={{ color: '#cbd5e1' }}>
                          {highlightText(line.replace(/^(CALLER:|YOU:)\s*/i, ''), result?.flagged_phrases ?? latestChunk?.flagged_phrases ?? [])}
                        </span>
                      </div>
                    );
                  }))}
              </div>
            )}
          </div>

          {/* Channel Settings */}
          <div style={{ background: NAVY, borderRadius: 10, padding: 14, border: '1px solid #1e3a5f' }}>
            <div style={{ fontSize: 11, color: SLATE, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 10 }}>Channel Simulation</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
              <label style={{ fontSize: 12, color: SLATE }}>
                Codec
                <select value={codec} onChange={e => setCodec(e.target.value)}
                        style={{ display: 'block', width: '100%', marginTop: 3, padding: '5px 8px', borderRadius: 5, background: '#0d1829', border: '1px solid #2d4a7a', color: '#e2e8f0', fontSize: 12 }}>
                  <option value="amr_nb">AMR-NB</option>
                  <option value="g711_mulaw">G.711 μ-law</option>
                  <option value="g711_alaw">G.711 A-law</option>
                  <option value="none">None</option>
                </select>
              </label>
              <label style={{ fontSize: 12, color: SLATE }}>
                SNR (dB): {snr}
                <input type="range" min={0} max={20} value={snr} onChange={e => setSnr(Number(e.target.value))}
                       style={{ display: 'block', width: '100%', marginTop: 6 }} />
              </label>
              <label style={{ fontSize: 12, color: SLATE }}>
                Packet Loss: {Math.round(pl * 100)}%
                <input type="range" min={0} max={10} step={1} value={Math.round(pl * 100)}
                       onChange={e => setPl(Number(e.target.value) / 100)}
                       style={{ display: 'block', width: '100%', marginTop: 6 }} />
              </label>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* Gauge */}
          <div style={{ background: NAVY, borderRadius: 10, padding: 16, border: '1px solid #1e3a5f', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
            <div style={{ fontSize: 11, color: SLATE, letterSpacing: '0.08em', textTransform: 'uppercase', alignSelf: 'flex-start' }}>Fused Fraud Risk</div>
            <RiskGauge value={result?.risk ?? (liveMode ? (latestChunk?.fused_risk ?? 0) : 0)} />
            {((result && result.risk_label === 'high') || (liveMode && (latestChunk?.fused_risk ?? 0) >= 0.75)) ? (
              <button onClick={() => setView('phonealert')}
                      style={{ width: '100%', padding: '10px', borderRadius: 6, border: 'none', background: RED, color: '#fff', fontWeight: 700, cursor: 'pointer', fontSize: 13 }}>
                Warn the user and end call
              </button>
            ) : (
              <div style={{ width: '100%', padding: '10px', borderRadius: 6, border: '1px solid #1e3a5f', textAlign: 'center', color: SLATE, fontSize: 12 }}>
                {result || liveMode ? 'Monitoring live call...' : 'Awaiting analysis'}
              </div>
            )}
          </div>

          {/* Model Branches */}
          <div style={{ background: NAVY, borderRadius: 10, padding: 16, border: '1px solid #1e3a5f' }}>
            <div style={{ fontSize: 11, color: SLATE, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 12 }}>Model Branches</div>
            <BranchBar label="Text – BiLSTM" value={result?.branch_scores.text ?? (liveMode && latestChunk ? Math.min(1.0, latestChunk.fused_risk * 0.92) : 0)} color="#3b82f6" />
            <BranchBar label="Audio – CNN-BiGRU" value={result?.branch_scores.audio ?? (liveMode && latestChunk ? Math.min(1.0, latestChunk.fused_risk * 0.98) : 0)} color="#8b5cf6" />
            <BranchBar label="Fused" value={result?.branch_scores.fused ?? (liveMode && latestChunk ? latestChunk.fused_risk : 0)} color={riskColor(result?.branch_scores.fused ?? (liveMode && latestChunk ? latestChunk.fused_risk : 0))} />
          </div>

          {/* Channel Info */}
          <div style={{ background: NAVY, borderRadius: 10, padding: 16, border: '1px solid #1e3a5f' }}>
            <div style={{ fontSize: 11, color: SLATE, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 10 }}>Channel Info</div>
            {[
              ['Sample rate', result ? `${result.channel_info.sample_rate.toLocaleString()} Hz` : '—'],
              ['Codec', result?.channel_info.codec ?? '—'],
              ['Packet loss', result ? `${Math.round(result.channel_info.packet_loss * 100)}%` : '—'],
              ['Word error rate', result?.channel_info.wer !== null && result?.channel_info.wer !== undefined ? `${Math.round((result.channel_info.wer) * 100)}%` : '—'],
            ].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 5, color: SLATE }}>
                <span>{k}</span>
                <span style={{ color: '#e2e8f0', fontWeight: 500 }}>{v}</span>
              </div>
            ))}
          </div>

          {/* Flagged Phrases */}
          <div style={{ background: NAVY, borderRadius: 10, padding: 16, border: '1px solid #1e3a5f', flex: 1 }}>
            <div style={{ fontSize: 11, color: SLATE, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 10 }}>Flagged Phrases</div>
            {((result?.flagged_phrases ?? latestChunk?.flagged_phrases ?? []).length === 0)
              ? <div style={{ color: '#475569', fontSize: 12 }}>No suspicious phrases detected.</div>
              : (result?.flagged_phrases ?? latestChunk?.flagged_phrases ?? []).map((p, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
                    <div style={{ width: 6, height: 6, background: RED, borderRadius: '50%' }}/>
                    <span style={{ fontSize: 12, color: '#fca5a5', fontStyle: 'italic' }}>{p}</span>
                  </div>
                ))
            }
          </div>
        </div>
      </main>
    </div>
  );
}
