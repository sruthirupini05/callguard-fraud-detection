import React from "react";
import type { AnalysisResult } from "./api";

export interface FlaggedPhrase {
  text: string;
  reason?: string;
}

export interface PhoneAlertProps {
  result?: AnalysisResult;
  phoneNumber?: string;
  riskPercent?: number;
  flaggedAt?: string;
  flaggedPhrases?: (string | FlaggedPhrase)[];
  onBack: () => void;
}

/**
 * PhoneAlert — simulated on-device incoming call screen shown during live analysis.
 *
 * NOTE: "End call" and "Report number" are DEMO actions only.
 * This application does not integrate with any real telephony system.
 */
const PhoneAlert: React.FC<PhoneAlertProps> = ({
  result,
  phoneNumber = "+91 98XXX XXX21",
  riskPercent,
  flaggedAt,
  flaggedPhrases,
  onBack,
}) => {
  const calculatedRisk =
    riskPercent ?? (result ? Math.round(result.risk * 100) : 91);
  const calculatedFlaggedAt =
    flaggedAt ??
    (result?.first_alert_seconds
      ? `${result.first_alert_seconds}s`
      : "0:48 first alert");

  const phrases: FlaggedPhrase[] =
    flaggedPhrases && flaggedPhrases.length > 0
      ? flaggedPhrases.map((p) =>
          typeof p === "string" ? { text: p, reason: "Identified scam keyword" } : p
        )
      : (result?.flagged_phrases ?? []).map((p) => ({
          text: p,
          reason: "Urgent social engineering pressure",
        }));

  // Fallback phrases if none detected
  if (phrases.length === 0) {
    phrases.push(
      { text: "Urgent tone in voice", reason: "Elevated pitch and stress" },
      { text: "Asked for OTP", reason: "Unauthorized credential solicitation" },
      { text: "Pressure to keep secret", reason: "Coercion tactic" }
    );
  }

  const riskLabel =
    calculatedRisk >= 75 ? "High risk" : calculatedRisk >= 40 ? "Medium risk" : "Low risk";

  return (
    <div
      className="phone-alert"
      role="dialog"
      aria-modal="true"
      aria-label="Fraud call alert"
      style={{
        background: "#14213D",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        padding: "0",
        fontFamily: "'Inter', sans-serif",
        color: "#ffffff",
        maxWidth: 390,
        margin: "0 auto",
      }}
    >
      {/* ── Status bar area ── */}
      <div
        style={{
          background: "#0d1726",
          padding: "12px 20px 8px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: 12,
          color: "#8899aa",
        }}
      >
        <span>Incoming call · 00:48</span>
        <button
          onClick={onBack}
          style={{
            background: "none",
            border: "none",
            color: "#8899aa",
            cursor: "pointer",
            fontSize: 12,
            padding: "4px 8px",
            minHeight: 44,
            minWidth: 44,
          }}
          aria-label="Return to analyst dashboard"
        >
          ← Back
        </button>
      </div>

      {/* ── Phone number ── */}
      <div style={{ padding: "24px 24px 0", textAlign: "center" }}>
        <p style={{ fontSize: 13, color: "#8899aa", marginBottom: 4 }}>
          Incoming call
        </p>
        <h1
          style={{
            fontSize: 28,
            fontWeight: 700,
            fontFamily: "'Space Grotesk', sans-serif",
            letterSpacing: "0.04em",
            color: "#ffffff",
            margin: 0,
          }}
        >
          {phoneNumber}
        </h1>
      </div>

      {/* ── Fraud alert card ── */}
      <div style={{ padding: "16px 20px 0" }}>
        <div
          role="alert"
          style={{
            background: "#fff8f7",
            borderRadius: 12,
            padding: "16px 18px",
            border: "1.5px solid #B42318",
          }}
        >
          <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
            {/* Warning icon */}
            <span
              aria-hidden="true"
              style={{
                fontSize: 18,
                color: "#B42318",
                flexShrink: 0,
                marginTop: 2,
              }}
            >
              ⚠
            </span>
            <div>
              <h2
                style={{
                  fontSize: 16,
                  fontWeight: 700,
                  color: "#B42318",
                  margin: "0 0 6px",
                  fontFamily: "'Space Grotesk', sans-serif",
                }}
              >
                Likely scam call
              </h2>
              <p
                style={{
                  fontSize: 13,
                  color: "#4a1a18",
                  margin: "0 0 10px",
                  lineHeight: 1.5,
                }}
              >
                The caller is asking for an OTP and urging you not to tell anyone. Banks never do this.
              </p>
              <p
                style={{
                  fontSize: 12,
                  color: "#7a2e28",
                  margin: 0,
                  fontWeight: 500,
                }}
              >
                Risk {calculatedRisk}% · {riskLabel} · Flagged at {calculatedFlaggedAt}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* ── Why we flagged it ── */}
      <div style={{ padding: "16px 20px 0" }}>
        <div
          style={{
            background: "#1e2f4d",
            borderRadius: 10,
            padding: "14px 16px",
          }}
        >
          <h3
            style={{
              fontSize: 12,
              fontWeight: 600,
              color: "#8899aa",
              margin: "0 0 10px",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            Why we flagged it
          </h3>
          <ul style={{ margin: 0, padding: 0, listStyle: "none" }}>
            {phrases.map((phrase, idx) => (
              <li
                key={idx}
                style={{
                  display: "flex",
                  alignItems: "baseline",
                  gap: 8,
                  marginBottom: idx < phrases.length - 1 ? 8 : 0,
                }}
              >
                <span
                  aria-hidden="true"
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "#F5A524",
                    flexShrink: 0,
                    marginTop: 5,
                    display: "inline-block",
                  }}
                />
                <span style={{ fontSize: 13, color: "#c8d8e8" }}>
                  <span style={{ color: "#F5A524", fontWeight: 600 }}>
                    {phrase.text}
                  </span>
                  {phrase.reason ? (
                    <span style={{ color: "#8899aa" }}>
                      {" "}
                      — {phrase.reason}
                    </span>
                  ) : null}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Spacer */}
      <div style={{ flex: 1, minHeight: 24 }} />

      {/* ── Action buttons ── */}
      <div style={{ padding: "0 20px 32px" }}>
        <p
          style={{
            fontSize: 11,
            color: "#556677",
            textAlign: "center",
            marginBottom: 14,
          }}
        >
          Demo only — buttons do not affect real calls
        </p>

        <button
          onClick={() => {
            alert(
              "DEMO: In a real integration, this would end the call and log it as fraud."
            );
          }}
          style={{
            width: "100%",
            background: "#B42318",
            color: "#ffffff",
            border: "none",
            borderRadius: 10,
            padding: "14px 0",
            fontSize: 16,
            fontWeight: 700,
            cursor: "pointer",
            marginBottom: 10,
            minHeight: 52,
            fontFamily: "'Inter', sans-serif",
          }}
          aria-label="End call (demo only)"
        >
          End call
        </button>

        <button
          onClick={() => {
            alert(
              "DEMO: In a real integration, this would report the number to the telecom authority."
            );
          }}
          style={{
            width: "100%",
            background: "transparent",
            color: "#c8d8e8",
            border: "1.5px solid #3a5070",
            borderRadius: 10,
            padding: "13px 0",
            fontSize: 15,
            fontWeight: 600,
            cursor: "pointer",
            minHeight: 52,
            fontFamily: "'Inter', sans-serif",
          }}
          aria-label="Report number (demo only)"
        >
          Report number
        </button>
      </div>
    </div>
  );
};

export default PhoneAlert;
