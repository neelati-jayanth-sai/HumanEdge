"use client";

import { useRef } from "react";
import { SCENARIOS } from "../lib/constants";
import type { ScenarioKey } from "../lib/types";

interface Props {
  language: "en" | "hi";
  accent: string;
  scenario: ScenarioKey;
  customScenario: string;
  onLanguageChange: (lang: "en" | "hi") => void;
  onAccentChange: (accent: string) => void;
  onScenarioChange: (s: ScenarioKey) => void;
  onCustomScenarioCommit: (text: string) => void;
}

export default function SettingsStrip({
  language,
  accent,
  scenario,
  customScenario,
  onLanguageChange,
  onAccentChange,
  onScenarioChange,
  onCustomScenarioCommit,
}: Props) {
  const customRef = useRef<HTMLInputElement>(null);

  return (
    <div className="settings-strip">
      {/* ── Language ── */}
      <div className="settings-group">
        <span className="settings-label">Language</span>
        <div className="pill-group">
          {([
            { value: "en", label: "English" },
            { value: "hi", label: "हिंदी" },
          ] as const).map((opt) => (
            <button
              key={opt.value}
              type="button"
              className={`pill-opt${language === opt.value ? " active" : ""}`}
              onClick={() => onLanguageChange(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Accent (English only) ── */}
      {language === "en" && (
        <div className="settings-group">
          <span className="settings-label">Accent</span>
          <div className="pill-group">
            {([
              { value: "en-US", label: "US" },
              { value: "en-GB", label: "UK" },
              { value: "en-IN", label: "Indian" },
            ] as const).map((opt) => (
              <button
                key={opt.value}
                type="button"
                className={`pill-opt${accent === opt.value ? " active" : ""}`}
                onClick={() => onAccentChange(opt.value)}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Scenario divider ── */}
      <div className="settings-divider" />

      {/* ── Scenario selector ── */}
      <div className="settings-group settings-group--full">
        <span className="settings-label">Context</span>
        <div className="scenario-group">
          {SCENARIOS.map((s) => (
            <button
              key={s.key}
              type="button"
              className={`scenario-pill${scenario === s.key ? " active" : ""}`}
              title={s.hint}
              onClick={() => {
                onScenarioChange(s.key as ScenarioKey);
                if (s.key === "custom") {
                  setTimeout(() => customRef.current?.focus(), 50);
                }
              }}
            >
              <span className="scenario-icon">{s.icon}</span>
              {s.label}
            </button>
          ))}

          {/* Custom text input — shown only when "custom" is selected */}
          {scenario === "custom" && (
            <input
              ref={customRef}
              type="text"
              className="scenario-custom-input"
              defaultValue={customScenario}
              placeholder="Describe your setting…"
              maxLength={120}
              onBlur={(e) => onCustomScenarioCommit(e.target.value.trim())}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.currentTarget.blur();
                }
              }}
            />
          )}
        </div>
      </div>

      {/* Active scenario badge (non-general, non-custom) */}
      {scenario !== "general" && (
        <div className="scenario-active-badge">
          {SCENARIOS.find((s) => s.key === scenario)?.icon}{" "}
          {scenario === "custom" && customScenario
            ? customScenario
            : SCENARIOS.find((s) => s.key === scenario)?.hint}
        </div>
      )}
    </div>
  );
}
