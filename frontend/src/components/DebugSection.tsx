"use client";

import type { DebugPayload } from "../lib/types";

interface Props {
  debug: DebugPayload | null;
  open: boolean;
  onToggle: () => void;
  clientDetectionActive: boolean;
  handTracked: boolean;
  currentGestureLabel: string | null;
  currentConfidence: number;
}

export default function DebugSection({
  debug,
  open,
  onToggle,
  clientDetectionActive,
  handTracked,
  currentGestureLabel,
  currentConfidence,
}: Props) {
  return (
    <div className="sec">
      <div
        className="debug-header"
        onClick={onToggle}
        role="button"
        aria-expanded={open}
      >
        <span className="debug-title-text">Debug Console</span>
        <span className="debug-mode-pill">
          {clientDetectionActive ? "browser" : "server"}
        </span>
        <span className="debug-caret">{open ? "▲" : "▼"}</span>
      </div>
      {open && (
        <div className="debug-body">
          <div className="debug-grid">
            <div className="debug-row">
              <span className="dbk">Backend</span>
              <span className="dbv">{debug?.backend ?? "—"}</span>
            </div>
            <div className="debug-row">
              <span className="dbk">Hand</span>
              <span className={`dbv ${handTracked ? "yes" : "no"}`}>
                {handTracked ? "Tracked" : "None"}
              </span>
            </div>
            <div className="debug-row">
              <span className="dbk">Raw Label</span>
              <span className="dbv">{debug?.raw_label ?? "—"}</span>
            </div>
            <div className="debug-row">
              <span className="dbk">Accepted</span>
              <span className="dbv highlight">{debug?.accepted_label ?? currentGestureLabel ?? "—"}</span>
            </div>
            <div className="debug-row">
              <span className="dbk">Confidence</span>
              <span className="dbv">
                {currentConfidence > 0 ? `${(currentConfidence * 100).toFixed(1)}%` : "—"}
              </span>
            </div>
            <div className="debug-row">
              <span className="dbk">Pending</span>
              <span className="dbv">{debug?.pending_label ?? "—"}</span>
            </div>
            <div className="debug-row">
              <span className="dbk">Stability</span>
              <span className="dbv">
                {debug ? `${debug.pending_elapsed_ms}/${debug.stable_window_ms}ms` : "—"}
              </span>
            </div>
            <div className="debug-row">
              <span className="dbk">Updated</span>
              <span className={`dbv ${debug?.token_updated ? "yes" : ""}`}>
                {debug?.token_updated ? "Yes" : "No"}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
