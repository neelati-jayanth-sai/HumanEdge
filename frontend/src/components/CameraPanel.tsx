"use client";

import type React from "react";
import type { DebugPayload } from "../lib/types";

interface Props {
  running: boolean;
  handTracked: boolean;
  currentGestureLabel: string | null;
  currentConfidence: number;
  clientDetectionActive: boolean;
  debug: DebugPayload | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
}

export default function CameraPanel({
  running,
  handTracked,
  currentGestureLabel,
  currentConfidence,
  clientDetectionActive,
  debug,
  videoRef,
  canvasRef,
}: Props) {
  const stabilityPct =
    debug && debug.stable_window_ms > 0
      ? Math.min(100, Math.round((debug.pending_elapsed_ms / debug.stable_window_ms) * 100))
      : 0;

  return (
    <>
      {/* Camera frame */}
      <div className={`cam-wrap${running ? " live" : ""}`}>
        <span className="cam-corner cam-corner-tl" aria-hidden />
        <span className="cam-corner cam-corner-tr" aria-hidden />
        <span className="cam-corner cam-corner-bl" aria-hidden />
        <span className="cam-corner cam-corner-br" aria-hidden />

        <video ref={videoRef} className="cam-video" muted playsInline />
        <canvas ref={canvasRef} hidden />

        {!running && (
          <div className="cam-idle">
            <span className="idle-icon">◎</span>
            <span className="idle-label">Camera offline</span>
          </div>
        )}

        {running && (
          <div className={`hand-badge${handTracked ? (currentGestureLabel ? " tracked" : " unknown") : " untracked"}`}>
            {handTracked
              ? currentGestureLabel
                ? "✋ Hand Detected"
                : "? Unknown gesture"
              : "◌ No Hand"}
          </div>
        )}

        {running && clientDetectionActive && currentGestureLabel && (
          <div className="live-gesture">
            <span className="live-label">{currentGestureLabel}</span>
            {currentConfidence > 0 && (
              <span className="live-conf">{(currentConfidence * 100).toFixed(0)}%</span>
            )}
          </div>
        )}
      </div>

      {/* Stability bar */}
      {debug?.pending_label && (
        <div className="stab-module">
          <div className="stab-meta">
            <span className="stab-gesture">{debug.pending_label}</span>
            <span className="stab-time">{debug.pending_elapsed_ms} / {debug.stable_window_ms} ms</span>
          </div>
          <div className="stab-track">
            <div className="stab-fill" style={{ width: `${stabilityPct}%` }} />
          </div>
        </div>
      )}
    </>
  );
}
