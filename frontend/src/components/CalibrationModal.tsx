"use client";

interface Props {
  open: boolean;
  phase: "idle" | "sampling" | "done";
  progress: number;
  threshold: number;
  running: boolean;
  onClose: () => void;
  onStart: () => void;
}

export default function CalibrationModal({
  open,
  phase,
  progress,
  threshold,
  running,
  onClose,
  onStart,
}: Props) {
  if (!open) return null;

  return (
    <div className="calib-overlay" onClick={() => phase !== "sampling" && onClose()}>
      <div className="calib-modal" onClick={(e) => e.stopPropagation()}>
        <div className="calib-title">◎ Hand Calibration</div>

        {phase === "idle" && (
          <>
            <p className="calib-desc">
              Hold <strong>OPEN_PALM</strong> (hand spread, facing camera) steady while calibration runs.
              The system will tune detection sensitivity to your hand and lighting.
            </p>
            <div className="calib-actions">
              {running ? (
                <button type="button" className="calib-start-btn" onClick={onStart}>
                  Start Calibration
                </button>
              ) : (
                <p className="calib-warn">⚠ Start the camera first, then calibrate.</p>
              )}
              <button type="button" className="calib-close-btn" onClick={onClose}>
                Cancel
              </button>
            </div>
          </>
        )}

        {phase === "sampling" && (
          <>
            <p className="calib-desc">
              <strong>Hold OPEN_PALM still…</strong> sampling {progress}%
            </p>
            <div className="calib-progress-wrap">
              <div className="calib-progress-bar" style={{ width: `${progress}%` }} />
            </div>
          </>
        )}

        {phase === "done" && (
          <>
            <p className="calib-desc">
              Calibration complete. Detection threshold set to{" "}
              <strong>{(threshold * 100).toFixed(0)}%</strong>.
            </p>
            <div className="calib-actions">
              <button type="button" className="calib-start-btn" onClick={onClose}>
                Done
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
