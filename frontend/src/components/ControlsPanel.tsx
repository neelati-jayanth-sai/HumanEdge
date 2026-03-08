"use client";

interface Props {
  running: boolean;
  speechEnabled: boolean;
  autoClear: boolean;
  guideOpen: boolean;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  onToggleSpeech: () => void;
  onToggleAutoClear: () => void;
  onToggleGuide: () => void;
  onClearHistory: () => void;
}

export default function ControlsPanel({
  running,
  speechEnabled,
  autoClear,
  guideOpen,
  onStart,
  onStop,
  onReset,
  onToggleSpeech,
  onToggleAutoClear,
  onToggleGuide,
  onClearHistory,
}: Props) {
  return (
    <div className="ctrl-zone">
      <div className="ctrl-primary">
        <button
          type="button"
          className="ctrl-btn start"
          onClick={onStart}
          disabled={running}
        >
          <span className="ctrl-icon">▶</span> Start
        </button>
        <button
          type="button"
          className="ctrl-btn stop"
          onClick={onStop}
          disabled={!running}
        >
          <span className="ctrl-icon">■</span> Stop
        </button>
        <button type="button" className="ctrl-btn" onClick={onReset}>
          <span className="ctrl-icon">↺</span> Clear
        </button>
      </div>
      <div className="ctrl-secondary">
        <button
          type="button"
          className={`ctrl-tog${speechEnabled ? " on" : ""}`}
          onClick={onToggleSpeech}
        >
          {speechEnabled ? "🔊" : "🔇"} Speech
        </button>
        <button
          type="button"
          className={`ctrl-tog${autoClear ? " on" : ""}`}
          onClick={onToggleAutoClear}
          title="Auto-clear tokens after each sentence"
        >
          Auto-clear
        </button>
        <button
          type="button"
          className={`ctrl-tog${guideOpen ? " on" : ""}`}
          onClick={onToggleGuide}
        >
          Guide
        </button>
        <button
          type="button"
          className="ctrl-tog danger"
          onClick={onClearHistory}
        >
          Clear Hist
        </button>
      </div>
    </div>
  );
}
