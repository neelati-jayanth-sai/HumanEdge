"use client";

interface Props {
  sentence: string;
  speakPending: boolean;
  hasSent: boolean;
  speakCountdownPct: number;
  onSpeakNow: () => void;
  onCancel: () => void;
  onNewSentence: () => void;
}

export default function SentenceSection({
  sentence,
  speakPending,
  hasSent,
  speakCountdownPct,
  onSpeakNow,
  onCancel,
  onNewSentence,
}: Props) {
  return (
    <div className="sec">
      <div className="sec-head">
        <span className="sec-title">Generated Sentence</span>
        {sentence && !speakPending && !hasSent && (
          <span className="sec-badge streaming">Generating…</span>
        )}
      </div>
      <div className="sec-body">
        <div className="sent-display">
          {sentence ? (
            <>
              {sentence}
              {!speakPending && !hasSent && sentence && (
                <span className="stream-cursor" aria-hidden />
              )}
            </>
          ) : (
            <span className="sent-placeholder">Awaiting signs…</span>
          )}
        </div>

        {/* Countdown bar */}
        {speakPending && (
          <div className="speak-countdown-wrap">
            <div className="speak-countdown-bar" style={{ width: `${speakCountdownPct}%` }} />
          </div>
        )}

        {sentence && (
          <div className="sent-actions">
            <button
              type="button"
              className="speak-btn"
              onClick={onSpeakNow}
            >
              🔊 Speak Now
            </button>
            {speakPending && (
              <button
                type="button"
                className="cancel-btn"
                onClick={onCancel}
              >
                ✕ Cancel
              </button>
            )}
            {hasSent && (
              <button
                type="button"
                className="new-sent-btn"
                onClick={onNewSentence}
              >
                → New Sentence
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
