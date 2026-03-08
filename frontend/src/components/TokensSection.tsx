"use client";

import { ALL_GESTURE_LABELS } from "../lib/constants";

interface Props {
  tokens: string[];
  tokenInput: string;
  onInputChange: (val: string) => void;
  onAdd: () => void;
  onDelete: (idx: number) => void;
  onRegenerate: () => void;
}

export default function TokensSection({
  tokens,
  tokenInput,
  onInputChange,
  onAdd,
  onDelete,
  onRegenerate,
}: Props) {
  return (
    <div className="sec">
      <div className="sec-head">
        <span className="sec-title">Detected Signs</span>
      </div>
      <div className="sec-body">
        <div className="token-area">
          {tokens.length === 0 && (
            <span className="token-empty">No signs detected yet…</span>
          )}
          {tokens.map((t, i) => (
            <span key={`${t}-${i}`} className="token-chip">
              {t}
              <button
                type="button"
                className="token-del"
                title="Remove token"
                onClick={() => onDelete(i)}
              >
                ×
              </button>
            </span>
          ))}
        </div>
        <div className="fix-bar">
          <input
            list="gesture-datalist"
            className="fix-input"
            placeholder="Add or correct a token…"
            value={tokenInput}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") onAdd(); }}
          />
          <datalist id="gesture-datalist">
            {ALL_GESTURE_LABELS.map((l) => <option key={l} value={l} />)}
          </datalist>
          <button
            type="button"
            className="fix-btn fix-add"
            onClick={onAdd}
            disabled={!tokenInput.trim()}
          >
            + Add
          </button>
          {tokens.length > 0 && (
            <button
              type="button"
              className="fix-btn fix-regen"
              onClick={onRegenerate}
            >
              ↺ Re-gen
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
