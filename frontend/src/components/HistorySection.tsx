"use client";

import type React from "react";
import type { HistoryEntry } from "../lib/types";

interface Props {
  history: HistoryEntry[];
  historyEndRef: React.RefObject<HTMLDivElement | null>;
}

export default function HistorySection({ history, historyEndRef }: Props) {
  if (history.length === 0) return null;

  return (
    <div className="sec">
      <div className="sec-head">
        <span className="sec-title">Conversation History</span>
      </div>
      <div className="sec-body" style={{ paddingTop: 8 }}>
        <div className="hist-list">
          {history.map((entry, i) => (
            <div key={i} className="hist-entry">
              <div className="hist-tokens">
                {entry.tokens.map((t, j) => (
                  <span key={j} className="hist-tok">{t}</span>
                ))}
              </div>
              <div className="hist-sent">{entry.sentence}</div>
            </div>
          ))}
          <div ref={historyEndRef} />
        </div>
      </div>
    </div>
  );
}
