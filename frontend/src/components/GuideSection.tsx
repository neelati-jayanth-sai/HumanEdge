"use client";

import { GESTURE_GUIDE_CATEGORIES } from "../lib/constants";

export default function GuideSection() {
  return (
    <div className="sec">
      <div className="sec-head">
        <span className="sec-title">ASL Reference Guide</span>
      </div>
      <div className="sec-body">
        <p className="guide-note">
          Hold each gesture still for ~0.5s until the token commits
        </p>
        {GESTURE_GUIDE_CATEGORIES.map((cat) => (
          <div key={cat.category} className="guide-cat">
            <div className="guide-cat-name">{cat.category}</div>
            <div className="guide-grid">
              {cat.signs.map((g) => (
                <div key={g.label} className="guide-card">
                  <span className="g-emoji">{g.emoji}</span>
                  <div className="g-info">
                    <span className="g-label">{g.label}</span>
                    <span className="g-desc">{g.how}</span>
                    <span className="g-meaning">{g.tip}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
