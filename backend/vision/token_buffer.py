from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class TokenBuffer:
    stable_window_ms: int = 500
    tokens: list[str] = field(default_factory=list)
    _pending_label: str | None = None
    _pending_since: float = 0.0
    _effective_window_ms: float = 500.0  # Adaptive; shrinks for high-confidence predictions

    def add_token(self, label: str) -> bool:
        if not self.tokens or self.tokens[-1] != label:
            self.tokens.append(label)
            return True
        return False

    def process_prediction(
        self,
        label: str | None,
        confidence: float = 1.0,
        now: float | None = None,
    ) -> bool:
        now = now if now is not None else time.monotonic()

        if not label:
            self._pending_label = None
            self._pending_since = 0.0
            return False

        if self._pending_label != label:
            # New candidate — start the stability clock and compute adaptive window.
            self._pending_label = label
            self._pending_since = now
            self._effective_window_ms = self._adaptive_window(confidence)
            return False

        elapsed_ms = (now - self._pending_since) * 1000.0
        if elapsed_ms >= self._effective_window_ms:
            return self.add_token(label)
        return False

    def _adaptive_window(self, confidence: float) -> float:
        """
        Scale the stability window for confidence level.
        Kept aggressive so fast signers are not bottlenecked.

          >= 0.92  → 35 %   (very high confidence — commit almost immediately)
          >= 0.82  → 55 %   (high confidence)
          >= 0.70  → 75 %   (medium confidence)
          <  0.70  → 100 %  (low confidence — wait the full window)
        """
        if confidence >= 0.92:
            return self.stable_window_ms * 0.35
        if confidence >= 0.82:
            return self.stable_window_ms * 0.55
        if confidence >= 0.70:
            return self.stable_window_ms * 0.75
        return float(self.stable_window_ms)

    def get_tokens(self) -> list[str]:
        return list(self.tokens)

    def reset(self) -> None:
        self.tokens.clear()
        self._pending_label = None
        self._pending_since = 0.0
        self._effective_window_ms = float(self.stable_window_ms)

    def debug_state(self, now: float | None = None) -> dict[str, str | int | float | None]:
        now = now if now is not None else time.monotonic()
        pending_elapsed_ms = 0
        if self._pending_label and self._pending_since > 0:
            pending_elapsed_ms = int(max(0.0, (now - self._pending_since) * 1000.0))
        return {
            "pending_label": self._pending_label,
            "pending_elapsed_ms": pending_elapsed_ms,
            # Report the effective (adaptive) window so the UI shows accurate progress
            "stable_window_ms": int(self._effective_window_ms),
        }
