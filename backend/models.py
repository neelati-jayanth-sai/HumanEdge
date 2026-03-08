from __future__ import annotations

from pydantic import BaseModel, Field


class DebugInfo(BaseModel):
    backend: str
    hand_tracked: bool
    raw_label: str | None = None
    accepted_label: str | None = None
    confidence: float = 0.0
    token_updated: bool = False
    pending_label: str | None = None
    pending_elapsed_ms: int = 0
    stable_window_ms: int = 0


class WebSocketResponse(BaseModel):
    tokens: list[str] = Field(default_factory=list)
    sentence: str = ""
    sentence_finalized: bool = False
    sentence_streaming: bool = False
    debug: DebugInfo | None = None
