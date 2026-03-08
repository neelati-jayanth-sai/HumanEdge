from __future__ import annotations

import asyncio
import base64
import os
import platform
import time
from collections import deque
from pathlib import Path
from typing import Any

import io

import cv2
import numpy as np
import orjson
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, ORJSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.llm import ConversationContext, LLMService, _clean_llm_output
from backend.models import DebugInfo, WebSocketResponse
from backend.vision.gesture_classifier import GestureClassifier, GestureClassifierConfig
from backend.vision.hand_detector import HandDetector, HandDetectorConfig
from backend.vision.mediapipe_gesture_classifier import (
    MediaPipeGestureClassifier,
    MediaPipeGestureClassifierConfig,
)
from backend.vision.token_buffer import TokenBuffer
from backend.vision.motion_classifier import MotionClassifier

load_dotenv()  # Must be called before RuntimeConfig is instantiated


def _clean_partial(text: str) -> str:
    """Light clean for streaming partials — strips leading quotes only."""
    return text.lstrip('"\'').rstrip('"\'') or text

_GESTURE_MODEL_FILE = Path(__file__).parent / "models" / "gesture_recognizer.task"
_HISTORY_FILE = Path(__file__).parent / "conversation_history.json"

# Motion gesture labels — these fire once (have their own cooldown) so they must
# bypass the TokenBuffer stability window and the GestureVoter.
_MOTION_LABELS: frozenset[str] = frozenset({
    "WAVE", "THANK_YOU", "PLEASE", "SORRY", "YES_NOD", "MORE", "EAT", "COME",
})


# ---------------------------------------------------------------------------
# Persistent conversation history (survives reconnects / restarts)
# ---------------------------------------------------------------------------

def _load_history() -> list[dict]:
    if _HISTORY_FILE.exists():
        try:
            data = orjson.loads(_HISTORY_FILE.read_bytes())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _save_history(history: list[dict]) -> None:
    try:
        _HISTORY_FILE.write_bytes(orjson.dumps(history))
    except Exception:
        pass


# Keep at most 200 entries on disk; in-memory list is the live copy.
_MAX_HISTORY = 200
_global_history: list[dict] = _load_history()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class RuntimeConfig:
    """Runtime configuration. All values read from env at instantiation time (after load_dotenv)."""

    def __init__(self) -> None:
        self.gesture_backend: str = os.getenv("GESTURE_BACKEND", "mediapipe").strip().lower()
        self.max_frame_bytes: int = int(os.getenv("MAX_FRAME_BYTES", "300000"))
        self.max_width: int = int(os.getenv("MAX_FRAME_WIDTH", "1280"))
        self.max_height: int = int(os.getenv("MAX_FRAME_HEIGHT", "720"))
        self.max_fps: int = int(os.getenv("MAX_FPS", "15"))
        self.stable_window_ms: int = int(os.getenv("STABLE_WINDOW_MS", "100"))
        self.confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
        self.mediapipe_confidence_threshold: float = float(
            os.getenv("MEDIAPIPE_CONFIDENCE_THRESHOLD", "0.65")
        )
        self.mediapipe_min_detection_confidence: float = float(
            os.getenv("MEDIAPIPE_MIN_DETECTION_CONFIDENCE", "0.65")
        )
        self.mediapipe_min_tracking_confidence: float = float(
            os.getenv("MEDIAPIPE_MIN_TRACKING_CONFIDENCE", "0.6")
        )
        self.mediapipe_static_image_mode: bool = _env_bool("MEDIAPIPE_STATIC_IMAGE_MODE", True)
        self.mediapipe_try_flipped_frame: bool = _env_bool("MEDIAPIPE_TRY_FLIPPED_FRAME", False)
        self.gesture_model_path: str = os.getenv("GESTURE_MODEL_PATH", "")
        self.emit_debug_frames: bool = _env_bool("EMIT_DEBUG_FRAMES", True)


# ---------------------------------------------------------------------------
# Temporal gesture voter
# ---------------------------------------------------------------------------

class GestureVoter:
    """
    Accumulates per-frame predictions over a sliding window and votes on the
    winning gesture by confidence-weighted frequency.

    A gesture wins when it appears in at least `min_vote_ratio` of the window
    frames — meaning up to 60 % missed detections are tolerated while still
    reacting quickly to real gesture changes.
    """

    def __init__(self, window_size: int = 7, min_vote_ratio: float = 0.40) -> None:
        self.window_size = window_size
        self.min_vote_ratio = min_vote_ratio
        self._history: deque[tuple[str | None, float]] = deque(maxlen=window_size)

    def update(self, label: str | None, confidence: float) -> tuple[str | None, float]:
        self._history.append((label, confidence))
        return self._vote()

    def _vote(self) -> tuple[str | None, float]:
        if not self._history:
            return None, 0.0

        votes: dict[str, float] = {}
        counts: dict[str, int] = {}
        for lbl, conf in self._history:
            if lbl is not None:
                votes[lbl] = votes.get(lbl, 0.0) + conf
                counts[lbl] = counts.get(lbl, 0) + 1

        if not votes:
            return None, 0.0

        best = max(votes, key=lambda k: votes[k])
        ratio = counts[best] / len(self._history)
        if ratio < self.min_vote_ratio:
            return None, 0.0

        avg_conf = votes[best] / counts[best]
        return best, avg_conf

    def reset(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Sign pipeline
# ---------------------------------------------------------------------------

class SignPipeline:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.hand_detector: HandDetector | None = None
        self.classifier: GestureClassifier | None = None
        self.mediapipe_classifier: MediaPipeGestureClassifier | None = None

        if config.gesture_backend == "mediapipe":
            self.mediapipe_classifier = MediaPipeGestureClassifier(
                MediaPipeGestureClassifierConfig(
                    confidence_threshold=config.mediapipe_confidence_threshold,
                    min_detection_confidence=config.mediapipe_min_detection_confidence,
                    min_tracking_confidence=config.mediapipe_min_tracking_confidence,
                    static_image_mode=config.mediapipe_static_image_mode,
                    try_flipped_frame=config.mediapipe_try_flipped_frame,
                )
            )
        elif config.gesture_backend == "pytorch":
            self.hand_detector = HandDetector(
                HandDetectorConfig(
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                )
            )
            self.classifier = GestureClassifier(
                GestureClassifierConfig(
                    model_path=config.gesture_model_path,
                    confidence_threshold=config.confidence_threshold,
                )
            )
        else:
            raise ValueError(
                f"Unsupported GESTURE_BACKEND='{config.gesture_backend}'. "
                "Use 'mediapipe' or 'pytorch'."
            )

        self.llm = LLMService()

    @staticmethod
    def _decode_frame(frame_bytes: bytes) -> np.ndarray | None:
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _validate_frame(self, frame_bytes: bytes, image: np.ndarray) -> None:
        if len(frame_bytes) > self.config.max_frame_bytes:
            raise ValueError("Frame payload too large.")
        h, w = image.shape[:2]
        if w <= 0 or h <= 0:
            raise ValueError("Invalid image dimensions.")
        if w > self.config.max_width or h > self.config.max_height:
            raise ValueError("Image dimensions exceed allowed limits.")

    def classify_frame(
        self,
        frame_bytes: bytes,
        voter: GestureVoter,
        motion: MotionClassifier,
    ) -> tuple[str | None, float, bool, str | None]:
        """
        Decode, classify, and smooth a raw image frame.

        Returns (voted_label, voted_confidence, hand_tracked, raw_label).
        The voted values come from `GestureVoter` — smoothed across recent frames;
        raw_label is the per-frame result useful for the debug panel.
        """
        frame = self._decode_frame(frame_bytes)
        if frame is None:
            raise ValueError("Could not decode frame.")
        self._validate_frame(frame_bytes, frame)

        label: str | None = None
        raw_label: str | None = None
        confidence = 0.0
        hand_tracked = False

        raw_landmarks: list | None = None

        if self.config.gesture_backend == "mediapipe":
            if self.mediapipe_classifier is None:
                raise RuntimeError("MediaPipe gesture classifier is not initialized.")
            label, confidence, hand_tracked, raw_label = self.mediapipe_classifier.predict(frame)
            # Extract landmarks for motion classifier (from raw mediapipe result)
            raw_landmarks = self.mediapipe_classifier.last_landmarks
        else:
            if self.hand_detector is None or self.classifier is None:
                raise RuntimeError("PyTorch gesture pipeline is not initialized.")
            lm_array = self.hand_detector.detect_landmarks(frame)
            hand_tracked = lm_array is not None
            raw_landmarks = None
            if lm_array is not None:
                import torch  # lazy import — only used when gesture_backend=pytorch
                features = torch.from_numpy(lm_array).float().unsqueeze(0)
                label, confidence, raw_label = self.classifier.predict(features)

        # Run motion classifier on raw landmarks
        motion_result = motion.update(raw_landmarks)

        # Motion gestures fire once (have their own cooldown) — bypass the temporal voter
        # so a single strong detection is sufficient to commit the label.
        if motion_result.label and motion_result.confidence >= 0.75:
            voter.update(None, 0.0)  # keep voter state advancing without injecting motion label
            return motion_result.label, motion_result.confidence, True, raw_label

        voted_label, voted_confidence = voter.update(label, confidence)
        return voted_label, voted_confidence, hand_tracked, raw_label


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(default_response_class=ORJSONResponse, title="Sign Sentence Speech API")

# Allow the Next.js browser client to fetch the model file from a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

runtime_config = RuntimeConfig()
pipeline = SignPipeline(runtime_config)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/history")
async def get_history() -> dict:
    return {"history": _global_history}


@app.post("/history/clear")
async def clear_history() -> dict:
    _global_history.clear()
    await asyncio.to_thread(_save_history, [])
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# TTS endpoint — uses gTTS (Google TTS), no system voices required
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    tld: str = "com"   # controls accent: "co.uk" = British, "co.in" = Indian, "com" = US/Hindi

# Simple in-memory cache to avoid re-fetching identical phrases
_tts_cache: dict[tuple[str, str, str], bytes] = {}

@app.post("/tts")
async def text_to_speech(req: TTSRequest) -> StreamingResponse:
    """Convert text to speech via gTTS. Returns MP3 audio bytes."""
    cache_key = (req.text.strip(), req.lang, req.tld)
    if cache_key in _tts_cache:
        return StreamingResponse(io.BytesIO(_tts_cache[cache_key]), media_type="audio/mpeg")

    def _generate() -> bytes:
        from gtts import gTTS  # lazy import — only needed at runtime
        tts = gTTS(text=req.text, lang=req.lang, tld=req.tld)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()

    try:
        audio_bytes = await asyncio.to_thread(_generate)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS error: {exc}") from exc

    if len(_tts_cache) > 256:   # evict oldest entries
        _tts_cache.pop(next(iter(_tts_cache)))
    _tts_cache[cache_key] = audio_bytes
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")


@app.get("/models/gesture_recognizer.task")
async def serve_gesture_model() -> FileResponse:
    """Serve the MediaPipe GestureRecognizer model to browser clients."""
    if not _GESTURE_MODEL_FILE.exists():
        raise HTTPException(status_code=404, detail="gesture_recognizer.task not found")
    return FileResponse(
        str(_GESTURE_MODEL_FILE),
        media_type="application/octet-stream",
        filename="gesture_recognizer.task",
    )


# ---------------------------------------------------------------------------
# WebSocket message parsing
# ---------------------------------------------------------------------------

def _parse_message(
    message: dict[str, Any],
) -> tuple[str, bytes | None, str | None, float, list[str]]:
    """
    Parse a raw WebSocket message.

    Returns (msg_type, frame_bytes, gesture_label, gesture_confidence, extra_tokens).
    msg_type: "frame" | "gesture" | "reset" | "set_tokens"
    extra_tokens is only populated for "set_tokens" messages.
    """
    if message.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect

    if "bytes" in message and message["bytes"] is not None:
        return "frame", message["bytes"], None, 0.0, []

    text_payload = message.get("text")
    if not text_payload:
        raise ValueError("Missing frame payload.")

    payload = orjson.loads(text_payload)
    msg_type = payload.get("type", "frame")

    if msg_type == "reset":
        return "reset", None, None, 0.0, []

    if msg_type == "config":
        lang = str(payload.get("language", "en")).lower()
        scenario = str(payload.get("scenario", "general")).lower()
        custom_scenario = str(payload.get("custom_scenario", ""))[:120]  # cap length
        return "config", None, None, 0.0, [lang, scenario, custom_scenario]

    # User manually edited the token list — sent from the correction UI
    if msg_type == "set_tokens":
        raw = payload.get("tokens", [])
        clean = [str(t).strip().upper() for t in raw if str(t).strip()]
        return "set_tokens", None, None, 0.0, clean

    # Client-side detection sends just the gesture label + confidence
    if msg_type == "gesture":
        label_raw = payload.get("label")
        label = label_raw if label_raw and label_raw != "None" else None
        confidence = float(payload.get("confidence", 0.0))
        return "gesture", None, label, confidence, []

    frame_b64 = payload.get("frame")
    if not frame_b64:
        raise ValueError("Missing base64 frame in payload.")
    return "frame", base64.b64decode(frame_b64), None, 0.0, []


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    token_buffer = TokenBuffer(stable_window_ms=runtime_config.stable_window_ms)
    gesture_voter = GestureVoter(window_size=7, min_vote_ratio=0.55)
    motion_classifier = MotionClassifier()
    conversation_context = ConversationContext(max_turns=pipeline.llm.config.context_turns)
    session_language = "en"        # updated via {type:"config", language:...}
    session_scenario = "general"   # updated via {type:"config", scenario:...}
    session_custom_scenario = ""   # free-text context for the "custom" scenario
    # Pre-populate context from persisted history so the LLM has continuity
    # across reconnects and restarts.
    for entry in _global_history[-pipeline.llm.config.context_turns:]:
        conversation_context.add_turn(
            entry.get("tokens", []), entry.get("sentence", "")
        )
    last_frame_ts = 0.0
    last_sentence = ""
    min_interval = 1.0 / max(1, runtime_config.max_fps)
    # Debounce LLM calls: fire only after user pauses signing for this long.
    # Cancelled and rescheduled on every new token so rapid signing produces
    # exactly ONE sentence call (when the user stops), not one per letter.
    _LLM_DEBOUNCE_S = float(os.getenv("LLM_DEBOUNCE_MS", "500")) / 1000.0
    pending_llm_task: asyncio.Task | None = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _send(obj: WebSocketResponse) -> None:
        await websocket.send_bytes(orjson.dumps(obj.model_dump()))

    async def _send_sentence_async(tokens: list[str]) -> None:
        """
        Run the Groq LLM call in the background, streaming partial results to
        the client as they arrive.  Sends a final message with sentence_finalized=True
        once the full sentence is ready.  Also persists the turn to disk.
        """
        nonlocal last_sentence
        # Capture session state at call time — session may change mid-flight
        lang = session_language
        scenario = session_scenario
        custom_scenario = session_custom_scenario
        try:
            # --- cache fast-path (no streaming needed) ---
            cache_key = pipeline.llm._cache_key(
                tokens=tokens, context=conversation_context,
                scenario=scenario, custom_scenario=custom_scenario,
            )
            cached = pipeline.llm.cache.get(cache_key)
            if cached:
                pipeline.llm.cache.move_to_end(cache_key)
                sentence = cached
                last_sentence = sentence
                await _send(WebSocketResponse(tokens=tokens, sentence=sentence, sentence_finalized=True))
                return

            # --- streaming path ---
            loop = asyncio.get_running_loop()
            q: asyncio.Queue[str | None] = asyncio.Queue()

            def _run_stream() -> str:
                full = ""
                try:
                    for chunk in pipeline.llm.stream_tokens(tokens, conversation_context, language=lang, scenario=scenario, custom_scenario=custom_scenario):
                        full += chunk
                        loop.call_soon_threadsafe(q.put_nowait, chunk)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)
                return full

            stream_task = asyncio.create_task(asyncio.to_thread(_run_stream))

            partial = ""
            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                partial += chunk
                await _send(WebSocketResponse(
                    tokens=tokens,
                    sentence=_clean_partial(partial),
                    sentence_streaming=True,
                ))

            raw = await stream_task
            sentence = _clean_llm_output(raw)
            last_sentence = sentence

            # Add to cache
            pipeline.llm._cache_set(cache_key, sentence)
            if conversation_context is not None:
                conversation_context.add_turn(tokens=tokens, sentence=sentence)

            # Persist to global history — update in place for the same utterance
            entry = {"tokens": tokens, "sentence": sentence}
            if _global_history:
                last = _global_history[-1]
                last_tokens = last.get("tokens", [])
                is_continuation = (
                    len(tokens) >= len(last_tokens) and
                    tokens[:len(last_tokens)] == last_tokens
                )
                if is_continuation:
                    _global_history[-1] = entry
                else:
                    _global_history.append(entry)
            else:
                _global_history.append(entry)
            if len(_global_history) > _MAX_HISTORY:
                _global_history[:] = _global_history[-_MAX_HISTORY:]
            await asyncio.to_thread(_save_history, list(_global_history))
            await _send(WebSocketResponse(tokens=tokens, sentence=sentence, sentence_finalized=True))
        except Exception:
            pass  # WebSocket may have closed; ignore

    def _build_debug(
        *,
        backend: str,
        hand_tracked: bool,
        raw_label: str | None,
        accepted_label: str | None,
        confidence: float,
        updated: bool,
        buffer_debug: dict[str, Any],
    ) -> DebugInfo:
        return DebugInfo(
            backend=backend,
            hand_tracked=hand_tracked,
            raw_label=raw_label,
            accepted_label=accepted_label,
            confidence=round(confidence, 4),
            token_updated=updated,
            pending_label=(
                buffer_debug["pending_label"]
                if isinstance(buffer_debug.get("pending_label"), str)
                else None
            ),
            pending_elapsed_ms=int(buffer_debug.get("pending_elapsed_ms", 0)),
            stable_window_ms=int(buffer_debug.get("stable_window_ms", 0)),
        )

    # ── Main loop ────────────────────────────────────────────────────────────

    try:
        while True:
            message = await websocket.receive()
            msg_type, payload, gesture_label, gesture_confidence, extra_tokens = _parse_message(message)

            # --- language / scenario config change ---------------------------
            if msg_type == "config":
                if len(extra_tokens) >= 1:
                    new_lang = extra_tokens[0].lower()
                    if new_lang in {"en", "hi"}:
                        session_language = new_lang
                if len(extra_tokens) >= 2:
                    new_scenario = extra_tokens[1].lower()
                    if new_scenario in {"general", "interview", "banking", "medical", "restaurant", "custom"}:
                        session_scenario = new_scenario
                if len(extra_tokens) >= 3:
                    session_custom_scenario = extra_tokens[2]
                continue

            # --- reset -------------------------------------------------------
            if msg_type == "reset":
                if pending_llm_task is not None and not pending_llm_task.done():
                    pending_llm_task.cancel()
                token_buffer.reset()
                gesture_voter.reset()
                motion_classifier.reset()
                conversation_context.reset()
                last_sentence = ""
                await _send(WebSocketResponse(tokens=[], sentence=""))
                continue

            # --- manual token correction from the frontend UI ----------------
            if msg_type == "set_tokens":
                token_buffer.tokens = extra_tokens
                tokens = token_buffer.get_tokens()
                last_sentence = ""          # old sentence no longer matches
                await _send(WebSocketResponse(tokens=tokens, sentence=""))
                if tokens:
                    asyncio.create_task(_send_sentence_async(list(tokens)))
                continue

            now = time.monotonic()

            # --- client-side gesture event -----------------------------------
            if msg_type == "gesture":
                # Motion gestures fire once — skip stability window, commit immediately
                if gesture_label in _MOTION_LABELS:
                    updated = token_buffer.add_token(gesture_label)
                else:
                    updated = token_buffer.process_prediction(
                        gesture_label, confidence=gesture_confidence, now=now
                    )
                tokens = token_buffer.get_tokens()
                buf = token_buffer.debug_state(now=now)
                debug = _build_debug(
                    backend="client-js",
                    hand_tracked=gesture_label is not None
                    or isinstance(buf.get("pending_label"), str),
                    raw_label=gesture_label,
                    accepted_label=gesture_label,
                    confidence=gesture_confidence,
                    updated=updated,
                    buffer_debug=buf,
                )

                if updated:
                    # Phase 1 — tokens immediately visible
                    await _send(WebSocketResponse(tokens=tokens, sentence=last_sentence, debug=debug))
                    # Phase 2 — debounced sentence: cancel previous, wait for pause in signing
                    if pending_llm_task is not None and not pending_llm_task.done():
                        pending_llm_task.cancel()
                    _snap = list(tokens)
                    async def _debounced_gesture(_toks: list[str] = _snap) -> None:
                        try:
                            await asyncio.sleep(_LLM_DEBOUNCE_S)
                            await _send_sentence_async(_toks)
                        except asyncio.CancelledError:
                            pass
                    pending_llm_task = asyncio.create_task(_debounced_gesture())
                elif runtime_config.emit_debug_frames:
                    await _send(WebSocketResponse(tokens=tokens, sentence=last_sentence, debug=debug))
                continue

            # --- server-side image frame -------------------------------------
            if now - last_frame_ts < min_interval:
                continue
            last_frame_ts = now

            voted_label, voted_confidence, hand_tracked, raw_label = pipeline.classify_frame(
                payload or b"",
                gesture_voter,
                motion_classifier,
            )
            # Motion gestures bypass the stability window — they already have their own cooldown
            if voted_label in _MOTION_LABELS:
                updated = token_buffer.add_token(voted_label)
            else:
                updated = token_buffer.process_prediction(
                    voted_label, confidence=voted_confidence, now=now
                )
            tokens = token_buffer.get_tokens()
            buf = token_buffer.debug_state(now=now)
            debug = _build_debug(
                backend=runtime_config.gesture_backend,
                hand_tracked=hand_tracked,
                raw_label=raw_label,
                accepted_label=voted_label,
                confidence=voted_confidence,
                updated=updated,
                buffer_debug=buf,
            )

            if updated:
                # Phase 1 — tokens immediately visible
                await _send(WebSocketResponse(tokens=tokens, sentence=last_sentence, debug=debug))
                # Phase 2 — debounced sentence: cancel previous, wait for pause in signing
                if pending_llm_task is not None and not pending_llm_task.done():
                    pending_llm_task.cancel()
                _snap = list(tokens)
                async def _debounced_frame(_toks: list[str] = _snap) -> None:
                    try:
                        await asyncio.sleep(_LLM_DEBOUNCE_S)
                        await _send_sentence_async(_toks)
                    except asyncio.CancelledError:
                        pass
                pending_llm_task = asyncio.create_task(_debounced_frame())
            elif runtime_config.emit_debug_frames:
                await _send(WebSocketResponse(tokens=tokens, sentence=last_sentence, debug=debug))

    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await websocket.send_bytes(orjson.dumps({"error": str(exc)}))
            await asyncio.sleep(0.05)
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    event_loop = "uvloop" if platform.system() != "Windows" else "asyncio"
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        loop=event_loop,
        ws="websockets",
        reload=False,
    )
