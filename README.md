# Real-Time Sign -> Sentence -> Speech System

Production architecture:

- `frontend` streams webcam frames via WebSocket and does UI + browser TTS only.
- `backend` supports lightweight MediaPipe Hands landmark tracking + rule-based gesture mapping (recommended local default) or PyTorch 200-class inference, plus token buffering and Groq sentence generation.

## Architecture diagram

```mermaid
flowchart LR
    A[User Webcam] --> B[Frontend]
    B -->|Binary JPEG / JSON frame| C[FastAPI WebSocket /ws]
    C --> D[Gesture Backend]
    D --> E[MediaPipe Hands<br/>landmark tracking + rules]
    D --> F[PyTorch Gesture Classifier<br/>.pt via GESTURE_MODEL_PATH]
    E --> H[Token Buffer]
    F --> H
    H -->|tokens| G[Groq LLM Service]
    G -->|sentence| C
    C -->|{ tokens, sentence }| B
    B --> H[UI Text + Browser TTS]
```

## Backend

1. Create a virtual environment and install dependencies:
   - `pip install -r requirements.txt`
2. Configure environment:
   - `copy .env.example .env`
   - Set `GROQ_API_KEY`
   - Default local mode is `GESTURE_BACKEND=mediapipe`
   - Optional PyTorch mode: set `GESTURE_BACKEND=pytorch` and set `GESTURE_MODEL_PATH` to your trained 200-class `.pt` weights
   - `MEDIAPIPE_CONFIDENCE_THRESHOLD` controls MediaPipe acceptance (recommended `0.25` to `0.40` for local webcam)
   - `MEDIAPIPE_MIN_DETECTION_CONFIDENCE` and `MEDIAPIPE_MIN_TRACKING_CONFIDENCE` control hand tracking sensitivity (default `0.5`)
   - `MEDIAPIPE_STATIC_IMAGE_MODE=true` forces per-frame palm detection (more reliable for WebSocket image streams)
   - `MEDIAPIPE_TRY_FLIPPED_FRAME=true` retries detection on mirrored frames if first pass fails
   - Optional: set `CONVERSATION_CONTEXT_TURNS` (default `6`) to control how many recent turns are used for LLM context
   - Optional: set `EMIT_DEBUG_FRAMES=true` to stream per-frame debug telemetry (`hand_tracked`, labels, confidence, stability)
3. Start backend:
   - Run from project root (`hackthon`), not from `frontend/`
   - `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --ws websockets`
   - If you are inside `frontend/`, use: `python -m uvicorn backend.main:app --app-dir .. --host 0.0.0.0 --port 8000 --ws websockets`

The backend runs with:

- FastAPI async endpoints
- WebSocket `/ws`
- `orjson` response serialization
- `uvicorn` with `uvloop` on Linux/macOS and `asyncio` on Windows
- single global `Groq` client instance

MediaPipe note:
- The default MediaPipe mode tracks hand landmarks using the same approach as `kinivi/hand-gesture-recognition-mediapipe` and maps common gestures (`OPEN`, `CLOSE`, `POINTER`, `OK`, `VICTORY`, `THUMB_UP`, `THUMB_DOWN`, `ILOVEYOU`).
- It is not a full sign-language word classifier; use `GESTURE_BACKEND=pytorch` with your trained 200-class model for full vocabulary.

If you still see `Hand Tracked: No`:
- Ensure backend is restarted after `.env` edits.
- Keep one hand centered and near camera (roughly 30-80cm), with good front lighting.
- Try `MEDIAPIPE_MIN_DETECTION_CONFIDENCE=0.15` and `MEDIAPIPE_MIN_TRACKING_CONFIDENCE=0.15`.
- Keep `EMIT_DEBUG_FRAMES=true` and verify the Debug panel updates continuously.

## Frontend

1. Install packages:
   - `cd frontend`
   - `npm install`
2. Set backend WebSocket URL (optional):
   - `set NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws`
3. Start:
   - `npm run dev`

## Backend module layout

- `backend/main.py`: FastAPI app and `/ws` real-time pipeline
- `backend/vision/hand_detector.py`: MediaPipe hand landmarks extraction and normalization
- `backend/vision/gesture_classifier.py`: PyTorch 200-class model architecture and runtime classifier
- `backend/vision/mediapipe_gesture_classifier.py`: MediaPipe Hands landmark + rule-based classifier
- `backend/vision/token_buffer.py`: server-side stable token buffering
- `backend/llm.py`: Groq (`llama-3.1-8b-instant`) integration
- `backend/models.py`: API response models

## WebSocket protocol

Input:

- Binary JPEG frame payloads (preferred), or JSON `{ "frame": "<base64>" }`
- JSON `{ "type": "reset" }` to clear tokens

Output:

```json
{
  "tokens": ["I", "WANT"],
  "sentence": "I would like some water."
}
```
