# Sign Scribe — Technical Document

**Version:** 1.0
**Date:** March 2026
**Platform:** Windows 11 / Cross-platform

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Repository Structure](#4-repository-structure)
5. [Backend — Detailed Design](#5-backend--detailed-design)
   - 5.1 [Entry Point & Configuration](#51-entry-point--configuration)
   - 5.2 [Gesture Detection Pipeline](#52-gesture-detection-pipeline)
   - 5.3 [Temporal Smoothing — GestureVoter](#53-temporal-smoothing--gesturevoter)
   - 5.4 [Token Buffer](#54-token-buffer)
   - 5.5 [Motion Classifier](#55-motion-classifier)
   - 5.6 [MediaPipe Gesture Classifier](#56-mediapipe-gesture-classifier)
   - 5.7 [ASL Rule-Based Classifier](#57-asl-rule-based-classifier)
   - 5.8 [PyTorch Gesture Classifier](#58-pytorch-gesture-classifier)
   - 5.9 [Hand Detector](#59-hand-detector)
   - 5.10 [LLM Service](#510-llm-service)
   - 5.11 [WebSocket Endpoint](#511-websocket-endpoint)
   - 5.12 [REST API Endpoints](#512-rest-api-endpoints)
   - 5.13 [Persistent History](#513-persistent-history)
   - 5.14 [Text-to-Speech (TTS) Endpoint](#514-text-to-speech-tts-endpoint)
6. [Frontend — Detailed Design](#6-frontend--detailed-design)
   - 6.1 [Application Layout & Routing](#61-application-layout--routing)
   - 6.2 [SignStreamClient — Main Orchestrator](#62-signstreamclient--main-orchestrator)
   - 6.3 [Client-Side Gesture Classifier](#63-client-side-gesture-classifier)
   - 6.4 [useTTS Hook](#64-usetts-hook)
   - 6.5 [UI Components](#65-ui-components)
   - 6.6 [Design System & CSS](#66-design-system--css)
7. [Data Flow & Communication Protocol](#7-data-flow--communication-protocol)
   - 7.1 [WebSocket Message Types](#71-websocket-message-types)
   - 7.2 [End-to-End Frame Lifecycle](#72-end-to-end-frame-lifecycle)
   - 7.3 [Sentence Streaming Flow](#73-sentence-streaming-flow)
8. [Gesture Vocabulary](#8-gesture-vocabulary)
9. [LLM Prompt Engineering](#9-llm-prompt-engineering)
10. [Calibration System](#10-calibration-system)
11. [Configuration Reference](#11-configuration-reference)
12. [Dependencies & Libraries](#12-dependencies--libraries)
13. [Security Considerations](#13-security-considerations)
14. [Known Limitations & Trade-offs](#14-known-limitations--trade-offs)

---

## 1. Project Overview

**Sign Scribe** is a real-time American Sign Language (ASL) recognition system that converts hand gestures captured via a webcam into natural language sentences. It bridges communication between deaf/hard-of-hearing individuals and hearing people by:

1. **Detecting hand gestures** using MediaPipe and/or a custom PyTorch neural network.
2. **Accumulating gesture tokens** (e.g., `THANK_YOU`, `W`, `A`, `T`, `E`, `R`) with temporal stability logic to filter noise.
3. **Generating natural sentences** from token sequences using the Groq LLM API (Llama 3.3 70B).
4. **Speaking the sentence aloud** via Google TTS (gTTS) or the browser Web Speech API.

The system operates in real time with sub-second gesture detection latency, streaming sentence generation, and multi-language output (English and Hindi).

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Browser (Next.js)                          │
│                                                                     │
│  Webcam → MediaPipe WASM → ClientGestureDetector                   │
│               │                       │                            │
│               │  gesture label +      │  JPEG frames               │
│               │  confidence (JSON)    │  (fallback, binary)        │
│               └───────────┬───────────┘                            │
│                           │  WebSocket /ws                         │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                       FastAPI Backend (Python)                       │
│                                                                     │
│  WebSocket Session                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  GestureVoter (7-frame sliding window)                      │   │
│  │  MotionClassifier (25-frame window, geometry-based)         │   │
│  │  MediaPipeGestureClassifier OR PyTorchGestureClassifier     │   │
│  │  TokenBuffer (adaptive stability window)                    │   │
│  │  ConversationContext (up to 6 turns)                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           │ tokens                                                  │
│           ▼                                                         │
│  LLMService ──► Groq API (llama-3.3-70b-versatile, streaming)      │
│           │ sentence chunks                                         │
│           ▼                                                         │
│  WebSocket → browser (sentence_streaming / sentence_finalized)      │
│                                                                     │
│  REST Endpoints:                                                    │
│    GET  /health                                                     │
│    GET  /history                                                    │
│    POST /history/clear                                              │
│    POST /tts          → gTTS → MP3 audio bytes                     │
│    GET  /models/gesture_recognizer.task → MediaPipe model file     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

### Backend

| Component | Technology | Version / Notes |
|-----------|-----------|-----------------|
| Web Framework | FastAPI | Async, ASGI |
| ASGI Server | Uvicorn | `asyncio` loop on Windows, `uvloop` on Linux/macOS |
| WebSocket | `websockets` library via Uvicorn |
| Computer Vision | OpenCV (`cv2`) | Frame decode, CLAHE, gamma correction |
| Hand Landmark Detection | MediaPipe (`mediapipe`) | GestureRecognizer `.task` model + Hands solution |
| Deep Learning (optional) | PyTorch | Custom 200-class ASL classifier |
| JSON Serialization | `orjson` | Faster than stdlib `json` |
| LLM API | Groq Python SDK | `groq` package |
| LLM Model | Llama 3.3 70B Versatile | `llama-3.3-70b-versatile` via Groq |
| TTS | gTTS (Google Text-to-Speech) | MP3 audio streaming |
| Data Validation | Pydantic v2 | `BaseModel`, `dataclass(slots=True)` |
| Environment Config | `python-dotenv` | `.env` file loading |
| Numerical Computing | NumPy | Landmark arrays |
| Async | Python `asyncio` | `asyncio.to_thread`, `asyncio.Queue` |

### Frontend

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Next.js | 15.4.6 (App Router) |
| Language | TypeScript | 5.9.2 |
| UI Library | React | 19.1.0 |
| Hand Detection (browser) | `@mediapipe/tasks-vision` | 0.10.14 |
| Fonts | Google Fonts (Cormorant Garamond, Space Mono, Outfit) | via `next/font/google` |
| TTS (fallback) | Web Speech API (`SpeechSynthesisUtterance`) | Browser native |
| WebSocket | Browser native `WebSocket` |
| CSS | Plain CSS with CSS custom properties (design tokens) |
| Build Tool | Next.js bundler (webpack/turbopack) |

---

## 4. Repository Structure

```
hackthon/
├── .env                              ← Live runtime configuration
├── .env.example                      ← Configuration template
├── TECHNICAL_DOCUMENT.md             ← This document
├── mvp.doc                           ← MVP planning document
│
├── backend/
│   ├── __init__.py
│   ├── main.py                       ← FastAPI app, WebSocket, REST endpoints (710 lines)
│   ├── llm.py                        ← Groq LLM service, prompt engineering (310 lines)
│   ├── models.py                     ← Pydantic schemas (DebugInfo, WebSocketResponse)
│   ├── train_asl_classifier.py       ← PyTorch model training script
│   ├── train_asl_colab.py            ← Google Colab training variant
│   ├── conversation_history.json     ← Persisted session history (up to 200 entries)
│   ├── models/
│   │   ├── gesture_recognizer.task   ← MediaPipe GestureRecognizer model file
│   │   └── asl_labels.json          ← ASL class label mapping
│   └── vision/
│       ├── __init__.py
│       ├── hand_detector.py          ← MediaPipe Hands wrapper (landmark extraction)
│       ├── gesture_classifier.py     ← PyTorch 200-class inference wrapper
│       ├── mediapipe_gesture_classifier.py ← Primary classifier + ASL rules (575 lines)
│       ├── motion_classifier.py      ← Temporal dynamic sign detector (400 lines)
│       └── token_buffer.py           ← Adaptive stability window logic (84 lines)
│
└── frontend/
    ├── package.json
    ├── tsconfig.json
    ├── next.config.ts
    └── src/
        ├── app/
        │   ├── layout.tsx            ← Root layout, Google Fonts, metadata
        │   ├── page.tsx              ← Entry point
        │   └── globals.css           ← Full design system (1468 lines)
        ├── lib/
        │   ├── types.ts              ← Shared TypeScript types
        │   ├── constants.ts          ← API URLs, gesture labels, scenarios
        │   └── gestureClassifier.ts  ← Browser-side MediaPipe + ASL rules (626 lines)
        ├── hooks/
        │   └── useTTS.ts             ← TTS hook (gTTS + Web Speech API fallback)
        └── components/
            ├── SignStreamClient.tsx   ← Main orchestrator component (685 lines)
            ├── CameraPanel.tsx       ← Live video + gesture overlay
            ├── TokensSection.tsx     ← Token chips + manual editing
            ├── SentenceSection.tsx   ← Generated sentence + TTS controls
            ├── HistorySection.tsx    ← Past conversation history
            ├── ControlsPanel.tsx     ← Start/Stop/Clear buttons
            ├── SettingsStrip.tsx     ← Language, accent, scenario selector
            ├── GuideSection.tsx      ← Gesture reference guide
            ├── DebugSection.tsx      ← Collapsible debug console
            └── CalibrationModal.tsx  ← Confidence threshold calibration
```

---

## 5. Backend — Detailed Design

### 5.1 Entry Point & Configuration

**File:** `backend/main.py`

The application is a FastAPI application served by Uvicorn. At startup:

1. `load_dotenv()` loads all environment variables from `.env`.
2. `RuntimeConfig` reads every tunable parameter from environment variables with safe defaults.
3. `SignPipeline` initializes either MediaPipe or PyTorch backends based on `GESTURE_BACKEND`.
4. `LLMService` is instantiated, verifying `GROQ_API_KEY` is present.
5. Persistent history is loaded from `conversation_history.json`.

**`RuntimeConfig` parameters:**

| Parameter | Env Variable | Default | Description |
|-----------|-------------|---------|-------------|
| `gesture_backend` | `GESTURE_BACKEND` | `mediapipe` | `"mediapipe"` or `"pytorch"` |
| `max_frame_bytes` | `MAX_FRAME_BYTES` | `300000` | Max JPEG payload size (300 KB) |
| `max_width` | `MAX_FRAME_WIDTH` | `1280` | Max accepted frame width |
| `max_height` | `MAX_FRAME_HEIGHT` | `720` | Max accepted frame height |
| `max_fps` | `MAX_FPS` | `15` | Server-side frame rate cap |
| `stable_window_ms` | `STABLE_WINDOW_MS` | `100` | Base gesture stability window |
| `confidence_threshold` | `CONFIDENCE_THRESHOLD` | `0.75` | PyTorch backend threshold |
| `mediapipe_confidence_threshold` | `MEDIAPIPE_CONFIDENCE_THRESHOLD` | `0.65` | MediaPipe accept threshold |
| `mediapipe_min_detection_confidence` | `MEDIAPIPE_MIN_DETECTION_CONFIDENCE` | `0.65` | MediaPipe hand detection |
| `mediapipe_min_tracking_confidence` | `MEDIAPIPE_MIN_TRACKING_CONFIDENCE` | `0.60` | MediaPipe tracking |
| `mediapipe_static_image_mode` | `MEDIAPIPE_STATIC_IMAGE_MODE` | `True` | Treat each frame independently |
| `mediapipe_try_flipped_frame` | `MEDIAPIPE_TRY_FLIPPED_FRAME` | `False` | Also try horizontally flipped frame |
| `emit_debug_frames` | `EMIT_DEBUG_FRAMES` | `True` | Send debug info on every frame |

---

### 5.2 Gesture Detection Pipeline

**File:** `backend/main.py` — class `SignPipeline`

`SignPipeline` is the central vision component instantiated once at startup. It owns the gesture backend and the `LLMService`.

**Method: `classify_frame(frame_bytes, voter, motion)`**

```
frame_bytes (JPEG binary)
    │
    ▼
cv2.imdecode → np.ndarray (BGR)
    │
    ├── Validate: size ≤ max_frame_bytes, dimensions ≤ max_width × max_height
    │
    ├── [MediaPipe path]
    │   └── MediaPipeGestureClassifier.predict(frame)
    │       → (label, confidence, hand_tracked, raw_label)
    │       → also sets mediapipe_classifier.last_landmarks
    │
    ├── [PyTorch path]
    │   └── HandDetector.detect_landmarks(frame) → lm_array (21×3)
    │       └── GestureClassifier.predict(features) → (label, confidence, raw_label)
    │
    ├── MotionClassifier.update(raw_landmarks)
    │   → MotionResult(label, confidence)
    │   [if motion_result.confidence ≥ 0.75]
    │       → bypass voter, return motion label immediately
    │
    └── GestureVoter.update(label, confidence)
        → (voted_label, voted_confidence)
```

The motion classifier fires at most once per cooldown period (1.2–1.5 s) and bypasses the temporal voter to ensure dynamic signs like WAVE and THANK_YOU are not diluted across the window.

---

### 5.3 Temporal Smoothing — GestureVoter

**File:** `backend/main.py` — class `GestureVoter`

**Purpose:** Smooth noisy per-frame predictions over a sliding window to reduce false positives from transient hand poses.

**Algorithm:**

- Maintains a `deque` of the last `window_size=7` `(label, confidence)` pairs.
- On each `update()`, appends the new prediction and computes a weighted vote:
  - For each unique label, sums confidence scores across all frames in the window.
  - Selects the label with the highest total confidence score.
  - Checks that this label appears in at least `min_vote_ratio=0.55` of frames (i.e., at least 4 of 7).
  - If the ratio check passes, returns `(best_label, average_confidence)`.
  - Otherwise returns `(None, 0.0)`.

This tolerates up to 45% missed detections per window while still responding quickly to genuine sign changes.

---

### 5.4 Token Buffer

**File:** `backend/vision/token_buffer.py` — class `TokenBuffer`

**Purpose:** Convert a stream of frame-level predictions into discrete committed tokens, requiring a gesture to be held stably for a minimum duration before it is accepted.

**Key logic — `process_prediction(label, confidence, now)`:**

1. If `label` is `None` or empty, reset pending state. No token is added.
2. If the incoming `label` differs from `_pending_label`, start a new stability clock and compute the adaptive window.
3. If the same `label` has been held for `_effective_window_ms` milliseconds, call `add_token(label)`.
4. `add_token` only appends the label if it differs from the last token (prevents duplicates from holding).

**Adaptive stability window:**

| Confidence | % of base `stable_window_ms` | Rationale |
|-----------|------------------------------|-----------|
| ≥ 0.92 | 35% | Very high confidence — commit almost immediately |
| ≥ 0.82 | 55% | High confidence |
| ≥ 0.70 | 75% | Medium confidence |
| < 0.70 | 100% | Low confidence — full wait required |

With `STABLE_WINDOW_MS=100`, a 0.92+ confidence gesture commits in just 35 ms.

**Motion gesture bypass:** Motion labels (`WAVE`, `THANK_YOU`, etc.) call `add_token()` directly, skipping the stability window entirely. They have their own cooldown inside `MotionClassifier`.

---

### 5.5 Motion Classifier

**File:** `backend/vision/motion_classifier.py`

**Purpose:** Detect dynamic ASL signs that require movement rather than a static hand pose. No machine learning model is used — all detection is pure geometry on a 25-frame sliding window of MediaPipe landmarks.

**Architecture:**

- `MotionClassifier` maintains a `deque(maxlen=25)` of landmark frames (~1.7 s at 15 FPS).
- On each `update(landmarks)`, it runs all 8 `_Detector` subclasses and returns the highest-confidence result.
- If no hand is tracked for 8+ consecutive frames, the window is cleared.

**Detector implementations:**

| Class | Sign | Detection Logic | Cooldown |
|-------|------|-----------------|----------|
| `WaveDetector` | WAVE | Wrist oscillates horizontally ≥3 reversals; hand openness ≥0.6 | 1.5 s |
| `ThankYouDetector` | THANK_YOU | Open hand (openness >0.6) starts near chin (y ∈ [0.3, 0.65]) and moves down by >0.06 normalized units | 1.2 s |
| `PleaseDetector` | PLEASE | Open palm traces arc >1.2π radians on chest (circular path analysis via unwrapped angle) | 1.2 s |
| `SorryDetector` | SORRY | Closed fist traces arc >1.2π radians on chest | 1.5 s |
| `YesNodDetector` | YES_NOD | Closed fist bounces vertically ≥3 reversals | 1.2 s |
| `MoreDetector` | MORE | Thumb-index pinch distance oscillates ≥3 reversals (open/close repeatedly) | 1.2 s |
| `EatDetector` | EAT | Wrist near face level (y <0.65), vertical reversals ≥2 | 1.2 s |
| `ComeDetector` | COME | Net downward wrist motion >0.08 normalized units (beckoning gesture) | 1.2 s |

**Geometry helpers used:**
- `_dist(a, b)` — 2D Euclidean distance
- `_wrist(lms)` — landmark 0 coordinates
- `_palm_center(lms)` — average of wrist + 4 MCP joints
- `_hand_open_ratio(lms)` — average tip-to-wrist distance normalized by palm size
- `_pinch_dist(lms)` — thumb-tip to index-tip distance normalized by palm size
- `_wrist_y_relative_to_face(lms)` — raw y-coordinate used as face proximity heuristic

---

### 5.6 MediaPipe Gesture Classifier

**File:** `backend/vision/mediapipe_gesture_classifier.py`

**Purpose:** Primary computer vision module. Wraps MediaPipe in two modes with image preprocessing and frame variant generation.

**Initialization priority:**
1. If `models/gesture_recognizer.task` exists → use MediaPipe Tasks `GestureRecognizer` (`.task` model).
2. Otherwise → fallback to legacy `mediapipe.solutions.hands.Hands`.

**Image preprocessing — `_build_variants(frame_bgr)`:**

Generates up to 6 image variants to improve detection robustness:
1. Original frame (BGR→RGB)
2. CLAHE-enhanced (Contrast Limited Adaptive Histogram Equalization, `clipLimit=2.5`, `tileGridSize=8×8`)
3. Gamma-brightened (γ=1.6)
4. Horizontally flipped (if `try_flipped_frame=True`)
5. CLAHE of flipped
6. Gamma of flipped

Each variant is passed through the recognizer. The highest-confidence accepted result across all variants is returned. For flipped frames, the handedness label is inverted.

**Classification priority (Task model path):**
1. Run `_classify_asl(landmarks, handedness)` on each detected hand.
2. If ASL classifier confidence ≥ threshold → return immediately.
3. If ASL classifier has medium confidence (≥ threshold × 0.8) → prefer over built-in task gesture.
4. Fallback to built-in MediaPipe 8-gesture vocabulary (mapped to clean tokens via `_normalize_task_label`).

**Built-in gesture label mapping:**

| MediaPipe Label | Sign Scribe Token |
|----------------|------------------|
| `Open_Palm` | `OPEN_PALM` |
| `Closed_Fist` | `FIST` |
| `Pointing_Up` | `POINTING` |
| `Thumb_Up` | `THUMB_UP` |
| `Thumb_Down` | `THUMB_DOWN` |
| `Victory` | `V` |
| `ILoveYou` | `ILY` |

---

### 5.7 ASL Rule-Based Classifier

**Location:** `MediaPipeGestureClassifier._classify_asl()` (also mirrored in `frontend/src/lib/gestureClassifier.ts`)

**Purpose:** Classify ASL hand shapes from 21 MediaPipe landmarks using geometric rules. No trained model required.

**Input:** 21 normalized `(x, y, z)` landmark points (MediaPipe convention):

| Index | Landmark |
|-------|---------|
| 0 | Wrist |
| 1–4 | Thumb (CMC, MCP, IP, TIP) |
| 5–8 | Index finger (MCP, PIP, DIP, TIP) |
| 9–12 | Middle finger |
| 13–16 | Ring finger |
| 17–20 | Pinky finger |

**Core geometric features computed:**

- **`_finger_extended(tip, pip, mcp)`**: Vector dot-product test. A finger is "extended" if the PIP→TIP vector is in the same general direction as MCP→PIP, and tip-to-MCP distance > pip-to-MCP distance × 0.85.
- **`thumb_open`**: Thumb tip is far from index MCP (spread > 0.42 × palm size) OR thumb tip is on the lateral side of the IP joint.
- **`_curled(tip, pip, mcp)`**: Tip is closer to MCP than PIP is — finger is tightly curled.
- **`fingers_horizontal`**: Index tip's horizontal displacement > vertical displacement × 1.1.
- **Normalized inter-tip distances** (`d_ti`, `d_tm`, `d_im`, etc.) for spread/pinch detection.

**Classification decision tree (summary):**

```
non_thumb_up == 4:
    thumb_open           → OPEN_PALM (0.93)
    !thumb_open, close   → B (0.88)

non_thumb_up == 3:
    index+middle+ring    → W (0.87) [spread], W (0.78) [less spread]
    middle+ring+pinky + pinch → F (0.86)

index_up + pinky_up + thumb_open  → ILY (0.90)
pinky_up only + thumb_open        → Y (0.88)
pinky_up only + !thumb_open       → I (0.88)

index_up + middle_up:
    horizontal + tight   → H (0.82)
    thumb_open + vertical → K (0.83)
    crossed              → R (0.82)
    spread (d_im > 0.26) → V (0.90)
    together             → U (0.86)

index_up only:
    thumb_open           → L (0.88)
    !thumb_open, d_ti < 0.32 → D (0.83)
    tip below pip (hook) → X (0.80)
    otherwise            → G (0.80)

non_thumb_up == 0:
    all tips close to thumb → O (0.84)
    partial curl + spread   → C (0.78)
    fully curled:
        T (thumb between knuckles) → T (0.78)
        thumb_open + thumb_up      → THUMB_UP (0.88)
        thumb_open + thumb_down    → THUMB_DOWN (0.88)
        thumb over knuckles        → S (0.83)
        deep curl                  → E (0.78)
        thumb alongside fist       → A (0.84)
```

---

### 5.8 PyTorch Gesture Classifier

**File:** `backend/vision/gesture_classifier.py`

**Purpose:** Optional neural network-based gesture classifier supporting 200 ASL signs. Used when `GESTURE_BACKEND=pytorch`.

**Model:** A `.pt` file loaded from the path specified by `GESTURE_MODEL_PATH`. The model takes a `(1, 63)` tensor of normalized landmark coordinates and outputs logits over 200 classes.

**VOCAB_200** includes: I, YOU, HE, SHE, WE, THEY, WANT, NEED, LIKE, LOVE, HATE, EAT, DRINK, SLEEP, WORK, PLAY, GO, COME, STOP, HELP, PLEASE, THANK_YOU, SORRY, HELLO, GOODBYE, YES, NO, MAYBE, MORE, LESS, OPEN, CLOSE, HAPPY, SAD, ANGRY, TIRED, SICK, PAIN, HOT, COLD, WATER, FOOD, MONEY, PHONE, BOOK, HOME, SCHOOL, HOSPITAL, OFFICE, and many more.

**`GestureClassifier.predict(features)`:**
- Runs `torch.no_grad()` forward pass.
- Applies `torch.softmax` to logits.
- Returns `(label, confidence, raw_label)` if top confidence ≥ threshold, else `(None, 0.0, raw_label)`.

---

### 5.9 Hand Detector

**File:** `backend/vision/hand_detector.py`

**Purpose:** Used only with the PyTorch backend. Extracts normalized 21-landmark arrays from BGR frames using `mediapipe.solutions.hands`.

**`HandDetectorConfig`:**
- `max_num_hands = 1`
- `min_detection_confidence = 0.7`
- `min_tracking_confidence = 0.7`

**`detect_landmarks(frame_bgr)`:**
1. Converts BGR → RGB.
2. Runs `mp.solutions.hands.Hands.process()`.
3. If a hand is detected, extracts all 21 `(x, y, z)` landmarks.
4. Centers on wrist (landmark 0) and normalizes by maximum distance from wrist.
5. Returns a `np.ndarray` of shape `(63,)` (21 points × 3 coordinates).

---

### 5.10 LLM Service

**File:** `backend/llm.py`

#### GroqConfig

```python
model           = "llama-3.3-70b-versatile"
temperature     = 0.2      # Low for deterministic sentence generation
max_tokens      = 40       # Short output — one sentence only
top_p           = 0.9
cache_size      = 1024     # LRU cache entries
context_turns   = 6        # Multi-turn conversation memory
```

#### Token Preprocessing — `_preprocess_tokens(tokens)`

Before sending to the LLM, consecutive single-letter tokens (fingerspelled letters) are merged into `[SPELL:WORD]` annotations:

```
["W", "A", "T", "E", "R"]  →  ["[SPELL:WATER]"]
["OPEN_PALM", "H", "I"]    →  ["OPEN_PALM", "[SPELL:HI]"]
["I"]                      →  ["I"]   (single letter left unchanged)
```

This dramatically improves LLM accuracy for fingerspelled words.

#### Prompt Architecture — `_build_messages(tokens, context, language, scenario, custom_scenario)`

The message list is built as a multi-turn chat:

```
[system]  SYSTEM_PROMPT + scenario_addendum + language_addendum
[user]    "Tokens: OPEN_PALM W A T E R"       ← from context turn 1
[assistant] "I want water please."             ← from context turn 1
...
[user]    "Tokens: THANK_YOU"                  ← current tokens
```

**SYSTEM_PROMPT** (key rules):
- Output ONLY the sentence — no explanations, no reasoning, no preamble.
- No quotation marks around output.
- Decode `[SPELL:...]` tokens directly.
- Use conversation history for context but never repeat previous sentences.
- If unable to form a sentence, output a single best-guess word.

**Scenario addenda** (injected when scenario ≠ "general"):

| Scenario | Context injected |
|----------|-----------------|
| `interview` | Job interview setting; candidate speaking formally to interviewer |
| `banking` | Bank teller interaction; deposits, withdrawals, account queries |
| `medical` | Doctor/hospital; symptoms, prescriptions, appointments |
| `restaurant` | Food ordering; waitstaff interaction, dietary preferences |
| `custom` | Free-text context provided by the user (capped at 120 chars) |

**Language addenda:**
- `hi`: Forces output in Hindi Devanagari script with an example (`धन्यवाद`).

#### LRU Cache

Cache key = `tuple("TOKENS", *tokens, "SCENARIO", scenario, "CUSTOM", custom_scenario, "CTX", ctx_token_text, ctx_sentence, ...)`

The `OrderedDict`-based LRU evicts the oldest entry when size exceeds 1024. Cache hits bypass the Groq API entirely.

#### `_clean_llm_output(text)`

A regex (`_META_PATTERN`) strips meta-reasoning prefixes that the model sometimes leaks:
- `"I interpret the sequence as: ..."`
- `"The sequence means: ..."`
- `"translates to: ..."`
- `"With the additional token X, output: ..."`

After stripping, surrounding quotes are removed.

#### Streaming

`stream_tokens()` calls `client.chat.completions.create(..., stream=True)` and yields each delta chunk. The WebSocket handler consumes these chunks via an `asyncio.Queue` and relays each partial sentence to the browser with `sentence_streaming=True`.

#### ConversationContext

Maintains a `deque(maxlen=6)` of `(token_text, sentence)` pairs. Persisted across reconnects by pre-populating from `_global_history` at WebSocket connection time.

---

### 5.11 WebSocket Endpoint

**Route:** `ws://host:8000/ws`

**File:** `backend/main.py` — `async def websocket_endpoint(websocket)`

Each WebSocket connection gets its own isolated session with:
- `TokenBuffer` — stable gesture token accumulation
- `GestureVoter` — temporal smoothing (server-side image frames only)
- `MotionClassifier` — dynamic sign detection
- `ConversationContext` — multi-turn LLM context
- `pending_llm_task` — debounced asyncio task

**LLM Debouncing:**

Every time a new token is committed, any pending LLM task is cancelled and a new one is scheduled with a `_LLM_DEBOUNCE_S` (default 500 ms) delay. This means the LLM is called only once after the user pauses signing — not once per letter during rapid signing.

**Two-phase token update:**

```
Phase 1: WebSocket → browser {tokens: [...], sentence: last_sentence}
         (immediate, so tokens appear instantly in UI)

Phase 2: (after debounce) Groq API → stream chunks → browser
         {tokens: [...], sentence: partial, sentence_streaming: true}
         ...
         {tokens: [...], sentence: full, sentence_finalized: true}
```

**Frame rate limiting (server-side path):**

`min_interval = 1.0 / max_fps` — frames arriving faster than this are silently dropped.

**Error handling:**

Exceptions in the WebSocket loop send `{"error": str(exc)}` to the client before closing the connection.

---

### 5.12 REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `GET` | `/history` | Returns `{"history": [...]}` — up to 200 persisted entries |
| `POST` | `/history/clear` | Clears all history from memory and disk |
| `POST` | `/tts` | Converts text to MP3 audio via gTTS. Body: `{text, lang, tld}` |
| `GET` | `/models/gesture_recognizer.task` | Serves the MediaPipe model binary to browser clients |

**CORS:** All origins allowed (`*`) to support the Next.js dev server on a different port.

**Response serialization:** `ORJSONResponse` is the default for all REST responses (faster than stdlib JSON).

---

### 5.13 Persistent History

**File:** `backend/conversation_history.json`

- Stored as a JSON array of `{"tokens": [...], "sentence": "..."}` objects.
- Written to disk via `asyncio.to_thread(_save_history, ...)` on every new finalized sentence (non-blocking).
- Capped at 200 entries (oldest are trimmed).
- **Continuation detection:** If new tokens are a superset prefix of the last entry's tokens, the last entry is updated in-place rather than appending. This prevents duplicate entries when the user adds one more sign to an existing phrase.

---

### 5.14 Text-to-Speech (TTS) Endpoint

**Route:** `POST /tts`
**Body:** `{text: string, lang: string, tld: string}`

- `lang`: ISO 639-1 code (`"en"` or `"hi"`)
- `tld`: Controls gTTS accent:
  - `"com"` → US English / Hindi
  - `"co.uk"` → British English
  - `"co.in"` → Indian English

**In-memory cache:** Up to 256 `(text, lang, tld)` → `bytes` entries. Cache is evicted FIFO when full. Avoids repeated API calls for the same phrase.

**Audio:** Returns `StreamingResponse` with `media_type="audio/mpeg"` (MP3).

---

## 6. Frontend — Detailed Design

### 6.1 Application Layout & Routing

**File:** `frontend/src/app/layout.tsx`

- Uses the Next.js App Router (`app/` directory).
- Root layout loads three Google Fonts as CSS variables:
  - `--font-display`: Cormorant Garamond (weights: 300, 400, 600; italic)
  - `--font-mono`: Space Mono (weights: 400, 700)
  - `--font-body`: Outfit (weights: 300, 400, 500, 600)
- Sets `<title>Sign Scribe</title>` and meta description.

**File:** `frontend/src/app/page.tsx`

Single entry point rendering `<SignStreamClient />`.

---

### 6.2 SignStreamClient — Main Orchestrator

**File:** `frontend/src/components/SignStreamClient.tsx` (685 lines)

This is the brain of the frontend. It manages all state, the WebSocket connection, the client-side gesture detection loop, and orchestrates all child components.

**State managed:**

| State | Type | Description |
|-------|------|-------------|
| `running` | `boolean` | Whether camera + detection is active |
| `tokens` | `string[]` | Current gesture token list |
| `sentence` | `string` | Current generated sentence (streaming or final) |
| `sentenceStreaming` | `boolean` | Whether sentence is still streaming |
| `sentenceFinalized` | `boolean` | Whether sentence is complete |
| `history` | `HistoryEntry[]` | Past token→sentence pairs |
| `debug` | `DebugPayload \| null` | Debug info from backend |
| `language` | `"en" \| "hi"` | Output language |
| `accent` | `string` | TTS accent (us/uk/in) |
| `scenario` | `ScenarioKey` | Context scenario |
| `customScenario` | `string` | Free-text custom context |
| `showGuide` | `boolean` | Gesture reference guide visibility |
| `calibrating` | `boolean` | Calibration modal state |

**WebSocket lifecycle:**
- Opens `ws://${API_BASE}/ws` on start.
- On each incoming binary message: parses `orjson`-serialized `WebSocketResponse`.
- Sends gesture labels as JSON or JPEG frames as binary depending on which detection mode is active.
- On disconnect: cleans up the detection loop and camera stream.

**Client-side detection loop (`requestAnimationFrame`):**
1. If `ClientGestureDetector` is `ready`: detect gesture on current video frame, send `{"type": "gesture", "label": ..., "confidence": ...}` JSON message.
2. If detector failed to initialize: capture video frame to canvas as JPEG, send raw bytes as binary WebSocket message (server-side fallback).
3. Loop continues at display frame rate (~60 FPS), but server enforces 15 FPS cap.

**Auto-speak flow:**
- When `sentence_finalized=true` arrives and auto-speech is enabled, `speak()` is called from `useTTS`.
- The TTS hook shows a 2.5 s countdown; after it expires, the sentence is spoken.
- If the user makes a new gesture within 4 s of finalization, the auto-clear and speak are cancelled.

---

### 6.3 Client-Side Gesture Classifier

**File:** `frontend/src/lib/gestureClassifier.ts` (626 lines)

**Purpose:** Run MediaPipe gesture detection entirely in the browser, eliminating the need to stream raw video frames to the server. This reduces bandwidth and latency significantly.

**Class: `ClientGestureDetector`**

States: `idle` → `loading` → `ready` | `error`

**`initialize()`:**
1. Fetches the `gesture_recognizer.task` model binary from `/models/gesture_recognizer.task` on the backend.
2. Creates a `GestureRecognizer` instance from `@mediapipe/tasks-vision` using the fetched model.
3. Suppresses verbose MediaPipe WASM/TensorFlow console logs by patching `console.error`.

**`detect(videoElement, timestamp)`:**
1. Calls `gestureRecognizer.recognizeForVideo(videoElement, timestamp)`.
2. Runs the browser-side ASL rule classifier on detected landmarks.
3. Runs the browser-side motion classifier on the landmark history.
4. Motion result (if confidence ≥ 0.75) takes priority over static pose.
5. Returns `{label, confidence, handTracked}`.

**Browser-side ASL Classifier** (TypeScript port of Python `_classify_asl`):
- Identical geometric logic to the Python backend.
- Classifies all ASL letters and common signs from 21 normalized landmarks.
- Ensures that client-side detection produces consistent results with server-side fallback.

**Browser-side Motion Classifier** (TypeScript):
- Maintains a 25-frame deque of landmark arrays.
- Implements all 8 dynamic sign detectors with identical geometry to the Python version.
- Per-gesture cooldown tracked via `Date.now()`.

**Calibration support:**
- `sampleConfidence()`: captures 60 frames of OPEN_PALM confidence samples.
- Computes the 20th percentile, scales to 85%, saves to `localStorage` as confidence threshold.
- `loadCalibrationThreshold()` / `saveCalibrationThreshold()` for persistence.

---

### 6.4 useTTS Hook

**File:** `frontend/src/hooks/useTTS.ts`

**Purpose:** Manage text-to-speech with a 2.5 s countdown review window before auto-speak.

**Accent to gTTS parameter mapping:**

| UI Accent | Language | `lang` | `tld` |
|-----------|----------|--------|-------|
| US English | en | `en` | `com` |
| British English | en | `en` | `co.uk` |
| Indian English | en | `en` | `co.in` |
| Hindi | hi | `hi` | `com` |

**`speak(text, lang, accent)`:**
1. Posts to `POST /tts` with mapped parameters.
2. Decodes MP3 response, creates an `AudioContext`, decodes audio data, plays via `AudioBufferSourceNode`.
3. Fallback: If network request fails, uses `window.speechSynthesis` (Web Speech API) with locale matching.

**`speakNow(text)`:** Skips the countdown and speaks immediately.

**Countdown timer:** Uses `requestAnimationFrame` to animate `speakCountdownPct` from 0→100% over 2500 ms, then auto-calls `speakNow`.

---

### 6.5 UI Components

**`CameraPanel.tsx`**
- Renders `<video>` element with live webcam feed and art-deco corner bracket overlay.
- Shows a badge: green "Hand Detected" / amber "Unknown gesture" / gray "No Hand".
- Overlays the current gesture label and confidence percentage.
- Shows a stability progress bar that fills as the pending gesture approaches the stability threshold.

**`TokensSection.tsx`**
- Renders each token as a chip with a delete (×) button.
- Manual token input via `<input>` with `<datalist>` autocomplete from the full gesture vocabulary.
- "Regenerate" button sends `{"type": "set_tokens", "tokens": [...]}` to re-run the LLM on the current token list.

**`SentenceSection.tsx`**
- Shows the generated sentence with a blinking cursor during streaming.
- "Speak Now" button skips the countdown.
- Countdown bar (amber gradient) animates the 2.5 s review window.
- "Cancel" button aborts the pending speak.
- "New Sentence" button to dismiss finalized state.

**`HistorySection.tsx`**
- Scrollable list of past `{tokens, sentence}` pairs.
- Fetches history from `GET /history` on mount.
- Updates live as new sentences are finalized.

**`ControlsPanel.tsx`**
- Primary row: Start (▶), Stop (⏹), Clear (🗑)
- Secondary row: Speech toggle, Auto-clear toggle, Guide toggle, Clear History

**`SettingsStrip.tsx`**
- Language radio: English / हिंदी
- Accent selector (hidden for Hindi): US / UK / Indian
- Scenario pills: General / Interview / Bank / Medical / Restaurant / Custom
- Custom scenario text input (max 120 chars, visible only when Custom selected)
- Sends `{"type": "config", ...}` to WebSocket on any change.

**`GuideSection.tsx`**
- Collapsible reference card grid showing 21 supported gestures.
- Each card: emoji + label + hand position description + meaning.
- Categories: Common Signs (5), ASL Letters A–I (8), ASL Letters K–Y (8).

**`DebugSection.tsx`**
- Collapsible debug console updated every frame.
- Shows: backend type (client-js / mediapipe / pytorch), hand tracked, raw label, accepted label, confidence, pending gesture, elapsed stability time, effective stability window, whether token was updated.

**`CalibrationModal.tsx`**
- Phase 1 (idle): Instructions to hold OPEN_PALM in front of camera.
- Phase 2 (sampling): Progress bar while 60 frames are captured.
- Phase 3 (done): Shows computed threshold percentage.
- Prevents interaction with the rest of the app while open.

---

### 6.6 Design System & CSS

**File:** `frontend/src/app/globals.css` (1468 lines)

The design system is named "Human Edge — Silent Cinema Design." It uses CSS custom properties as design tokens:

**Color palette:**

| Token | Value | Use |
|-------|-------|-----|
| `--bg-void` | `#0B0A08` | Page background |
| `--bg-surface` | `#111009` | Card/panel backgrounds |
| `--bg-elevated` | `#18160E` | Elevated elements |
| `--accent-gold` | `#C49830` | Primary accent — borders, badges, highlights |
| `--accent-gold-dim` | `#8B6B20` | Dimmed accent |
| `--text-primary` | `#E8E0CC` | Main text |
| `--text-secondary` | `#8A8070` | Secondary/muted text |
| `--c-blue` | `#4A9EBF` | Info / streaming state |
| `--c-amber` | `#D4A017` | Warning / countdown |
| `--c-red` | `#C0392B` | Error / stop |
| `--c-green` | `#2ECC71` | Success / hand detected |
| `--c-violet` | `#9B59B6` | Special states |

**Layout:**
- Two-column responsive grid: 420px fixed sidebar (camera panel) + `1fr` main column.
- Section cards with a 3px left `--accent-gold` border.
- Art-deco corner bracket overlays on the camera panel (pure CSS, no SVG).

**Animations:**
- `scan-line`: Animated golden scan line sweeping vertically over the camera feed.
- `blink`: Cursor blink for streaming sentence.
- `spin-slow`: Loading spinner.
- `fadeIn`, `slideUp`: Entry animations for cards and tokens.
- `countdown-fill`: Progress bar animation for TTS countdown.

---

## 7. Data Flow & Communication Protocol

### 7.1 WebSocket Message Types

**Client → Server:**

| Type | Format | Description |
|------|--------|-------------|
| Binary frame | Raw JPEG bytes | Server-side gesture detection (fallback mode) |
| `gesture` | `{"type":"gesture","label":"OPEN_PALM","confidence":0.91}` | Client-detected gesture |
| `config` | `{"type":"config","language":"en","scenario":"banking","custom_scenario":"..."}` | Session configuration |
| `reset` | `{"type":"reset"}` | Clear all tokens and sentence |
| `set_tokens` | `{"type":"set_tokens","tokens":["THANK","YOU"]}` | Manual token correction |

**Server → Client:**

All server messages are binary `orjson`-serialized `WebSocketResponse` objects:

```typescript
{
  tokens: string[];           // Current committed token list
  sentence: string;          // Current sentence (partial or final)
  sentence_finalized: boolean; // True when LLM response is complete
  sentence_streaming: boolean; // True while LLM is still generating
  debug?: {
    backend: string;          // "client-js" | "mediapipe" | "pytorch"
    hand_tracked: boolean;
    raw_label: string | null;  // Per-frame raw detection
    accepted_label: string | null; // After voter smoothing
    confidence: number;
    token_updated: boolean;    // Whether a token was just committed
    pending_label: string | null; // Gesture being held but not yet committed
    pending_elapsed_ms: number; // How long pending gesture has been held
    stable_window_ms: number;  // Effective stability threshold
  }
}
```

---

### 7.2 End-to-End Frame Lifecycle

```
1. Browser: requestAnimationFrame fires

2. ClientGestureDetector.detect(videoElement, timestamp)
   → MediaPipe WASM runs GestureRecognizer on current frame
   → Returns hand landmarks (21 points)
   → ASL rule classifier → (label, confidence)
   → Motion classifier → (motion_label, motion_confidence)
   → Best result returned

3. Browser: WebSocket.send(JSON.stringify({type:"gesture", label, confidence}))

4. Backend WebSocket handler receives JSON message
   → _parse_message() → msg_type="gesture"

5. If label ∈ _MOTION_LABELS:
   → token_buffer.add_token(label) directly

   Else:
   → token_buffer.process_prediction(label, confidence, now)
     → Adaptive stability window
     → If held long enough → token_buffer.add_token(label)

6. If token was updated:
   → Phase 1: send {tokens, sentence=last_sentence, debug}

   → Schedule _debounced_gesture() task (500ms delay)
     → After 500ms silence: _send_sentence_async(tokens)
       → Check LRU cache
       → If cached: send {tokens, sentence, sentence_finalized=True}
       → Else: Groq stream_tokens()
           → yield chunk → asyncio.Queue → WebSocket {sentence_streaming=True}
           → When done → _clean_llm_output() → cache → persist history
           → send {tokens, sentence, sentence_finalized=True}

7. Browser: receives {sentence_finalized: true}
   → if autoSpeak: useTTS.speak() → 2.5s countdown → gTTS MP3 or Web Speech
```

---

### 7.3 Sentence Streaming Flow

```
Time →
  T=0ms   Token committed (e.g., user finished signing "THANK_YOU")
  T=0ms   Phase 1: {tokens:["THANK_YOU"], sentence:"", sentence_finalized:false}
  T=500ms LLM call fires (debounce elapsed)
  T=520ms Chunk "Thank" → {sentence:"Thank", sentence_streaming:true}
  T=535ms Chunk " you" → {sentence:"Thank you", sentence_streaming:true}
  T=550ms Chunk "." → {sentence:"Thank you.", sentence_streaming:true}
  T=552ms Stream done → {sentence:"Thank you.", sentence_finalized:true}
  T=552ms TTS countdown starts (2.5s)
  T=3052ms Sentence spoken aloud
```

---

## 8. Gesture Vocabulary

**Total: 44 supported tokens**

### Static Hand Poses (ASL)

| Token | Hand Shape | Meaning/Use |
|-------|-----------|-------------|
| `OPEN_PALM` | All 5 fingers extended, thumb spread | Hello / Stop / 5 |
| `THUMB_UP` | Fist with thumb pointing up | Good / Yes / Agree |
| `THUMB_DOWN` | Fist with thumb pointing down | Bad / No / Disagree |
| `V` | Index + middle extended, spread | Victory / Peace / 2 |
| `ILY` | Index + pinky + thumb extended | I Love You |
| `A` | Fist with thumb alongside | Letter A |
| `B` | Four fingers together, thumb tucked | Letter B |
| `C` | Curved fingers forming C-shape | Letter C |
| `D` | Index up, thumb-index forming circle | Letter D |
| `E` | Fingers deeply curled | Letter E |
| `F` | Middle-ring-pinky up, index-thumb pinch | Letter F |
| `G` | Index pointing sideways | Letter G |
| `H` | Index + middle horizontal | Letter H |
| `I` | Pinky only up | Letter I |
| `K` | Index + middle up, thumb open | Letter K |
| `L` | Index up + thumb out (L-shape) | Letter L |
| `O` | All fingertips meeting thumb (circle) | Letter O |
| `R` | Index crossed over middle | Letter R |
| `S` | Fist with thumb over fingers | Letter S |
| `T` | Thumb between index and middle | Letter T |
| `U` | Index + middle up, together | Letter U |
| `W` | Index + middle + ring up | Letter W / Water |
| `X` | Index hook (tip below pip) | Letter X |
| `Y` | Pinky + thumb extended | Letter Y |

### Dynamic Signs (Motion-Based)

| Token | Motion | Meaning |
|-------|--------|---------|
| `WAVE` | Wrist oscillates horizontally ≥3 times, open hand | Hello / Wave |
| `THANK_YOU` | Open hand sweeps down from chin | Thank you |
| `PLEASE` | Open palm circles on chest | Please |
| `SORRY` | Closed fist circles on chest | Sorry / Apologize |
| `YES_NOD` | Closed fist bounces up-down ≥3 times | Yes |
| `MORE` | Pinch open/close ≥3 times | More |
| `EAT` | Hand moves toward mouth ≥2 times | Eat / Food |
| `COME` | Net downward/inward wrist motion | Come / Here |

---

## 9. LLM Prompt Engineering

### System Prompt (full)

```
You are an ASL (American Sign Language) interpreter.
You receive tokens that are either ASL sign labels (e.g. OPEN_PALM, THUMB_UP, ILY)
or individual fingerspelled letters (e.g. H E L L O → Hello, W A T E R → Water).
Tokens wrapped in [SPELL:...] are pre-grouped fingerspelled words — decode them directly
(e.g. [SPELL:WATER] → Water, [SPELL:HELLO] → Hello, [SPELL:NAME] → Name).
Convert the full token sequence into ONE short, natural English sentence.
Use conversation history for context only — do not repeat previous sentences.
Rules you must follow without exception:
- Output ONLY the sentence. Nothing else.
- No explanations, no reasoning, no preamble, no meta-commentary.
- No phrases like 'I interpret', 'the sequence means', 'with token', 'as:'.
- No quotation marks around the sentence.
- If you cannot form a sentence, output a single best-guess word.
Your entire response must be the sentence itself and nothing more.
```

### Model Parameters

```
model:       llama-3.3-70b-versatile
temperature: 0.2   (near-deterministic, consistent output)
max_tokens:  40    (enforces brevity — one sentence)
top_p:       0.9
stream:      True  (real-time chunk delivery to browser)
```

### Output Cleanup — `_clean_llm_output`

Regex pattern strips common meta-reasoning leaks:
```python
r".+?(?:
  interpret(?:ed|s)?(?: the sequence)? as[:\s]+
  |(?:the )?sequence (?:means?|is)[:\s]+
  |translates? to[:\s]+
  |(?:can be )?(?:read|expressed) as[:\s]+
  |with (?:the )?(?:additional )?token .+?[:,]\s*
  |output[:\s]+
  |sentence[:\s]+
)"
```

---

## 10. Calibration System

**Purpose:** Personalize the confidence threshold for each user's lighting conditions and webcam quality.

**Procedure:**

1. User opens the calibration modal (camera must be running).
2. User holds OPEN_PALM in front of the camera.
3. The browser captures 60 frames of confidence samples.
4. The 20th percentile of samples is computed (robust to outliers).
5. The threshold is set to `p20 × 0.85` — 15% below the typical good-condition confidence.
6. The threshold is saved to `localStorage` under key `sign-scribe-calibration-threshold`.
7. All subsequent `ClientGestureDetector.detect()` calls use this threshold.

**Why p20 × 0.85:** The 20th percentile represents a "slightly below average" confidence sample, accounting for brief occlusions or hand position variation. The 0.85 factor provides a 15% margin below this floor so the sign is still accepted during normal variation.

---

## 11. Configuration Reference

### `.env` (full example)

```bash
# LLM
GROQ_API_KEY=your_groq_api_key_here
CONVERSATION_CONTEXT_TURNS=6
LLM_DEBOUNCE_MS=500

# Gesture backend
GESTURE_BACKEND=mediapipe         # "mediapipe" | "pytorch"
GESTURE_MODEL_PATH=               # Required only for pytorch backend

# Frame limits
MAX_FRAME_BYTES=300000
MAX_FRAME_WIDTH=1280
MAX_FRAME_HEIGHT=720
MAX_FPS=15

# Stability
STABLE_WINDOW_MS=100

# MediaPipe thresholds
CONFIDENCE_THRESHOLD=0.15
MEDIAPIPE_CONFIDENCE_THRESHOLD=0.15
MEDIAPIPE_MIN_DETECTION_CONFIDENCE=0.25
MEDIAPIPE_MIN_TRACKING_CONFIDENCE=0.25
MEDIAPIPE_STATIC_IMAGE_MODE=true
MEDIAPIPE_TRY_FLIPPED_FRAME=true

# Trained ASL model (optional)
USE_TRAINED_ASL=1
TRAINED_ASL_THRESHOLD=0.70
TRAINED_ASL_TTA=1

# Debug
EMIT_DEBUG_FRAMES=true

# Frontend (optional — override API endpoint)
NEXT_PUBLIC_API_BASE=localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 12. Dependencies & Libraries

### Backend Python Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Async web framework, WebSocket support |
| `uvicorn` | ASGI server (with `websockets` transport) |
| `mediapipe` | Hand landmark detection + GestureRecognizer |
| `opencv-python` (`cv2`) | Image decode, CLAHE, gamma correction, color conversion |
| `numpy` | Landmark arrays, numerical operations |
| `groq` | Groq API Python client |
| `gtts` | Google Text-to-Speech (lazy-imported in `/tts` handler) |
| `orjson` | Fast JSON serialization/deserialization |
| `pydantic` | Data validation (`BaseModel`, `dataclass`) |
| `python-dotenv` | `.env` file loading |
| `torch` | PyTorch (optional — only loaded when `GESTURE_BACKEND=pytorch`) |

### Frontend npm Dependencies

| Package | Purpose |
|---------|---------|
| `next` | React meta-framework (App Router, SSR, font optimization) |
| `react` | UI component library |
| `react-dom` | React DOM renderer |
| `@mediapipe/tasks-vision` | MediaPipe WebAssembly gesture recognition in browser |

### Frontend Dev Dependencies

| Package | Purpose |
|---------|---------|
| `typescript` | Type system |
| `@types/node` | Node.js type definitions |
| `@types/react` | React type definitions |
| `@types/react-dom` | React DOM type definitions |

---

## 13. Security Considerations

| Area | Status | Notes |
|------|--------|-------|
| API Key exposure | Risk | `GROQ_API_KEY` in `.env` — use secrets manager in production |
| CORS | Open | `allow_origins=["*"]` — restrict to known origins in production |
| Frame validation | Implemented | Size (300KB), dimensions (1280×720) enforced per frame |
| Input sanitization | Partial | `set_tokens` sanitizes to uppercase alphanumeric; custom scenario capped at 120 chars |
| WebSocket authentication | Not implemented | No auth token required — add in production |
| History file access | Local only | JSON file on server disk, not exposed directly |
| TTS cache | In-memory | Cleared on restart; no persistence of audio |
| Frame rate limiting | Implemented | Max 15 FPS enforced server-side |

---

## 14. Known Limitations & Trade-offs

| Limitation | Detail |
|-----------|--------|
| Single hand only | Both MediaPipe and PyTorch backends detect one hand at a time (`max_num_hands=1`) |
| Static image mode | MediaPipe runs in `static_image_mode=True` (server-side) — no temporal tracking benefit between frames; compensated by GestureVoter |
| Motion signs require 25 frames | At 15 FPS, this is ~1.7 s of context. Fast dynamic signs may not be detected reliably |
| ASL alphabet coverage | Letters J and Z require motion (not currently classified as static poses) |
| LLM max_tokens=40 | Long sentences may be truncated; configurable but trade-off against cost/speed |
| No user accounts | History is global across all sessions (single-user design) |
| Hindi support | Output only — input ASL vocabulary is English-centric |
| Browser TTS fallback | Web Speech API voice quality varies significantly by OS and browser |
| PyTorch model | Not included in the repo — must be trained separately using `train_asl_classifier.py` |
| Windows Uvicorn loop | Uses `asyncio` event loop on Windows (slower than `uvloop` on Linux/macOS) |
