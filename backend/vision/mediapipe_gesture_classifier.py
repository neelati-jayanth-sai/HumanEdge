from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

_TASK_MODEL_PATH = Path(__file__).parent.parent / "models" / "gesture_recognizer.task"
_TRAINED_MODEL_PATH = Path(__file__).parent.parent / "models" / "asl_classifier.pth"

# ---------------------------------------------------------------------------
# Optional trained model — loaded when USE_TRAINED_ASL=1
# ---------------------------------------------------------------------------

_trained_predictor = None   # type: ignore[assignment]

def _try_load_trained_predictor() -> None:
    global _trained_predictor
    if not (os.getenv("USE_TRAINED_ASL", "").strip().lower() in {"1", "true", "yes"}):
        return
    if not _TRAINED_MODEL_PATH.exists():
        logging.warning(
            f"USE_TRAINED_ASL=1 but model not found at {_TRAINED_MODEL_PATH}. "
            "Run: python -m backend.train_asl_classifier --dataset-dir <path>"
        )
        return
    try:
        from backend.train_asl_classifier import ASLPredictor
        _trained_predictor = ASLPredictor.load(
            _TRAINED_MODEL_PATH,
            threshold=float(os.getenv("TRAINED_ASL_THRESHOLD", "0.70")),
            tta=os.getenv("TRAINED_ASL_TTA", "1").strip() in {"1", "true", "yes"},
        )
        logging.info("Trained ASL classifier loaded — rule-based classifier disabled.")
    except Exception as exc:
        logging.warning(f"Failed to load trained ASL predictor: {exc}")

_try_load_trained_predictor()


@dataclass(slots=True)
class MediaPipeGestureClassifierConfig:
    confidence_threshold: float = 0.65
    max_num_hands: int = 1
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.60
    static_image_mode: bool = True
    try_flipped_frame: bool = False


class MediaPipeGestureClassifier:
    """
    Gesture classifier with two modes:
    1. Primary: MediaPipe Tasks GestureRecognizer (.task model).
       The built-in recognizer returns only 8 gestures, but also gives us 21
       hand landmarks — we run our own ASL rule-based classifier on those
       landmarks, so the full ASL alphabet + common signs are available.
    2. Fallback: MediaPipe Hands + ASL rule classifier (same logic).
    """

    def __init__(self, config: MediaPipeGestureClassifierConfig) -> None:
        self.config = config
        self.threshold = config.confidence_threshold
        self._recognizer = None  # MediaPipe Tasks GestureRecognizer
        self._hands = None       # Legacy MediaPipe Hands solution
        self.last_landmarks: list[tuple[float, float, float]] | None = None

        if _TASK_MODEL_PATH.exists():
            try:
                self._init_task_recognizer()
            except Exception as exc:
                print(
                    f"[MediaPipeGestureClassifier] Task model init failed ({exc}), "
                    "falling back to Hands solution."
                )
                self._init_hands_solution()
        else:
            print(
                f"[MediaPipeGestureClassifier] Task model not found at {_TASK_MODEL_PATH}, "
                "using Hands solution."
            )
            self._init_hands_solution()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_task_recognizer(self) -> None:
        from mediapipe.tasks import python as mp_python  # type: ignore[import]
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore[import]

        detect_conf = self.config.min_detection_confidence
        track_conf = self.config.min_tracking_confidence

        options = mp_vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(_TASK_MODEL_PATH)
            ),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=self.config.max_num_hands,
            min_hand_detection_confidence=detect_conf,
            min_hand_presence_confidence=track_conf,
            min_tracking_confidence=track_conf,
        )
        self._recognizer = mp_vision.GestureRecognizer.create_from_options(options)
        print(
            f"[MediaPipeGestureClassifier] Using .task GestureRecognizer "
            f"(detect_conf={detect_conf:.2f}, track_conf={track_conf:.2f})"
        )

    def _init_hands_solution(self) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=self.config.static_image_mode,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            model_complexity=0,
        )
        print(
            f"[MediaPipeGestureClassifier] Using Hands solution "
            f"(detect_conf={self.config.min_detection_confidence:.2f})"
        )

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _clahe_enhance(bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _gamma_brighten(bgr: np.ndarray, gamma: float = 1.6) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
        )
        return cv2.LUT(bgr, lut)

    def _build_variants(self, frame_bgr: np.ndarray) -> list[tuple[np.ndarray, bool]]:
        base_frames: list[tuple[np.ndarray, bool]] = [
            (frame_bgr, False),
            (self._clahe_enhance(frame_bgr), False),
            (self._gamma_brighten(frame_bgr), False),
        ]
        if self.config.try_flipped_frame:
            flipped = cv2.flip(frame_bgr, 1)
            base_frames += [
                (flipped, True),
                (self._clahe_enhance(flipped), True),
                (self._gamma_brighten(flipped), True),
            ]
        return [
            (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), is_flipped)
            for bgr, is_flipped in base_frames
        ]

    # ------------------------------------------------------------------
    # Public predict interface
    # ------------------------------------------------------------------

    def predict(self, frame_bgr: np.ndarray) -> tuple[str | None, float, bool, str | None]:
        self.last_landmarks = None  # Reset each call; set only when a hand is detected
        if self._recognizer is not None:
            return self._predict_task(frame_bgr)
        return self._predict_hands(frame_bgr)

    # ------------------------------------------------------------------
    # Task-model path
    # ------------------------------------------------------------------

    def _predict_task(
        self, frame_bgr: np.ndarray
    ) -> tuple[str | None, float, bool, str | None]:
        best_hand_tracked = False
        best_raw: str | None = None
        best_score = 0.0
        # Best result from ASL landmark classifier across all variants
        best_asl_label: str | None = None
        best_asl_score = 0.0

        for rgb, is_flipped in self._build_variants(frame_bgr):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                result = self._recognizer.recognize(mp_image)
            except Exception:
                continue

            if result.hand_landmarks:
                best_hand_tracked = True

                # ── Run ASL classifier on all detected hands ──────────────────
                for h_idx in range(len(result.hand_landmarks)):
                    hand_lm = result.hand_landmarks[h_idx]
                    pts = [(lm.x, lm.y, lm.z) for lm in hand_lm]
                    # Store first hand's landmarks for motion classifier
                    if h_idx == 0 and not is_flipped:
                        self.last_landmarks = pts

                    handedness = "Right"
                    if result.handedness and h_idx < len(result.handedness):
                        raw_side = result.handedness[h_idx][0].category_name
                        # Flip label when we fed a mirrored image
                        handedness = ("Left" if raw_side == "Right" else "Right") if is_flipped else raw_side

                    asl_label, asl_score = self._classify_asl(pts, handedness)
                    if asl_label and asl_score > best_asl_score:
                        best_asl_score = asl_score
                        best_asl_label = asl_label

                    # If ASL classifier is confident enough, return immediately
                    if asl_label and asl_score >= self.threshold:
                        return asl_label, asl_score, True, asl_label

            # ── Fall back to built-in task gesture ───────────────────────────
            if not result.gestures:
                continue

            gesture = result.gestures[0][0]
            raw_label: str = gesture.category_name
            score = float(gesture.score)

            if raw_label in ("None", ""):
                continue

            if score > best_score:
                best_score = score
                best_raw = raw_label

            if score < self.threshold:
                continue

            # If ASL classifier had a medium-confidence result, prefer it
            if best_asl_label and best_asl_score >= self.threshold * 0.8:
                return best_asl_label, best_asl_score, True, best_asl_label

            label = self._normalize_task_label(raw_label)
            return label, score, True, raw_label

        # Nothing passed threshold — return best sub-threshold result for debug only (label=None)
        if best_hand_tracked and best_asl_label and best_asl_score > best_score:
            return None, best_asl_score, best_hand_tracked, best_asl_label
        return None, best_score, best_hand_tracked, best_raw

    # ------------------------------------------------------------------
    # Hands-solution path (fallback)
    # ------------------------------------------------------------------

    def _predict_hands(
        self, frame_bgr: np.ndarray
    ) -> tuple[str | None, float, bool, str | None]:
        if self._hands is None:
            return None, 0.0, False, None

        best_hand_tracked = False
        best_raw: str | None = None
        best_score = 0.0

        for rgb, is_flipped in self._build_variants(frame_bgr):
            results = self._hands.process(rgb)
            if not results.multi_hand_landmarks:
                continue

            best_hand_tracked = True

            # Process all detected hands; keep the highest-confidence result.
            for h_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = "Right"
                if results.multi_handedness and h_idx < len(results.multi_handedness):
                    raw_side = results.multi_handedness[h_idx].classification[0].label
                    handedness = ("Left" if raw_side == "Right" else "Right") if is_flipped else raw_side

                points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                # Store first hand's landmarks for motion classifier
                if h_idx == 0 and not is_flipped:
                    self.last_landmarks = points
                raw_label, score = self._classify_asl(points, handedness)

                if score > best_score:
                    best_score = score
                    best_raw = raw_label

                if raw_label is None or score < self.threshold:
                    continue

                return raw_label, score, True, raw_label

        return None, best_score, best_hand_tracked, best_raw

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_label(label: str) -> str:
        token = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_")
        return token.upper()

    @staticmethod
    def _normalize_task_label(label: str) -> str:
        """Map the 8 built-in task gesture names to clean tokens."""
        _MAP = {
            "Open_Palm": "OPEN_PALM",
            "Closed_Fist": "FIST",
            "Pointing_Up": "POINTING",
            "Thumb_Up": "THUMB_UP",
            "Thumb_Down": "THUMB_DOWN",
            "Victory": "V",
            "ILoveYou": "ILY",
        }
        return _MAP.get(label, re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_").upper())

    @staticmethod
    def _dist2d(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _finger_extended(
        tip: tuple[float, float, float],
        pip: tuple[float, float, float],
        mcp: tuple[float, float, float],
    ) -> bool:
        vx = pip[0] - mcp[0]
        vy = pip[1] - mcp[1]
        wx = tip[0] - pip[0]
        wy = tip[1] - pip[1]
        dot = vx * wx + vy * wy
        dist_mcp_tip = ((tip[0] - mcp[0]) ** 2 + (tip[1] - mcp[1]) ** 2) ** 0.5
        dist_mcp_pip = ((pip[0] - mcp[0]) ** 2 + (pip[1] - mcp[1]) ** 2) ** 0.5
        return dot > 0 and dist_mcp_tip > dist_mcp_pip * 0.85

    # ------------------------------------------------------------------
    # Comprehensive ASL rule-based classifier
    # ------------------------------------------------------------------

    def _classify_asl(
        self,
        points: list[tuple[float, float, float]],
        handedness: str,
    ) -> tuple[str | None, float]:
        # Use trained model when available — higher accuracy than rule-based
        if _trained_predictor is not None:
            return _trained_predictor.predict(points, handedness)
        """
        Classify ASL letters and common signs from 21 MediaPipe hand landmarks.

        Supports: A B C D E F I K L O R S U V W X Y
                  OPEN_PALM  THUMB_UP  THUMB_DOWN  ILY
        """
        if len(points) != 21:
            return None, 0.0

        # ── Unpack key landmarks ────────────────────────────────────────
        wrist = points[0]
        # thumb: cmc=1, mcp=2, ip=3, tip=4
        thumb_tip, thumb_ip = points[4], points[3]
        # index: mcp=5, pip=6, dip=7, tip=8
        index_tip,  index_pip,  index_mcp  = points[8],  points[6],  points[5]
        # middle: mcp=9, pip=10, dip=11, tip=12
        middle_tip, middle_pip, middle_mcp = points[12], points[10], points[9]
        # ring: mcp=13, pip=14, dip=15, tip=16
        ring_tip,   ring_pip,   ring_mcp   = points[16], points[14], points[13]
        # pinky: mcp=17, pip=18, dip=19, tip=20
        pinky_tip,  pinky_pip,  pinky_mcp  = points[20], points[18], points[17]

        palm_size = max(1e-6, self._dist2d(wrist, points[9]))

        # ── Extension (vector-based, tilt-robust) ───────────────────────
        index_up  = self._finger_extended(index_tip,  index_pip,  index_mcp)
        middle_up = self._finger_extended(middle_tip, middle_pip, middle_mcp)
        ring_up   = self._finger_extended(ring_tip,   ring_pip,   ring_mcp)
        pinky_up  = self._finger_extended(pinky_tip,  pinky_pip,  pinky_mcp)

        # ── Thumb ───────────────────────────────────────────────────────
        # Spread: tip far from index MCP
        thumb_spread = self._dist2d(thumb_tip, points[5]) / palm_size > 0.42
        # Directional open: for right hand, tip is left of IP joint
        if handedness == "Right":
            thumb_dir_open = thumb_tip[0] < thumb_ip[0]
        else:
            thumb_dir_open = thumb_tip[0] > thumb_ip[0]
        thumb_open = thumb_spread or thumb_dir_open

        # ── Tight-curl for each finger ──────────────────────────────────
        def _curled(tip_p, pip_p, mcp_p):
            return self._dist2d(tip_p, mcp_p) < self._dist2d(pip_p, mcp_p) * 0.92

        index_curled  = _curled(index_tip,  index_pip,  index_mcp)
        middle_curled = _curled(middle_tip, middle_pip, middle_mcp)
        ring_curled   = _curled(ring_tip,   ring_pip,   ring_mcp)
        pinky_curled  = _curled(pinky_tip,  pinky_pip,  pinky_mcp)

        non_thumb_up = sum([index_up, middle_up, ring_up, pinky_up])

        # ── Finger direction: horizontal vs vertical ─────────────────────
        # Used to distinguish H (sideways) from U / V / K (upward).
        idx_dx = abs(index_tip[0] - index_mcp[0])
        idx_dy = abs(index_tip[1] - index_mcp[1])
        fingers_horizontal = index_up and (idx_dx > idx_dy * 1.1)

        # ── Normalized inter-landmark distances ─────────────────────────
        def d(a, b):
            return self._dist2d(a, b) / palm_size

        d_ti   = d(thumb_tip, index_tip)    # thumb ↔ index tip
        d_tm   = d(thumb_tip, middle_tip)   # thumb ↔ middle tip
        d_tr   = d(thumb_tip, ring_tip)     # thumb ↔ ring tip
        d_tp   = d(thumb_tip, pinky_tip)    # thumb ↔ pinky tip
        d_im   = d(index_tip, middle_tip)   # index ↔ middle spread
        d_mr   = d(middle_tip, ring_tip)    # middle ↔ ring spread
        d_rp   = d(ring_tip,  pinky_tip)    # ring ↔ pinky spread

        # ──────────────────────────────────────────────────────────────────
        # 4 fingers up
        # ──────────────────────────────────────────────────────────────────

        # B — four fingers straight up and together, thumb tucked
        if non_thumb_up == 4 and not thumb_open and d_im < 0.22:
            return "B", 0.88

        # OPEN_PALM / 5 / HELLO — four fingers up, thumb spread
        if non_thumb_up == 4 and thumb_open:
            return "OPEN_PALM", 0.93

        # ──────────────────────────────────────────────────────────────────
        # 3 fingers up
        # ──────────────────────────────────────────────────────────────────

        # W — index + middle + ring up, spread (also WATER)
        if index_up and middle_up and ring_up and not pinky_up:
            if d_im > 0.14 and d_mr > 0.14:
                return "W", 0.87
            return "W", 0.78   # less spread variant

        # F — middle + ring + pinky up, index + thumb pinch
        if middle_up and ring_up and pinky_up and not index_up and d_ti < 0.30:
            return "F", 0.86

        # ──────────────────────────────────────────────────────────────────
        # Exactly 2 special fingers (index + pinky combos)
        # ──────────────────────────────────────────────────────────────────

        # ILY — index + pinky + thumb out (ASL "I Love You")
        if index_up and pinky_up and not middle_up and not ring_up and thumb_open:
            return "ILY", 0.90

        # ──────────────────────────────────────────────────────────────────
        # Pinky-only cases (determine before 2-finger cases)
        # ──────────────────────────────────────────────────────────────────

        # Y — thumb + pinky out, other 3 curled
        if pinky_up and not index_up and not middle_up and not ring_up and thumb_open:
            return "Y", 0.88

        # I — only pinky up, thumb tucked
        if pinky_up and not index_up and not middle_up and not ring_up and not thumb_open:
            return "I", 0.88

        # ──────────────────────────────────────────────────────────────────
        # 2 fingers up (index + middle)
        # ──────────────────────────────────────────────────────────────────

        if index_up and middle_up and not ring_up and not pinky_up:
            # H — index + middle extended sideways (horizontal), not upward
            if fingers_horizontal and d_im < 0.22 and not thumb_open:
                return "H", 0.82

            # K — thumb open + fingers pointing upward (not horizontal)
            if thumb_open and not fingers_horizontal:
                return "K", 0.83

            if not fingers_horizontal:
                # R — index tip crosses over middle tip
                if handedness == "Right":
                    crossed = index_tip[0] > middle_tip[0]
                else:
                    crossed = index_tip[0] < middle_tip[0]
                if crossed and d_im < 0.18:
                    return "R", 0.82

                # V — spread
                if d_im > 0.26:
                    return "V", 0.90

                # U — together
                return "U", 0.86

            # Horizontal + thumb open — angled H/K variant
            return "H", 0.78

        # ──────────────────────────────────────────────────────────────────
        # 1 finger up
        # ──────────────────────────────────────────────────────────────────

        if index_up and not middle_up and not ring_up and not pinky_up:
            # L — index up + thumb out to side → forms L
            if thumb_open:
                return "L", 0.88

            # D — index up, thumb tip near index tip (they form a circle/arch)
            if not thumb_open and d_ti < 0.32:
                return "D", 0.83

            # X — index hooked: tip bent downward below pip in image coords
            if index_tip[1] > index_pip[1] + 0.015:
                return "X", 0.80

            # G — pointing (index only, no thumb)
            return "G", 0.80

        # ──────────────────────────────────────────────────────────────────
        # 0 fingers up (all-curled variants)
        # ──────────────────────────────────────────────────────────────────

        if non_thumb_up == 0:
            # O — all fingertips pinched toward thumb, forming a rounded O
            # Check: all four tips within comfortable distance from thumb
            if d_ti < 0.34 and d_tm < 0.44 and d_tr < 0.54:
                return "O", 0.84

            # C — fingers NOT tightly curled (partial bend), thumb spread
            # Between open and closed: tips are at "C" arc distance from MCPs
            if (thumb_open and
                    not index_curled and not middle_curled and
                    not ring_curled and not pinky_curled and
                    d_ti > 0.34):
                return "C", 0.78

            # Fully-curled fist variants (A / S / E / T)
            if index_curled and middle_curled and ring_curled:
                # T — thumb tucked between index and middle at pip height
                if not thumb_open and pinky_curled:
                    mid_knuckle_x = (index_mcp[0] + middle_mcp[0]) / 2.0
                    knuckle_width = abs(index_mcp[0] - middle_mcp[0]) + palm_size * 0.3
                    thumb_x_centered = abs(thumb_tip[0] - mid_knuckle_x) < knuckle_width * 0.7
                    thumb_at_pip = abs(thumb_tip[1] - index_pip[1]) < palm_size * 0.45
                    if thumb_x_centered and thumb_at_pip:
                        return "T", 0.78

                if thumb_open:
                    # Thumb pointing upward or downward
                    palm_cy = points[9][1]
                    if thumb_tip[1] < palm_cy - 0.06:
                        return "THUMB_UP", 0.88
                    if thumb_tip[1] > palm_cy + 0.06:
                        return "THUMB_DOWN", 0.88
                    # Thumb out to side → A variant
                    return "A", 0.78

                # S — thumb tip crosses over (above) the curled fingers' knuckles
                #   In image coords y=0 is top. Thumb "over" = smaller y than pip.
                if thumb_tip[1] < index_pip[1] - 0.015:
                    return "S", 0.83

                # E — fingertips are very close to the palm (deep curl)
                #   Approximation: all finger tips close to wrist vs their MCPs
                if (d(index_tip, wrist)  < d(index_mcp,  wrist) * 0.85 and
                        d(middle_tip, wrist) < d(middle_mcp, wrist) * 0.85):
                    return "E", 0.78

                # A — thumb alongside the fist
                return "A", 0.84

        return None, 0.0
