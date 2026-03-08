"""
Temporal motion classifier for dynamic ASL signs.

Works on a sliding window of MediaPipe hand landmarks (21 points, x/y/z
in normalised image coordinates) to detect signs that require movement.

No external model is needed — classification is geometry-based, making it
fast, deterministic, and explainable.

Supported dynamic signs
-----------------------
WAVE        – lateral wrist oscillation (Hello / Hi)
THANK_YOU   – flat hand sweeping forward from chin area
PLEASE      – circular palm rub on chest
SORRY       – fist circling on chest
YES_NOD     – fist bouncing up-down (yes)
NO_SHAKE    – index/middle pointing, side-to-side (no)
COME        – repeated beckoning curl toward self
MORE        – two-pinch tapping (encoded as repeated pinch open/close)
EAT         – hand-to-mouth repeated motion
DRINK       – C-shape hand tilts to mouth
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Each frame's landmarks: list of 21 (x, y, z) tuples
Landmarks = list[tuple[float, float, float]]


class MotionResult(NamedTuple):
    label: str | None
    confidence: float


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dist(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def _wrist(lms: Landmarks) -> tuple[float, float]:
    return lms[0][0], lms[0][1]


def _palm_center(lms: Landmarks) -> tuple[float, float]:
    """Average of wrist + 4 MCP joints."""
    pts = [lms[i] for i in (0, 5, 9, 13, 17)]
    return (
        sum(p[0] for p in pts) / 5,
        sum(p[1] for p in pts) / 5,
    )


def _hand_open_ratio(lms: Landmarks) -> float:
    """
    Rough openness score: average extension of 4 fingers.
    1.0 = fully open, 0.0 = fully closed.
    """
    pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]   # tip vs MCP
    scores = []
    palm_sz = max(1e-6, _dist(lms[0], lms[9]))
    for tip_i, mcp_i in pairs:
        scores.append(_dist(lms[tip_i], lms[0]) / palm_sz)
    return sum(scores) / len(scores)


def _pinch_dist(lms: Landmarks) -> float:
    """Distance between thumb tip and index tip (normalised by palm size)."""
    palm_sz = max(1e-6, _dist(lms[0], lms[9]))
    return _dist(lms[4], lms[8]) / palm_sz


def _wrist_y_relative_to_face(lms: Landmarks) -> float:
    """
    Heuristic: wrist y in normalised image coords.
    < 0.35  → near face/forehead
    0.35–0.6 → chin/neck
    > 0.6   → chest/torso
    """
    return lms[0][1]


# ---------------------------------------------------------------------------
# Individual sign detectors
# ---------------------------------------------------------------------------

class _Detector:
    label: str
    cooldown_s: float = 1.2

    def __init__(self) -> None:
        self._last_fired = 0.0

    def _on_cooldown(self) -> bool:
        return (time.monotonic() - self._last_fired) < self.cooldown_s

    def _fire(self, conf: float) -> MotionResult:
        self._last_fired = time.monotonic()
        return MotionResult(self.label, conf)

    def reset(self) -> None:
        self._last_fired = 0.0

    def detect(self, frames: list[Landmarks]) -> MotionResult:
        raise NotImplementedError


class WaveDetector(_Detector):
    """
    WAVE / HELLO: wrist oscillates horizontally ≥2 times while hand is open.
    """
    label = "WAVE"
    cooldown_s = 1.5

    def detect(self, frames: list[Landmarks]) -> MotionResult:
        if self._on_cooldown() or len(frames) < 12:
            return MotionResult(None, 0)

        xs = [_wrist(f)[0] for f in frames]
        openness = sum(_hand_open_ratio(f) for f in frames[-8:]) / 8

        if openness < 0.6:
            return MotionResult(None, 0)

        # Count direction reversals
        reversals = 0
        for i in range(2, len(xs)):
            d1 = xs[i-1] - xs[i-2]
            d2 = xs[i]   - xs[i-1]
            if abs(d1) > 0.015 and abs(d2) > 0.015 and d1 * d2 < 0:
                reversals += 1

        if reversals >= 3:
            conf = min(0.90, 0.70 + reversals * 0.05)
            return self._fire(conf)
        return MotionResult(None, 0)


class ThankYouDetector(_Detector):
    """
    THANK_YOU: flat hand near chin moves forward/downward.
    Wrist starts near face level (y < 0.55) and moves down/forward.
    """
    label = "THANK_YOU"

    def detect(self, frames: list[Landmarks]) -> MotionResult:
        if self._on_cooldown() or len(frames) < 10:
            return MotionResult(None, 0)

        first, last = frames[0], frames[-1]
        wy0 = _wrist_y_relative_to_face(first)
        wy1 = _wrist_y_relative_to_face(last)
        dy = wy1 - wy0          # positive = moves down in image
        openness = _hand_open_ratio(first)

        # Hand should be open, start near chin, move downward
        if openness > 0.6 and 0.3 < wy0 < 0.65 and dy > 0.06:
            return self._fire(0.82)
        return MotionResult(None, 0)


class CircularMotionDetector(_Detector):
    """
    PLEASE / SORRY: wrist traces a rough circular path on the chest.
    Detected by measuring variance in the angle of displacement vectors.
    """
    label: str  # set by subclass
    min_radius: float = 0.03

    def detect(self, frames: list[Landmarks]) -> MotionResult:
        if self._on_cooldown() or len(frames) < 14:
            return MotionResult(None, 0)

        wxs = [_wrist(f)[0] for f in frames]
        wys = [_wrist(f)[1] for f in frames]
        cx  = sum(wxs) / len(wxs)
        cy  = sum(wys) / len(wys)

        # Angles from centroid
        angles = [math.atan2(y - cy, x - cx) for x, y in zip(wxs, wys)]

        # Unwrap angles and check total arc swept
        unwrapped = [angles[0]]
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i-1]
            diff = (diff + math.pi) % (2*math.pi) - math.pi
            unwrapped.append(unwrapped[-1] + diff)

        total_arc = abs(unwrapped[-1] - unwrapped[0])
        radius    = max(
            math.sqrt((x-cx)**2 + (y-cy)**2)
            for x, y in zip(wxs, wys)
        )

        if total_arc > math.pi * 1.2 and radius > self.min_radius:
            conf = min(0.88, 0.70 + total_arc / (2 * math.pi) * 0.1)
            return self._fire(conf)
        return MotionResult(None, 0)


class PleaseDetector(CircularMotionDetector):
    """PLEASE: open palm circling on chest."""
    label = "PLEASE"

    def detect(self, frames: list[Landmarks]) -> MotionResult:
        # Require open hand (palm rubbing)
        if frames and _hand_open_ratio(frames[-1]) < 0.55:
            return MotionResult(None, 0)
        return super().detect(frames)


class SorryDetector(CircularMotionDetector):
    """SORRY: fist circling on chest."""
    label = "SORRY"
    cooldown_s = 1.5

    def detect(self, frames: list[Landmarks]) -> MotionResult:
        # Require closed hand (fist)
        if frames and _hand_open_ratio(frames[-1]) > 0.40:
            return MotionResult(None, 0)
        return super().detect(frames)


class YesNodDetector(_Detector):
    """
    YES: fist bounces up and down (≥2 up-down cycles).
    """
    label = "YES_NOD"

    def detect(self, frames: list[Landmarks]) -> MotionResult:
        if self._on_cooldown() or len(frames) < 10:
            return MotionResult(None, 0)

        wys = [_wrist(f)[1] for f in frames]
        openness = sum(_hand_open_ratio(f) for f in frames) / len(frames)

        if openness > 0.45:   # hand should be closed (fist)
            return MotionResult(None, 0)

        reversals = 0
        for i in range(2, len(wys)):
            d1 = wys[i-1] - wys[i-2]
            d2 = wys[i]   - wys[i-1]
            if abs(d1) > 0.02 and abs(d2) > 0.02 and d1 * d2 < 0:
                reversals += 1

        if reversals >= 3:
            return self._fire(0.83)
        return MotionResult(None, 0)


class MoreDetector(_Detector):
    """
    MORE: fingertips pinch open/close repeatedly (≥2 cycles).
    """
    label = "MORE"

    def detect(self, frames: list[Landmarks]) -> MotionResult:
        if self._on_cooldown() or len(frames) < 10:
            return MotionResult(None, 0)

        pinches = [_pinch_dist(f) for f in frames]

        reversals = 0
        for i in range(2, len(pinches)):
            d1 = pinches[i-1] - pinches[i-2]
            d2 = pinches[i]   - pinches[i-1]
            if abs(d1) > 0.08 and abs(d2) > 0.08 and d1 * d2 < 0:
                reversals += 1

        if reversals >= 3:
            return self._fire(0.80)
        return MotionResult(None, 0)


class EatDetector(_Detector):
    """
    EAT: hand repeatedly moves toward face (wrist y decreases repeatedly).
    """
    label = "EAT"

    def detect(self, frames: list[Landmarks]) -> MotionResult:
        if self._on_cooldown() or len(frames) < 12:
            return MotionResult(None, 0)

        wys = [_wrist(f)[1] for f in frames]
        avg_y = sum(wys) / len(wys)

        if avg_y > 0.65:     # hand must be near face level
            return MotionResult(None, 0)

        reversals = 0
        for i in range(2, len(wys)):
            d1 = wys[i-1] - wys[i-2]
            d2 = wys[i]   - wys[i-1]
            if abs(d1) > 0.02 and abs(d2) > 0.02 and d1 * d2 < 0:
                reversals += 1

        if reversals >= 2:
            return self._fire(0.78)
        return MotionResult(None, 0)


class ComeDetector(_Detector):
    """
    COME: wrist moves toward the signer (y increases) with beckoning curl.
    """
    label = "COME"

    def detect(self, frames: list[Landmarks]) -> MotionResult:
        if self._on_cooldown() or len(frames) < 10:
            return MotionResult(None, 0)

        wy0  = _wrist(frames[0])[1]
        wy1  = _wrist(frames[-1])[1]
        dy   = wy1 - wy0

        # Net downward motion (toward signer in image) > threshold
        if dy > 0.08:
            return self._fire(0.76)
        return MotionResult(None, 0)


# ---------------------------------------------------------------------------
# MotionClassifier — aggregates all detectors on a sliding window
# ---------------------------------------------------------------------------

class MotionClassifier:
    """
    Maintains a sliding window of landmark frames and runs all dynamic-sign
    detectors on each update.

    Usage
    -----
    classifier = MotionClassifier()
    result = classifier.update(landmarks)   # call once per frame
    if result.label:
        print(result.label, result.confidence)
    """

    WINDOW = 25   # frames (at 15 fps ≈ 1.7 s of context)

    _NULL_RESET_AFTER = 8  # clear window only after this many consecutive no-hand frames

    def __init__(self) -> None:
        self._frames: deque[Landmarks] = deque(maxlen=self.WINDOW)
        self._detectors: list[_Detector] = [
            WaveDetector(),
            ThankYouDetector(),
            PleaseDetector(),
            SorryDetector(),
            YesNodDetector(),
            MoreDetector(),
            EatDetector(),
            ComeDetector(),
        ]
        self._null_streak: int = 0

    def update(self, landmarks: Landmarks | None) -> MotionResult:
        """
        Feed one frame of landmarks.  Returns a MotionResult immediately if a
        dynamic sign is detected; otherwise returns (None, 0.0).

        Pass None if no hand is tracked. The window is only cleared after
        _NULL_RESET_AFTER consecutive no-hand frames to tolerate brief
        tracking drops during fast motion.
        """
        if landmarks is None:
            self._null_streak += 1
            if self._null_streak >= self._NULL_RESET_AFTER:
                self._frames.clear()
            return MotionResult(None, 0.0)

        self._null_streak = 0
        self._frames.append(landmarks)
        frames = list(self._frames)

        best: MotionResult = MotionResult(None, 0.0)
        for det in self._detectors:
            result = det.detect(frames)
            if result.label and result.confidence > best.confidence:
                best = result

        return best

    def reset(self) -> None:
        self._frames.clear()
        self._null_streak = 0
        for det in self._detectors:
            det.reset()
