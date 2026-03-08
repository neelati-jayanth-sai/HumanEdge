from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass(slots=True)
class HandDetectorConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7


class HandDetector:
    def __init__(self, config: HandDetectorConfig | None = None) -> None:
        self.config = config or HandDetectorConfig()
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def detect_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        hand_landmarks = result.multi_hand_landmarks[0]
        points = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )
        return self._normalize(points)

    @staticmethod
    def _normalize(points: np.ndarray) -> np.ndarray:
        wrist = points[0].copy()
        centered = points - wrist
        scale = np.linalg.norm(centered[:, :2], axis=1).max()
        if scale < 1e-6:
            scale = 1.0
        normalized = centered / scale
        return normalized.reshape(-1)

