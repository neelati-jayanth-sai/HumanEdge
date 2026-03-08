from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Sequence

import torch
import torch.nn as nn


VOCAB_200: list[str] = [
    "I",
    "YOU",
    "HE",
    "SHE",
    "WE",
    "THEY",
    "ME",
    "HIM",
    "HER",
    "US",
    "THEM",
    "IT",
    "WANT",
    "NEED",
    "GO",
    "HELP",
    "EAT",
    "DRINK",
    "GIVE",
    "TAKE",
    "SEE",
    "KNOW",
    "HAVE",
    "DO",
    "MAKE",
    "COME",
    "LEAVE",
    "SLEEP",
    "WORK",
    "READ",
    "WRITE",
    "OPEN",
    "CLOSE",
    "PLAY",
    "CALL",
    "WAIT",
    "STOP",
    "START",
    "THINK",
    "REMEMBER",
    "FORGET",
    "FIND",
    "LOSE",
    "BUY",
    "SELL",
    "PAY",
    "STUDY",
    "TEACH",
    "LISTEN",
    "WATCH",
    "HAPPY",
    "SAD",
    "ANGRY",
    "TIRED",
    "EXCITED",
    "SCARED",
    "CALM",
    "CONFUSED",
    "SURPRISED",
    "PROUD",
    "WORRIED",
    "SICK",
    "STRONG",
    "WEAK",
    "WHAT",
    "WHY",
    "WHERE",
    "WHEN",
    "HOW",
    "WHO",
    "WHICH",
    "CAN",
    "SHOULD",
    "WATER",
    "FOOD",
    "PHONE",
    "MONEY",
    "BOOK",
    "HOUSE",
    "HOME",
    "SCHOOL",
    "OFFICE",
    "HOSPITAL",
    "STORE",
    "ROAD",
    "CAR",
    "BUS",
    "TRAIN",
    "BIKE",
    "TABLE",
    "CHAIR",
    "BED",
    "DOOR",
    "WINDOW",
    "BAG",
    "PEN",
    "PAPER",
    "COMPUTER",
    "LAPTOP",
    "KEY",
    "GLASS",
    "BOTTLE",
    "PLATE",
    "SPOON",
    "FORK",
    "RICE",
    "BREAD",
    "MILK",
    "COFFEE",
    "TEA",
    "FRUIT",
    "VEGETABLE",
    "MEDICINE",
    "CLOTHES",
    "SHOES",
    "UMBRELLA",
    "NOW",
    "TODAY",
    "TOMORROW",
    "YESTERDAY",
    "LATER",
    "MORNING",
    "AFTERNOON",
    "EVENING",
    "NIGHT",
    "WEEK",
    "MONTH",
    "YEAR",
    "MORE",
    "LESS",
    "VERY",
    "PLEASE",
    "QUICKLY",
    "SLOWLY",
    "BIG",
    "SMALL",
    "GOOD",
    "BAD",
    "HOT",
    "COLD",
    "NEAR",
    "FAR",
    "NEW",
    "OLD",
    "YES",
    "NO",
    "OK",
    "SORRY",
    "THANK_YOU",
    "ZERO",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "ELEVEN",
    "TWELVE",
    "THIRTEEN",
    "FOURTEEN",
    "FIFTEEN",
    "SIXTEEN",
    "SEVENTEEN",
    "EIGHTEEN",
    "NINETEEN",
    "TWENTY",
    "MOTHER",
    "FATHER",
    "BROTHER",
    "SISTER",
    "FAMILY",
    "FRIEND",
    "CHILD",
    "BABY",
    "DOCTOR",
    "TEACHER",
    "POLICE",
    "NAME",
    "AGE",
    "ADDRESS",
    "CITY",
    "COUNTRY",
    "LANGUAGE",
    "SIGN",
    "UNDERSTAND",
    "REPEAT",
    "AGAIN",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "INSIDE",
    "OUTSIDE",
    "BECAUSE",
    "IF",
    "MAYBE",
]

if len(VOCAB_200) != 200:
    raise RuntimeError(f"Vocabulary size must be exactly 200, got {len(VOCAB_200)}")


class GestureNet(nn.Module):
    def __init__(self, input_dim: int = 63, output_dim: int = 200) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass(slots=True)
class GestureClassifierConfig:
    model_path: str
    confidence_threshold: float = 0.75
    device: str = "cpu"


class GestureClassifier:
    def __init__(
        self,
        config: GestureClassifierConfig,
        labels: Sequence[str] = VOCAB_200,
    ) -> None:
        if len(labels) < 200:
            raise ValueError("Minimum supported vocabulary is 200 signs.")
        if not config.model_path:
            raise ValueError("GESTURE_MODEL_PATH is required.")

        torch.set_num_threads(1)
        self.labels = list(labels)
        self.device = torch.device(config.device)
        self.threshold = config.confidence_threshold
        self.net = GestureNet(input_dim=63, output_dim=len(self.labels)).to(self.device)
        self._load_weights(config.model_path)
        self.net.eval()

    def _load_weights(self, model_path: str) -> None:
        weights_path = Path(model_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        try:
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError(
                "Checkpoint format is not compatible with this classifier. "
                "Expected a plain PyTorch state_dict for GestureNet (63 -> 200). "
                "The provided file appears to be a full model checkpoint (for example, Ultralytics/YOLO). "
                "Set GESTURE_MODEL_PATH to your trained gesture classifier .pt weights."
            ) from exc

        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]

        if not isinstance(state, dict):
            raise ValueError(
                "Invalid model checkpoint. Expected a state_dict dictionary of tensors."
            )

        if "model.0.weight" not in state:
            if all(k.startswith("module.") for k in state.keys()):
                state = {k.removeprefix("module."): v for k, v in state.items()}
            if all(k.startswith("net.") for k in state.keys()):
                state = {k.removeprefix("net."): v for k, v in state.items()}

        try:
            self.net.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint tensors do not match GestureNet architecture (63 input features, 200 output classes). "
                "Please provide the correct gesture classification weights."
            ) from exc

    @torch.inference_mode()
    def predict(self, features: torch.Tensor) -> tuple[str | None, float, str]:
        if features.shape[-1] != 63:
            raise ValueError("Expected 63 feature inputs (21 x 3 landmarks).")

        logits = self.net(features.to(self.device))
        probs = torch.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)
        score = float(conf.item())
        raw_label = self.labels[int(idx.item())]
        if score < self.threshold:
            return None, score, raw_label
        return raw_label, score, raw_label
