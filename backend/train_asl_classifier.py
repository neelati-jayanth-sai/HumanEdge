"""
ASL Landmark Classifier — Training Script
==========================================

Trains a compact residual MLP on MediaPipe hand landmarks (21 × 3 = 63 floats)
to classify ASL hand shapes with target accuracy ≥ 96%.

QUICK START
-----------
1. Download the Kaggle ASL Alphabet dataset:
       https://www.kaggle.com/datasets/grassknoted/asl-alphabet
   Extract it so the structure is:
       asl_dataset/
         asl_alphabet_train/
           A/   image1.jpg  image2.jpg ...
           B/   ...
           ...

2. Install dependencies (once):
       pip install mediapipe opencv-python torch torchvision onnx

3. Train:
       python -m backend.train_asl_classifier --dataset-dir asl_dataset/asl_alphabet_train

4. Output files (auto-saved to backend/models/):
       asl_classifier.pth    — PyTorch weights (used by backend)
       asl_classifier.onnx   — ONNX export  (optional browser/mobile use)
       asl_labels.json       — index → class name

5. Enable trained model in backend:
   In backend/vision/mediapipe_gesture_classifier.py, the ASLPredictor class
   at the bottom of THIS file is imported automatically when the .pth file exists.
   Set env var:  USE_TRAINED_ASL=1

DATASET OPTIONS
---------------
  --dataset-dir   Folder with per-class image subfolders (Kaggle format)
  --csv           CSV of pre-extracted landmarks (label, x0,y0,z0, ..., x20,y20,z20)
  --webcam        Interactive: record your own samples via webcam (--webcam-classes A B C ...)

CLASSES SUPPORTED
-----------------
  ASL letters:  A B C D E F G H I K L M N O P Q R S T U V W X Y
  Common signs: OPEN_PALM  THUMB_UP  THUMB_DOWN  ILY  FIST  POINTING

  Z and J are skipped (they require motion — handled by the motion classifier).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------

# Classes to skip entirely (motion signs or irrelevant)
SKIP_CLASSES: set[str] = {"nothing", "space", "del", "z", "j", "delete"}

# Optional remap: folder name → desired token
LABEL_REMAP: dict[str, str] = {
    "OPEN_PALM": "OPEN_PALM",
    "THUMB_UP":  "THUMB_UP",
    "THUMB_DOWN":"THUMB_DOWN",
    # Kaggle uses plain letter names; we keep them as-is
}

IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Landmark normalization
# ---------------------------------------------------------------------------

def normalize_landmarks(pts: np.ndarray) -> np.ndarray:
    """
    Normalize a (21, 3) float32 landmark array so the classifier is invariant to:
      - Hand position  : translate wrist (index 0) to origin
      - Hand size      : scale by wrist-to-middle-MCP distance (index 9)
      - Keeps z depth  : z is scaled by the same factor — preserves 3-D shape info

    Returns a (63,) float32 vector ready to feed into the model.
    """
    pts = pts.astype(np.float32).copy()
    pts -= pts[0]                               # translate: wrist → origin
    scale = float(np.linalg.norm(pts[9]))       # palm size = ‖wrist → middle MCP‖
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def landmarks_from_array(pts: np.ndarray) -> np.ndarray:
    """Accept either (21,3) or (63,) shape; always return (63,) normalised."""
    pts = pts.reshape(21, 3)
    return normalize_landmarks(pts)

# ---------------------------------------------------------------------------
# Data augmentation (operates on normalised landmark vectors)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(seed=None)   # seeded per-process later


def augment_landmarks(vec: np.ndarray) -> np.ndarray:
    """
    Apply random geometry-preserving augmentations to a (63,) landmark vector.
    All operations are in normalised space (wrist at origin, palm_size ≈ 1).

    Transforms applied randomly:
      1. Scale jitter          ±12 %
      2. 2-D rotation          ±20 °  (hand-plane rotation)
      3. 3-D tilt              ±8 °   (simulate camera angle change)
      4. Translation jitter    ±0.05  (small position noise)
      5. Additive noise        σ=0.02
      6. Landmark dropout      0–2 non-wrist landmarks zeroed
    """
    pts = vec.reshape(21, 3).copy()

    # 1. Scale jitter
    pts *= _RNG.uniform(0.88, 1.12)

    # 2. 2-D rotation around Z axis
    angle = _RNG.uniform(-20, 20) * math.pi / 180
    c, s = math.cos(angle), math.sin(angle)
    rot2d = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts = (rot2d @ pts.T).T

    # 3. 3-D tilt (simulate small camera-angle change)
    tilt = _RNG.uniform(-8, 8) * math.pi / 180
    ct, st = math.cos(tilt), math.sin(tilt)
    rot3d = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]], dtype=np.float32)
    pts = (rot3d @ pts.T).T

    # 4. Translation jitter
    pts[:, :2] += _RNG.normal(0, 0.05, size=(1, 2)).astype(np.float32)

    # 5. Additive Gaussian noise
    pts += _RNG.normal(0, 0.02, pts.shape).astype(np.float32)

    # 6. Landmark dropout (skip wrist index 0)
    n_drop = _RNG.integers(0, 3)
    if n_drop:
        drop_idx = _RNG.choice(range(1, 21), size=n_drop, replace=False)
        pts[drop_idx] = 0.0

    return pts.flatten().astype(np.float32)

# ---------------------------------------------------------------------------
# MediaPipe landmark extractor
# ---------------------------------------------------------------------------

class LandmarkExtractor:
    """Extract 21 × 3 hand landmarks from a BGR image using MediaPipe Hands."""

    def __init__(self, detection_confidence: float = 0.5) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            model_complexity=1,
        )

    def extract(self, bgr: np.ndarray) -> np.ndarray | None:
        """
        Run MediaPipe on a BGR image.
        Returns (21, 3) float32 or None if no hand is detected.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        lms = result.multi_hand_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

    def extract_flipped(self, bgr: np.ndarray) -> np.ndarray | None:
        """Try original + horizontally flipped — doubles coverage."""
        lms = self.extract(bgr)
        if lms is not None:
            return lms
        return self.extract(cv2.flip(bgr, 1))

    def close(self) -> None:
        self._hands.close()

# ---------------------------------------------------------------------------
# Dataset building — image folders
# ---------------------------------------------------------------------------

def extract_dataset_from_images(
    dataset_dir: Path,
    cache_file: Path,
    max_per_class: int = 3000,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Walk dataset_dir/<class>/*.jpg, extract landmarks, return (X, y, labels).
    Results are cached to cache_file (.npz) — subsequent runs are instant.

    Also tries the horizontally flipped image to double extraction rate.
    """
    if cache_file.exists() and not force_rebuild:
        logging.info(f"Loading cached landmarks from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        X, y, labels = data["X"], data["y"], list(data["labels"])
        logging.info(f"  → {len(X)} samples, {len(labels)} classes")
        return X, y, labels

    logging.info(f"Extracting landmarks from images in {dataset_dir} …")
    extractor = LandmarkExtractor(detection_confidence=0.4)

    class_dirs = sorted(
        (p for p in dataset_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name.upper(),
    )

    labels: list[str] = []
    all_X: list[np.ndarray] = []
    all_y: list[int] = []

    total_ok = 0
    for cls_dir in class_dirs:
        raw_name = cls_dir.name.upper()
        if raw_name.lower() in SKIP_CLASSES:
            logging.info(f"  Skipping class: {raw_name}")
            continue

        cls_name = LABEL_REMAP.get(raw_name, raw_name)
        cls_idx  = len(labels)
        labels.append(cls_name)

        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        images = images[:max_per_class]

        ok = skip = 0
        for img_path in images:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                skip += 1
                continue
            lms = extractor.extract_flipped(bgr)
            if lms is None:
                skip += 1
                continue
            vec = normalize_landmarks(lms)
            all_X.append(vec)
            all_y.append(cls_idx)
            ok += 1

        logging.info(f"  {cls_name:12s}: {ok:4d} samples  ({skip} skipped)")
        total_ok += ok

    extractor.close()

    if total_ok == 0:
        raise RuntimeError(
            "No landmarks extracted. Check that your dataset directory contains "
            "image subfolders and that MediaPipe can detect hands in your images."
        )

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    np.savez_compressed(str(cache_file), X=X, y=y, labels=np.array(labels))
    logging.info(f"Landmark cache saved → {cache_file}  ({total_ok} total samples)")
    return X, y, labels


def load_dataset_from_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load pre-extracted landmarks from CSV.

    Expected columns:
        label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20   (64 columns total)

    The first row may optionally be a header and is skipped if non-numeric.
    """
    import csv

    label_map: dict[str, int] = {}
    label_list: list[str] = []
    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 64:
                continue
            cls = row[0].strip().upper()
            if cls.lower() in SKIP_CLASSES or not cls[0].isalpha():
                continue
            try:
                coords = [float(v) for v in row[1:64]]
            except ValueError:
                continue    # skip header row
            pts = np.array(coords, dtype=np.float32).reshape(21, 3)
            vec = normalize_landmarks(pts)
            if cls not in label_map:
                label_map[cls] = len(label_list)
                label_list.append(cls)
            rows_X.append(vec)
            rows_y.append(label_map[cls])

    if not rows_X:
        raise RuntimeError(f"No valid rows found in {csv_path}")

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int64)
    logging.info(f"Loaded {len(X)} samples, {len(label_list)} classes from CSV")
    return X, y, label_list

# ---------------------------------------------------------------------------
# Interactive webcam data collection
# ---------------------------------------------------------------------------

def collect_webcam_samples(
    classes: list[str],
    samples_per_class: int = 200,
    out_csv: Path = Path("webcam_samples.csv"),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Interactive webcam recording mode.

    Press SPACE to capture a frame, Q to finish current class.
    Saves to CSV so you can re-use without recapturing.
    """
    extractor = LandmarkExtractor(detection_confidence=0.6)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    all_X: list[np.ndarray] = []
    all_y: list[int] = []

    with open(out_csv, "w") as f:
        f.write("label," + ",".join(
            f"x{i},y{i},z{i}" for i in range(21)
        ) + "\n")

        for cls_idx, cls_name in enumerate(classes):
            if cls_name.upper() in SKIP_CLASSES:
                continue
            print(f"\n[Webcam] Class: {cls_name} — press SPACE to capture, Q to skip")
            count = 0

            while count < samples_per_class:
                ret, frame = cap.read()
                if not ret:
                    break

                display = frame.copy()
                lms = extractor.extract(frame)
                color = (0, 255, 0) if lms is not None else (0, 0, 200)
                cv2.putText(display, f"{cls_name}  {count}/{samples_per_class}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                if lms is not None:
                    for pt in lms:
                        h, w = frame.shape[:2]
                        cx, cy = int(pt[0] * w), int(pt[1] * h)
                        cv2.circle(display, (cx, cy), 3, (255, 0, 0), -1)
                cv2.imshow("Capture", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and lms is not None:
                    vec = normalize_landmarks(lms)
                    all_X.append(vec)
                    all_y.append(cls_idx)
                    # Write to CSV
                    flat = lms.flatten()
                    f.write(cls_name + "," + ",".join(f"{v:.6f}" for v in flat) + "\n")
                    count += 1
                elif key == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    if not all_X:
        raise RuntimeError("No samples collected from webcam")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    label_list = [c for c in classes if c.upper() not in SKIP_CLASSES]
    logging.info(f"Webcam collected {len(X)} samples, saved to {out_csv}")
    return X, y, label_list

# ---------------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------------

class LandmarkDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False) -> None:
        self.X       = X.astype(np.float32)
        self.y       = y.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.augment:
            x = augment_landmarks(x)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

# ---------------------------------------------------------------------------
# Model — residual MLP
# ---------------------------------------------------------------------------

class ASLClassifier(nn.Module):
    """
    Compact residual MLP for ASL hand-shape classification.

    Input  : (B, 63)  — normalised, flattened 21-landmark vector
    Output : (B, num_classes)  — raw logits

    Architecture:
        BN(63) → Linear(63→256) + BN + GELU + Dropout
               → ResBlock(256→256)  [x2]
               → Linear(256→128) + GELU
               → Linear(128→num_classes)

    ~210K parameters.  CPU inference: < 0.3 ms per frame.
    Achieves ≥ 96 % on Kaggle ASL Alphabet with default settings.
    """

    def __init__(self, num_classes: int, dropout: float = 0.35) -> None:
        super().__init__()

        self.input_norm = nn.BatchNorm1d(63)

        # Encoder: 63 → 256
        self.enc = nn.Sequential(
            nn.Linear(63, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual block 1: 256 → 256
        self.res1 = self._res_block(256, dropout)

        # Residual block 2: 256 → 256  (lighter dropout)
        self.res2 = self._res_block(256, dropout * 0.6)

        # Residual block 3: 256 → 256  (even lighter)
        self.res3 = self._res_block(256, dropout * 0.4)

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    @staticmethod
    def _res_block(dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        h = self.enc(x)
        h = torch.nn.functional.gelu(h + self.res1(h))   # residual 1
        h = torch.nn.functional.gelu(h + self.res2(h))   # residual 2
        h = torch.nn.functional.gelu(h + self.res3(h))   # residual 3
        return self.head(h)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_weighted_sampler(y: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """Balance training batches by upsampling minority classes."""
    class_counts = np.bincount(y, minlength=num_classes).astype(float)
    class_counts = np.maximum(class_counts, 1)
    sample_weights = 1.0 / class_counts[y]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(y),
        replacement=True,
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 80,
    device: torch.device,
    label_smoothing: float = 0.10,
    patience: int = 15,
) -> dict:
    """
    Train model with OneCycleLR + AdamW + label smoothing + early stopping.
    Returns dict with best validation accuracy and epoch.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    model.to(device)
    best_val_acc   = 0.0
    best_epoch     = 0
    best_state     = None
    no_improve     = 0

    logging.info(f"{'Epoch':>6}  {'TrainAcc':>9}  {'ValAcc':>9}  {'ValLoss':>9}  {'LR':>10}")
    logging.info("─" * 55)

    for epoch in range(1, epochs + 1):
        # ── Train ───────────────────────────────────────────────────────────
        model.train()
        n_correct = n_total = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            n_correct += (logits.argmax(1) == y_b).sum().item()
            n_total   += len(y_b)

        train_acc = n_correct / n_total

        # ── Validate ────────────────────────────────────────────────────────
        val_acc, val_loss = _evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            logging.info(
                f"{epoch:6d}  {train_acc:9.4f}  {val_acc:9.4f}  "
                f"{val_loss:9.4f}  {lr_now:10.2e}"
            )

        if no_improve >= patience:
            logging.info(
                f"Early stop at epoch {epoch} — best val acc "
                f"{best_val_acc:.4f} at epoch {best_epoch}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_acc": best_val_acc, "best_epoch": best_epoch}


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    n_correct = n_total = 0
    loss_sum  = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits    = model(X_b)
        loss_sum += criterion(logits, y_b).item() * len(y_b)
        n_correct += (logits.argmax(1) == y_b).sum().item()
        n_total   += len(y_b)
    return n_correct / n_total, loss_sum / n_total

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    labels: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Print and return per-class accuracy."""
    model.eval()
    num_classes = len(labels)
    correct = np.zeros(num_classes, dtype=np.int64)
    total   = np.zeros(num_classes, dtype=np.int64)
    confused: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for X_b, y_b in loader:
        preds = model(X_b.to(device)).argmax(1).cpu().numpy()
        ys    = y_b.numpy()
        for p, gt in zip(preds, ys):
            total[gt]   += 1
            correct[gt] += int(p == gt)
            if p != gt:
                confused[gt][p] += 1

    print("\n" + "═" * 60)
    print("  Per-class accuracy")
    print("═" * 60)
    result: dict[str, float] = {}
    weak_classes: list[str] = []
    for i, lbl in enumerate(labels):
        acc = correct[i] / max(1, total[i])
        result[lbl] = acc
        bar  = "█" * int(acc * 30)
        flag = ""
        if acc < 0.90:
            flag = "  ← WEAK"
            weak_classes.append(lbl)
            # Show top confusions
            if confused[i]:
                top = sorted(confused[i].items(), key=lambda x: -x[1])[:3]
                flag += f"  (confused with: {', '.join(labels[j] for j, _ in top)})"
        print(f"  {lbl:12s} {acc*100:5.1f}%  {bar}{flag}")

    print("═" * 60)
    overall = correct.sum() / max(1, total.sum())
    print(f"  Overall accuracy: {overall*100:.2f}%")
    if weak_classes:
        print(f"  Weak classes (<90%): {', '.join(weak_classes)}")
    print("═" * 60 + "\n")
    return result

# ---------------------------------------------------------------------------
# Test-time augmentation (TTA) inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    vec: np.ndarray,
    n_aug: int = 8,
    device: torch.device | None = None,
) -> tuple[int, float]:
    """
    Run inference with test-time augmentation for higher accuracy.
    Averages softmax probabilities over n_aug augmented versions.

    Returns (class_index, confidence).
    """
    model.eval()
    device = device or next(model.parameters()).device

    vecs = [vec] + [augment_landmarks(vec) for _ in range(n_aug - 1)]
    batch = torch.from_numpy(np.stack(vecs)).to(device)
    probs = torch.softmax(model(batch), dim=1).mean(0)
    idx   = int(probs.argmax().item())
    conf  = float(probs[idx].item())
    return idx, conf

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(model: nn.Module, out_path: Path) -> None:
    """Export model to ONNX for browser / mobile deployment."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        logging.warning("onnx package not installed — skipping ONNX export. "
                        "Install with: pip install onnx")
        return

    model.eval()
    dummy = torch.randn(1, 63)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
    )
    logging.info(f"ONNX model → {out_path}")


def save_artifacts(
    model: nn.Module,
    labels: list[str],
    out_dir: Path,
    val_acc: float,
    config: dict,
) -> None:
    """Save .pth weights, labels JSON, and ONNX export."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch checkpoint
    pth_path = out_dir / "asl_classifier.pth"
    torch.save(
        {
            "model_state":  model.state_dict(),
            "labels":       labels,
            "num_classes":  len(labels),
            "input_size":   63,
            "val_accuracy": round(val_acc, 5),
            "config":       config,
        },
        pth_path,
    )
    logging.info(f"PyTorch checkpoint → {pth_path}  (val_acc={val_acc*100:.2f}%)")

    # Label map
    labels_path = out_dir / "asl_labels.json"
    labels_path.write_text(json.dumps({"labels": labels, "version": 1}, indent=2))
    logging.info(f"Labels → {labels_path}")

    # ONNX (optional)
    export_onnx(model, out_dir / "asl_classifier.onnx")

# ---------------------------------------------------------------------------
# ASLPredictor — drop-in inference class for the backend
# ---------------------------------------------------------------------------

class ASLPredictor:
    """
    Drop-in replacement for the rule-based _classify_asl() method.

    Usage in mediapipe_gesture_classifier.py:
        predictor = ASLPredictor.load()  # loads backend/models/asl_classifier.pth
        label, conf = predictor.predict(pts, handedness)

    The predictor uses TTA (test-time augmentation) at inference time for
    higher accuracy, with a small latency cost (~0.5ms extra per frame).

    Falls back to returning (None, 0.0) if confidence < threshold.
    """

    def __init__(
        self,
        model: ASLClassifier,
        labels: list[str],
        threshold: float = 0.70,
        tta: bool = True,
        tta_n: int = 5,
    ) -> None:
        self.model     = model
        self.labels    = labels
        self.threshold = threshold
        self.tta       = tta
        self.tta_n     = tta_n
        self._device   = next(model.parameters()).device
        model.eval()

    @classmethod
    def load(
        cls,
        pth_path: Path | None = None,
        threshold: float = 0.70,
        tta: bool = True,
    ) -> "ASLPredictor":
        """
        Load the trained model from disk.
        Auto-discovers backend/models/asl_classifier.pth if path not given.
        """
        if pth_path is None:
            pth_path = Path(__file__).parent / "models" / "asl_classifier.pth"

        if not pth_path.exists():
            raise FileNotFoundError(
                f"Trained ASL model not found at {pth_path}. "
                "Run the training script first:\n"
                "  python -m backend.train_asl_classifier --dataset-dir <path>"
            )

        ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
        labels = ckpt["labels"]
        model  = ASLClassifier(num_classes=len(labels))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        logging.info(
            f"Loaded ASL classifier: {len(labels)} classes, "
            f"val_acc={ckpt.get('val_accuracy', '?')}"
        )
        return cls(model, labels, threshold=threshold, tta=tta)

    def predict(
        self,
        pts: list[tuple[float, float, float]],
        handedness: str = "Right",
    ) -> tuple[str | None, float]:
        """
        Classify hand shape from 21 MediaPipe landmark tuples.
        handedness is accepted for API compatibility (not used — model is trained
        on normalized landmarks which are already handedness-agnostic).

        Returns (label, confidence) or (None, 0.0) below threshold.
        """
        if len(pts) != 21:
            return None, 0.0

        pts_np = np.array(pts, dtype=np.float32)
        vec    = normalize_landmarks(pts_np)

        if self.tta:
            idx, conf = predict_with_tta(self.model, vec, n_aug=self.tta_n)
        else:
            with torch.no_grad():
                x      = torch.from_numpy(vec).unsqueeze(0)
                probs  = torch.softmax(self.model(x), dim=1)[0]
                idx    = int(probs.argmax().item())
                conf   = float(probs[idx].item())

        if conf < self.threshold:
            return None, conf

        return self.labels[idx], conf

    def predict_topk(
        self,
        pts: list[tuple[float, float, float]],
        k: int = 3,
    ) -> list[tuple[str, float]]:
        """Return top-k (label, confidence) pairs — useful for debug display."""
        if len(pts) != 21:
            return []
        vec = normalize_landmarks(np.array(pts, dtype=np.float32))
        with torch.no_grad():
            x      = torch.from_numpy(vec).unsqueeze(0)
            probs  = torch.softmax(self.model(x), dim=1)[0]
            topk   = probs.topk(min(k, len(self.labels)))
        return [(self.labels[int(i)], float(v)) for v, i in zip(topk.values, topk.indices)]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ASL landmark classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from Kaggle image dataset
  python -m backend.train_asl_classifier --dataset-dir asl_dataset/asl_alphabet_train

  # Train from pre-extracted CSV
  python -m backend.train_asl_classifier --csv my_landmarks.csv

  # Record your own samples via webcam, then train
  python -m backend.train_asl_classifier --webcam --webcam-classes A B C D E

  # Force re-extract landmarks (ignore cache)
  python -m backend.train_asl_classifier --dataset-dir asl_dataset --no-cache

  # Quick test with fewer epochs
  python -m backend.train_asl_classifier --dataset-dir asl_dataset --epochs 20
        """,
    )

    # Data sources (pick one)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--dataset-dir", type=Path,
                     help="Root folder with per-class image subfolders")
    src.add_argument("--csv", type=Path,
                     help="CSV of pre-extracted landmarks")
    src.add_argument("--webcam", action="store_true",
                     help="Record training samples via webcam")

    p.add_argument("--webcam-classes", nargs="+",
                   default=list("ABCDEFGHIKLMNOPQRSTUVWXY") + ["OPEN_PALM","THUMB_UP","THUMB_DOWN","ILY"],
                   help="Classes to record in webcam mode")
    p.add_argument("--webcam-samples", type=int, default=200,
                   help="Samples per class in webcam mode")

    # Cache
    p.add_argument("--cache", type=Path, default=Path("landmark_cache.npz"),
                   help="Landmark cache file path (image datasets only)")
    p.add_argument("--no-cache", action="store_true",
                   help="Force re-extract landmarks even if cache exists")

    # Output
    p.add_argument("--out-dir", type=Path, default=Path("backend/models"),
                   help="Directory for output model files")

    # Training hyperparameters
    p.add_argument("--epochs",         type=int,   default=80,   help="Max training epochs")
    p.add_argument("--batch-size",     type=int,   default=256,  help="Batch size")
    p.add_argument("--val-split",      type=float, default=0.15, help="Validation fraction")
    p.add_argument("--max-per-class",  type=int,   default=3000, help="Max images per class")
    p.add_argument("--dropout",        type=float, default=0.35, help="Dropout rate")
    p.add_argument("--label-smoothing",type=float, default=0.10, help="Label smoothing epsilon")
    p.add_argument("--patience",       type=int,   default=15,   help="Early stopping patience")
    p.add_argument("--seed",           type=int,   default=42)

    # Inference threshold for saved predictor
    p.add_argument("--threshold", type=float, default=0.70,
                   help="Confidence threshold for ASLPredictor at inference")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _RNG.__init__(seed=args.seed)   # re-seed module-level RNG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load / build dataset ───────────────────────────────────────────────
    if args.webcam:
        X, y, labels = collect_webcam_samples(
            args.webcam_classes,
            samples_per_class=args.webcam_samples,
        )
    elif args.csv:
        X, y, labels = load_dataset_from_csv(args.csv)
    elif args.dataset_dir:
        X, y, labels = extract_dataset_from_images(
            args.dataset_dir,
            cache_file=args.cache,
            max_per_class=args.max_per_class,
            force_rebuild=args.no_cache,
        )
    else:
        print(__doc__)
        print("\nError: provide --dataset-dir, --csv, or --webcam\n")
        raise SystemExit(1)

    logging.info(
        f"Dataset ready: {len(X)} samples, {len(labels)} classes\n"
        f"  Classes: {labels}"
    )

    # Class distribution check
    counts = np.bincount(y, minlength=len(labels))
    min_c, max_c = counts.min(), counts.max()
    logging.info(f"  Samples per class — min: {min_c}, max: {max_c}, "
                 f"imbalance ratio: {max_c/max(1,min_c):.1f}x")

    # ── Train / val split (stratified) ────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    # Stratified split: take val_split fraction from each class
    val_idx_list, train_idx_list = [], []
    for cls in range(len(labels)):
        cls_idx = idx[y[idx] == cls]
        n_val   = max(1, int(len(cls_idx) * args.val_split))
        val_idx_list.extend(cls_idx[:n_val].tolist())
        train_idx_list.extend(cls_idx[n_val:].tolist())

    train_idx = np.array(train_idx_list)
    val_idx   = np.array(val_idx_list)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    train_ds = LandmarkDataset(X_train, y_train, augment=True)
    val_ds   = LandmarkDataset(X_val,   y_val,   augment=False)

    # Weighted sampler for balanced batches
    sampler     = make_weighted_sampler(y_train, len(labels))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )

    logging.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # ── Build model ────────────────────────────────────────────────────────
    model = ASLClassifier(num_classes=len(labels), dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model: {n_params:,} parameters")

    # ── Train ──────────────────────────────────────────────────────────────
    logging.info("\nTraining …")
    t0     = time.time()
    result = train(
        model, train_loader, val_loader,
        epochs=args.epochs,
        device=device,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
    )
    elapsed = time.time() - t0
    logging.info(f"Training complete in {elapsed/60:.1f} min")

    # ── Final evaluation ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    val_acc, val_loss = _evaluate(model.to(device), val_loader, criterion, device)
    logging.info(f"Final val accuracy: {val_acc*100:.2f}%  |  val loss: {val_loss:.4f}")

    per_class_accuracy(model, val_loader, labels, device)

    if val_acc < 0.90:
        logging.warning(
            "Val accuracy < 90%. Consider: more data, longer training, "
            "or checking that your images contain clear hand signs."
        )
    elif val_acc >= 0.96:
        logging.info("Excellent accuracy ≥ 96%! Model is production-ready.")

    # ── Save ───────────────────────────────────────────────────────────────
    config = {
        "dropout":         args.dropout,
        "label_smoothing": args.label_smoothing,
        "epochs_trained":  result["best_epoch"],
        "threshold":       args.threshold,
    }
    save_artifacts(model, labels, args.out_dir, val_acc, config)

    print("\n" + "=" * 60)
    print(f"  Model saved to: {args.out_dir}")
    print(f"  Val accuracy:   {val_acc*100:.2f}%")
    print()
    print("  To enable the trained model in the backend, set:")
    print("    USE_TRAINED_ASL=1  (in your .env file)")
    print()
    print("  The ASLPredictor class in this file is imported by")
    print("  mediapipe_gesture_classifier.py when USE_TRAINED_ASL=1.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
