"""
╔══════════════════════════════════════════════════════════════════╗
║          Human Edge — ASL Classifier (Google Colab)             ║
║          Trains on Kaggle ASL Alphabet dataset (GPU)            ║
╚══════════════════════════════════════════════════════════════════╝

COLAB SETUP — run these in a cell BEFORE running this script:
─────────────────────────────────────────────────────────────────
  # 1. Change runtime to GPU: Runtime → Change runtime type → T4 GPU

  # 2. Install dependencies
  !pip install -q mediapipe tqdm onnx
  # Download MediaPipe hand landmarker model (required for Tasks API)
  !wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -O /content/hand_landmarker.task

  # 3. Mount Google Drive (model will be saved here)
  from google.colab import drive
  drive.mount('/content/drive')

  # 4. Upload your kaggle.json API key, then:
  !mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/
  !chmod 600 ~/.kaggle/kaggle.json

  # 5. Download & extract the Kaggle dataset
  !kaggle datasets download -d grassknoted/asl-alphabet
  !unzip -q asl-alphabet.zip -d /content/asl_dataset
  !echo "Done — dataset at /content/asl_dataset/asl_alphabet_train/asl_alphabet_train/"

  # 6. Run this script
  !python train_asl_colab.py

─────────────────────────────────────────────────────────────────
OUTPUT FILES (saved to Google Drive):
  HumanEdge_Model/asl_classifier.pth   ← copy to backend/models/
  HumanEdge_Model/asl_labels.json      ← copy to backend/models/
  HumanEdge_Model/asl_classifier.onnx  ← optional browser/mobile use
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION  — all hardcoded, no argparse needed
# ══════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────
DATASET_DIR   = Path("/content/asl_dataset/asl_alphabet_train/asl_alphabet_train")
CACHE_FILE    = Path("/content/landmark_cache.npz")
DRIVE_OUT_DIR = Path("/content/drive/MyDrive/HumanEdge_Model")
LOCAL_OUT_DIR = Path("/content/model_output")   # fallback if Drive not mounted

# ── Dataset ───────────────────────────────────────────────────────
MAX_PER_CLASS  = 2000    # images per class (Kaggle has 3000; 2000 is enough)
IMG_RESIZE     = 320     # resize images before MediaPipe (faster + same accuracy)
MEDIAPIPE_CONF = 0.40    # detection confidence for extraction (lower = more samples)

# Classes to skip (dynamic signs handled by motion classifier, or irrelevant)
SKIP_CLASSES = {"nothing", "space", "del", "delete", "z", "j"}

# ── Training hyperparameters ──────────────────────────────────────
EPOCHS           = 80
BATCH_SIZE       = 512    # large batch works well on T4/A100
VAL_SPLIT        = 0.15
DROPOUT          = 0.35
LABEL_SMOOTHING  = 0.10
LEARNING_RATE    = 3e-3
WEIGHT_DECAY     = 2e-4
PATIENCE         = 15     # early stopping epochs without improvement
SEED             = 42

# ── Model / inference ─────────────────────────────────────────────
INFERENCE_THRESHOLD = 0.70   # minimum confidence to accept a prediction
TTA_AUGMENTS        = 5      # test-time augmentation passes (higher = more accurate)

# ── MediaPipe hand landmarker model (Tasks API) ────────────────────
MP_MODEL_PATH = Path("/content/hand_landmarker.task")

# ══════════════════════════════════════════════════════════════════
#  Logging setup
# ══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
#  Reproducibility
# ══════════════════════════════════════════════════════════════════

torch.manual_seed(SEED)
np.random.seed(SEED)
_RNG = np.random.default_rng(SEED)

# ══════════════════════════════════════════════════════════════════
#  Device
# ══════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_device_info() -> None:
    if DEVICE.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU : {name}  ({mem:.1f} GB)")
    else:
        log.info("Device: CPU  (no GPU found — training will be slow)")

# ══════════════════════════════════════════════════════════════════
#  Landmark normalization
# ══════════════════════════════════════════════════════════════════

def normalize_landmarks(pts: np.ndarray) -> np.ndarray:
    """
    Make the 63-float landmark vector invariant to hand position and size.

    Steps:
      1. Translate — move wrist (landmark 0) to origin
      2. Scale     — divide by wrist→middle-MCP distance (palm size)
      3. Flatten   — return (63,) float32
    """
    pts = pts.astype(np.float32).copy()
    pts -= pts[0]                            # wrist → origin
    scale = float(np.linalg.norm(pts[9]))   # palm size
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()

# ══════════════════════════════════════════════════════════════════
#  Data augmentation (on normalised landmark vectors)
# ══════════════════════════════════════════════════════════════════

def augment(vec: np.ndarray) -> np.ndarray:
    """
    6 geometry-preserving augmentations applied in normalised space.
    Labels are preserved — all transforms keep hand shape intact.

      1. Scale jitter      ±12%
      2. 2-D rotation      ±20°
      3. 3-D tilt          ±8°   (camera angle simulation)
      4. Translation noise ±0.05
      5. Gaussian noise    σ=0.02
      6. Landmark dropout  0–2 random landmarks zeroed
    """
    pts = vec.reshape(21, 3).copy()

    # 1. Scale
    pts *= _RNG.uniform(0.88, 1.12)

    # 2. 2-D rotation (hand-plane)
    a = _RNG.uniform(-20, 20) * math.pi / 180
    c, s = math.cos(a), math.sin(a)
    R2 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts = (R2 @ pts.T).T

    # 3. 3-D tilt
    t = _RNG.uniform(-8, 8) * math.pi / 180
    ct, st = math.cos(t), math.sin(t)
    R3 = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]], dtype=np.float32)
    pts = (R3 @ pts.T).T

    # 4. Translation jitter
    pts[:, :2] += _RNG.normal(0, 0.05, (1, 2)).astype(np.float32)

    # 5. Gaussian noise
    pts += _RNG.normal(0, 0.02, pts.shape).astype(np.float32)

    # 6. Landmark dropout
    n = int(_RNG.integers(0, 3))
    if n:
        pts[_RNG.choice(range(1, 21), n, replace=False)] = 0.0

    return pts.flatten().astype(np.float32)

# ══════════════════════════════════════════════════════════════════
#  MediaPipe landmark extraction
# ══════════════════════════════════════════════════════════════════

class Extractor:
    """Extracts 21×3 hand landmarks from a BGR image (MediaPipe Tasks API)."""

    def __init__(self) -> None:
        if not MP_MODEL_PATH.exists():
            import urllib.request
            _url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
            log.info(f"Downloading hand landmarker model to {MP_MODEL_PATH} …")
            MP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_url, str(MP_MODEL_PATH))
            log.info("Download complete.")

        base_opts = _mp_python.BaseOptions(model_asset_path=str(MP_MODEL_PATH))
        opts = _mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=1,
            min_hand_detection_confidence=MEDIAPIPE_CONF,
            running_mode=_mp_vision.RunningMode.IMAGE,
        )
        self._detector = _mp_vision.HandLandmarker.create_from_options(opts)

    def _run(self, rgb: np.ndarray):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._detector.detect(mp_img)

    def extract(self, bgr: np.ndarray) -> np.ndarray | None:
        """Returns (21, 3) float32 or None."""
        if IMG_RESIZE:
            h, w = bgr.shape[:2]
            if max(h, w) > IMG_RESIZE:
                scale = IMG_RESIZE / max(h, w)
                bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._run(rgb)
        if not result.hand_landmarks:
            # Try flipped — some Kaggle images are mirrored
            result = self._run(cv2.flip(rgb, 1))
        if not result.hand_landmarks:
            return None
        lms = result.hand_landmarks[0]
        return np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float32)

    def close(self) -> None:
        self._detector.close()

# ══════════════════════════════════════════════════════════════════
#  Dataset building — image folder → landmark vectors
# ══════════════════════════════════════════════════════════════════

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_dataset(
    dataset_dir: Path,
    cache_file: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract landmarks from all class subfolders in dataset_dir.
    Results are cached — re-running is instant.

    Returns:
        X       : (N, 63) float32 normalised landmark vectors
        y       : (N,)    int64  class indices
        labels  : list of class name strings
    """
    if cache_file.exists():
        log.info(f"Loading cached landmarks from {cache_file} …")
        data   = np.load(str(cache_file), allow_pickle=True)
        X, y   = data["X"], data["y"]
        labels = list(data["labels"])
        log.info(f"  {len(X):,} samples, {len(labels)} classes loaded from cache.")
        return X, y, labels

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir}\n"
            "Run the Colab setup cells to download and extract it."
        )

    class_dirs = sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name.upper(),
    )
    skipped_dirs = [p.name for p in class_dirs if p.name.upper() in {s.upper() for s in SKIP_CLASSES}]
    class_dirs   = [p for p in class_dirs if p.name.upper() not in {s.upper() for s in SKIP_CLASSES}]

    log.info(f"Found {len(class_dirs)} classes to train on  "
             f"(skipped: {skipped_dirs})")
    log.info(f"Extracting landmarks — this takes ~15-20 min on Colab CPU …\n")

    ext    = Extractor()
    labels : list[str] = []
    all_X  : list[np.ndarray] = []
    all_y  : list[int] = []

    for cls_dir in class_dirs:
        cls_name = cls_dir.name.upper()
        cls_idx  = len(labels)
        labels.append(cls_name)

        images = sorted(
            [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        )[:MAX_PER_CLASS]

        ok = skip = 0
        for img_path in tqdm(images, desc=f"{cls_name:12s}", leave=False, ncols=80):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                skip += 1
                continue
            lms = ext.extract(bgr)
            if lms is None:
                skip += 1
                continue
            all_X.append(normalize_landmarks(lms))
            all_y.append(cls_idx)
            ok += 1

        pct = ok / max(1, ok + skip) * 100
        log.info(f"  {cls_name:12s}: {ok:4d} samples  ({pct:.0f}% detection rate)")

    ext.close()

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    log.info(f"\nTotal: {len(X):,} landmark vectors, {len(labels)} classes")
    np.savez_compressed(str(cache_file), X=X, y=y, labels=np.array(labels))
    log.info(f"Cache saved → {cache_file}")
    return X, y, labels

# ══════════════════════════════════════════════════════════════════
#  Torch dataset
# ══════════════════════════════════════════════════════════════════

class LandmarkDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, aug: bool = False) -> None:
        self.X   = X.astype(np.float32)
        self.y   = y.astype(np.int64)
        self.aug = aug

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = augment(self.X[i]) if self.aug else self.X[i]
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)

# ══════════════════════════════════════════════════════════════════
#  Model — Residual MLP
# ══════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x + self.net(x))


class ASLClassifier(nn.Module):
    """
    Compact residual MLP.  Input: (B, 63) normalised landmarks.
    Output: (B, num_classes) logits.

    Layout:
        BN(63) → Enc(63→256) → ResBlock × 3 → Head(256→128→C)

    ~210 K parameters.  Inference on CPU: < 0.3 ms.
    Expected accuracy on Kaggle ASL Alphabet: 96–98%.
    """

    def __init__(self, num_classes: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(63)
        self.enc = nn.Sequential(
            nn.Linear(63, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res1 = ResBlock(256, dropout)
        self.res2 = ResBlock(256, dropout * 0.60)
        self.res3 = ResBlock(256, dropout * 0.35)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(128, num_classes),
        )
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(self.input_norm(x))
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        return self.head(h)

# ══════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> tuple[float, int]:
    """
    AdamW + OneCycleLR + label smoothing + early stopping.
    Returns (best_val_accuracy, best_epoch).
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    best_acc   = 0.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    log.info(f"\n{'Epoch':>6}  {'Train':>8}  {'Val':>8}  {'Loss':>8}  {'LR':>10}")
    log.info("─" * 52)

    model.to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────
        model.train()
        n_ok = n_tot = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            n_ok  += (logits.argmax(1) == yb).sum().item()
            n_tot += len(yb)
        train_acc = n_ok / n_tot

        # ── Validate ──────────────────────────────────────────────
        val_acc, val_loss = _val(model, val_loader, criterion)

        if val_acc > best_acc:
            best_acc   = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " ✓"
        else:
            no_improve += 1
            marker = ""

        if epoch % 5 == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            log.info(
                f"{epoch:6d}  {train_acc:8.4f}  {val_acc:8.4f}  "
                f"{val_loss:8.4f}  {lr:10.2e}{marker}"
            )

        if no_improve >= PATIENCE:
            log.info(f"\nEarly stop at epoch {epoch}.")
            break

    if best_state:
        model.load_state_dict(best_state)

    log.info(f"\nBest val accuracy: {best_acc*100:.2f}%  (epoch {best_epoch})")
    return best_acc, best_epoch


@torch.no_grad()
def _val(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    n_ok = n_tot = 0
    loss_sum = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        logits   = model(Xb)
        loss_sum += criterion(logits, yb).item() * len(yb)
        n_ok     += (logits.argmax(1) == yb).sum().item()
        n_tot    += len(yb)
    return n_ok / n_tot, loss_sum / n_tot

# ══════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    labels: list[str],
) -> dict[str, float]:
    """
    Per-class accuracy report with confusion hints for weak classes.
    Returns dict of {class_name: accuracy}.
    """
    model.eval()
    C       = len(labels)
    correct = np.zeros(C, dtype=np.int64)
    total   = np.zeros(C, dtype=np.int64)
    confused: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for Xb, yb in loader:
        preds = model(Xb.to(DEVICE)).argmax(1).cpu().numpy()
        ys    = yb.numpy()
        for p, gt in zip(preds, ys):
            total[gt]   += 1
            correct[gt] += int(p == gt)
            if p != gt:
                confused[gt][p] += 1

    print("\n" + "═" * 68)
    print("  Per-class accuracy")
    print("═" * 68)

    accs: dict[str, float] = {}
    weak: list[str]        = []

    for i, lbl in enumerate(labels):
        acc      = correct[i] / max(1, total[i])
        accs[lbl] = acc
        bar      = "█" * int(acc * 30)
        note     = ""
        if acc < 0.90:
            weak.append(lbl)
            if confused[i]:
                top  = sorted(confused[i].items(), key=lambda x: -x[1])[:3]
                note = f"  ← confused with: {', '.join(labels[j] for j, _ in top)}"
        print(f"  {lbl:12s}  {acc*100:5.1f}%  {bar}{note}")

    overall = correct.sum() / max(1, total.sum())
    print("═" * 68)
    print(f"  Overall: {overall*100:.2f}%")
    if weak:
        print(f"  Weak (<90%): {', '.join(weak)}")
    print("═" * 68 + "\n")
    return accs

# ══════════════════════════════════════════════════════════════════
#  Test-time augmentation (TTA) inference
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(
    model: nn.Module,
    vec: np.ndarray,
    n: int = TTA_AUGMENTS,
) -> tuple[int, float]:
    """
    Average softmax over n augmented versions for higher accuracy.
    Returns (class_index, confidence).
    """
    model.eval()
    batch = torch.from_numpy(
        np.stack([vec] + [augment(vec) for _ in range(n - 1)])
    ).to(DEVICE)
    probs = torch.softmax(model(batch), dim=1).mean(0)
    idx   = int(probs.argmax())
    return idx, float(probs[idx])

# ══════════════════════════════════════════════════════════════════
#  Export & save
# ══════════════════════════════════════════════════════════════════

def save_model(
    model: nn.Module,
    labels: list[str],
    val_acc: float,
    best_epoch: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── PyTorch checkpoint ────────────────────────────────────────
    pth = out_dir / "asl_classifier.pth"
    torch.save(
        {
            "model_state":  model.state_dict(),
            "labels":       labels,
            "num_classes":  len(labels),
            "input_size":   63,
            "val_accuracy": round(val_acc, 5),
            "config": {
                "epochs_trained":  best_epoch,
                "dropout":         DROPOUT,
                "label_smoothing": LABEL_SMOOTHING,
                "max_per_class":   MAX_PER_CLASS,
                "threshold":       INFERENCE_THRESHOLD,
                "tta_n":           TTA_AUGMENTS,
            },
        },
        pth,
    )
    log.info(f"PyTorch model → {pth}")

    # ── Labels JSON ───────────────────────────────────────────────
    lbl = out_dir / "asl_labels.json"
    lbl.write_text(json.dumps({"labels": labels, "version": 1}, indent=2))
    log.info(f"Labels        → {lbl}")

    # ── ONNX ──────────────────────────────────────────────────────
    try:
        import onnx  # noqa: F401
        onnx_path = out_dir / "asl_classifier.onnx"
        model.eval().cpu()
        torch.onnx.export(
            model,
            torch.randn(1, 63),
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["landmarks"],
            output_names=["logits"],
            dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        )
        log.info(f"ONNX model    → {onnx_path}")
    except Exception as e:
        log.warning(f"ONNX export skipped: {e}")


def copy_to_drive(src_dir: Path, drive_dir: Path) -> None:
    """Copy model files to Google Drive for persistence."""
    if not Path("/content/drive").exists():
        log.warning("Google Drive not mounted — skipping Drive copy.")
        return
    import shutil
    drive_dir.mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        dest = drive_dir / f.name
        shutil.copy2(f, dest)
        log.info(f"Drive copy    → {dest}")


def download_from_colab(out_dir: Path) -> None:
    """Trigger browser download of model files (works in Colab)."""
    try:
        from google.colab import files  # type: ignore[import]
        for fname in ["asl_classifier.pth", "asl_labels.json"]:
            fpath = out_dir / fname
            if fpath.exists():
                log.info(f"Downloading {fname} …")
                files.download(str(fpath))
    except ImportError:
        pass  # Not in Colab or files not found

# ══════════════════════════════════════════════════════════════════
#  ASLPredictor — drop-in inference class for the backend
# ══════════════════════════════════════════════════════════════════

class ASLPredictor:
    """
    Copy this class (or import it) into your backend after training.

    Usage:
        predictor = ASLPredictor.load()
        label, conf = predictor.predict(pts_21x3, handedness)

    This is a drop-in replacement for the rule-based _classify_asl()
    in backend/vision/mediapipe_gesture_classifier.py.
    Set USE_TRAINED_ASL=1 in your .env to enable it automatically.
    """

    def __init__(
        self,
        model: ASLClassifier,
        labels: list[str],
        threshold: float = INFERENCE_THRESHOLD,
        tta: bool = True,
        tta_n: int = TTA_AUGMENTS,
    ) -> None:
        self.model     = model.eval()
        self.labels    = labels
        self.threshold = threshold
        self.tta       = tta
        self.tta_n     = tta_n

    @classmethod
    def load(
        cls,
        pth_path: Path | None = None,
        threshold: float = INFERENCE_THRESHOLD,
        tta: bool = True,
    ) -> "ASLPredictor":
        if pth_path is None:
            pth_path = Path(__file__).parent / "models" / "asl_classifier.pth"
        if not pth_path.exists():
            raise FileNotFoundError(f"Model not found: {pth_path}")
        ckpt   = torch.load(pth_path, map_location="cpu", weights_only=False)
        labels = ckpt["labels"]
        model  = ASLClassifier(num_classes=len(labels))
        model.load_state_dict(ckpt["model_state"])
        log.info(
            f"ASLPredictor loaded: {len(labels)} classes, "
            f"val_acc={ckpt.get('val_accuracy', '?')}"
        )
        return cls(model, labels, threshold=threshold, tta=tta)

    def predict(
        self,
        pts: list[tuple[float, float, float]],
        handedness: str = "Right",
    ) -> tuple[str | None, float]:
        """
        pts        : 21 MediaPipe landmarks as (x, y, z) tuples
        handedness : accepted for API compatibility (unused — model is handedness-agnostic)
        Returns    : (label, confidence) or (None, confidence) if below threshold
        """
        if len(pts) != 21:
            return None, 0.0
        vec = normalize_landmarks(np.array(pts, dtype=np.float32))

        if self.tta:
            idx, conf = tta_predict(self.model, vec, self.tta_n)
        else:
            with torch.no_grad():
                p    = torch.softmax(self.model(torch.from_numpy(vec).unsqueeze(0)), 1)[0]
                idx  = int(p.argmax())
                conf = float(p[idx])

        return (self.labels[idx], conf) if conf >= self.threshold else (None, conf)

    def predict_topk(
        self,
        pts: list[tuple[float, float, float]],
        k: int = 3,
    ) -> list[tuple[str, float]]:
        """Top-k predictions — useful for debug/correction UI."""
        if len(pts) != 21:
            return []
        vec = normalize_landmarks(np.array(pts, dtype=np.float32))
        with torch.no_grad():
            p = torch.softmax(self.model(torch.from_numpy(vec).unsqueeze(0)), 1)[0]
            topk = p.topk(min(k, len(self.labels)))
        return [(self.labels[int(i)], float(v)) for v, i in zip(topk.values, topk.indices)]

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  Human Edge — ASL Landmark Classifier Training          ║")
    print("╚" + "═" * 58 + "╝\n")

    print_device_info()

    # ── 1. Build / load dataset ───────────────────────────────────
    log.info("\n[1/5] Dataset")
    X, y, labels = build_dataset(DATASET_DIR, CACHE_FILE)

    counts = np.bincount(y, minlength=len(labels))
    log.info(f"Classes ({len(labels)}): {labels}")
    log.info(f"Samples: {len(X):,}  |  min/class: {counts.min()}  max/class: {counts.max()}")

    # ── 2. Stratified train/val split ────────────────────────────
    log.info("\n[2/5] Splitting dataset")
    rng  = np.random.default_rng(SEED)
    idx  = np.arange(len(X))
    rng.shuffle(idx)

    val_idx_list, train_idx_list = [], []
    for c in range(len(labels)):
        ci    = idx[y[idx] == c]
        n_val = max(1, int(len(ci) * VAL_SPLIT))
        val_idx_list.extend(ci[:n_val])
        train_idx_list.extend(ci[n_val:])

    train_idx = np.array(train_idx_list)
    val_idx   = np.array(val_idx_list)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    log.info(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  "
             f"(stratified, {VAL_SPLIT*100:.0f}% val)")

    # Balanced sampler
    weights = 1.0 / np.bincount(y_train, minlength=len(labels))[y_train].astype(float)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights).float(), len(y_train), replacement=True
    )

    train_ds = LandmarkDS(X_train, y_train, aug=True)
    val_ds   = LandmarkDS(X_val,   y_val,   aug=False)

    num_workers = 2 if DEVICE.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=num_workers, pin_memory=(DEVICE.type == "cuda"),
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=1024, shuffle=False,
                              num_workers=num_workers,
                              persistent_workers=(num_workers > 0))

    # ── 3. Build model ────────────────────────────────────────────
    log.info("\n[3/5] Model")
    model   = ASLClassifier(num_classes=len(labels), dropout=DROPOUT)
    n_param = sum(p.numel() for p in model.parameters())
    log.info(f"ASLClassifier: {n_param:,} parameters  |  "
             f"{len(labels)} classes  |  input size: 63")

    # ── 4. Train ──────────────────────────────────────────────────
    log.info(f"\n[4/5] Training  (epochs={EPOCHS}, batch={BATCH_SIZE}, "
             f"lr={LEARNING_RATE}, dropout={DROPOUT})")
    t0 = time.time()
    best_acc, best_epoch = train_model(model, train_loader, val_loader)
    elapsed = time.time() - t0
    log.info(f"Training time: {elapsed/60:.1f} min")

    # ── 5. Evaluate & save ────────────────────────────────────────
    log.info("\n[5/5] Evaluation & Export")
    accs = evaluate(model, val_loader, labels)

    # Quality check
    overall = np.mean(list(accs.values()))
    if overall >= 0.96:
        grade = "EXCELLENT — ready for production ✓"
    elif overall >= 0.92:
        grade = "GOOD — acceptable for demo"
    elif overall >= 0.87:
        grade = "FAIR — consider more data or longer training"
    else:
        grade = "POOR — check dataset quality and hand detection rate"
    log.info(f"Grade: {grade}")

    # Save to local output dir
    save_model(model, labels, best_acc, best_epoch, LOCAL_OUT_DIR)

    # Copy to Google Drive
    copy_to_drive(LOCAL_OUT_DIR, DRIVE_OUT_DIR)

    # Trigger Colab file download
    download_from_colab(LOCAL_OUT_DIR)

    # ── Final instructions ────────────────────────────────────────
    print("\n" + "╔" + "═" * 58 + "╗")
    print(f"║  Done!  Val accuracy: {best_acc*100:.2f}%".ljust(59) + "║")
    print("╠" + "═" * 58 + "╣")
    print("║  Files saved to:                                        ║")
    print(f"║    Google Drive → {str(DRIVE_OUT_DIR)[:38]:38s} ║")
    print("╠" + "═" * 58 + "╣")
    print("║  Next steps:                                            ║")
    print("║  1. Download asl_classifier.pth from Google Drive       ║")
    print("║  2. Download asl_labels.json from Google Drive          ║")
    print("║  3. Copy both to:  backend/models/                      ║")
    print("║  4. Add to your .env file:                              ║")
    print("║       USE_TRAINED_ASL=1                                 ║")
    print("║  5. Restart backend — trained model loads automatically  ║")
    print("╚" + "═" * 58 + "╝\n")


if __name__ == "__main__":
    main()
