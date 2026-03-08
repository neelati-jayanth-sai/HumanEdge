/**
 * Client-side gesture detection using MediaPipe Tasks Vision.
 *
 * The built-in GestureRecognizer model recognises only 8 gestures, but it also
 * outputs 21 hand landmarks with every frame.  We run our own ASL rule-based
 * classifier on those landmarks so most of the ASL alphabet is available
 * without any additional model download.
 *
 * Priority:  ASL rule classifier (confident) → built-in gesture → null
 */

const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** MediaPipe WASM files CDN — must match the installed npm package version. */
const WASM_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

/** Model served by our own backend so no external dependency is needed at runtime. */
const MODEL_URL = `${API_URL}/models/gesture_recognizer.task`;

// ---------------------------------------------------------------------------
// Suppress MediaPipe WASM log noise from the Next.js dev-tools error overlay.
// ---------------------------------------------------------------------------

let _wasmFilterInstalled = false;

function _installWasmLogFilter(): void {
  if (_wasmFilterInstalled || typeof window === "undefined") return;
  _wasmFilterInstalled = true;

  const prev = console.error.bind(console);
  console.error = function wasmLogFilter(...args: unknown[]) {
    const m = String(args[0] ?? "");
    if (
      /^(INFO|WARNING|W\d{4}|I\d{4}):/.test(m) ||
      m.includes("TensorFlow Lite") ||
      m.includes("XNNPACK") ||
      m.includes("XNNPack") ||
      m.includes("feedback tensor") ||
      m.includes("Feedback manager") ||
      m.includes("gesture_recognizer_graph") ||
      m.includes("hand_gesture_recognizer")
    ) {
      return;
    }
    prev(...args);
  };
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type GestureResult = {
  label: string | null;
  confidence: number;
  handTracked: boolean;
};

export type DetectorState = "idle" | "loading" | "ready" | "error";

// ---------------------------------------------------------------------------
// ASL rule-based classifier (mirrors the Python backend logic)
// ---------------------------------------------------------------------------

type Pt = { x: number; y: number; z: number };

function dist2D(a: Pt, b: Pt): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function fingerExtended(tip: Pt, pip: Pt, mcp: Pt): boolean {
  const vx = pip.x - mcp.x, vy = pip.y - mcp.y;
  const wx = tip.x - pip.x, wy = tip.y - pip.y;
  const dot = vx * wx + vy * wy;
  const dTip = dist2D(tip, mcp);
  const dPip = dist2D(pip, mcp);
  return dot > 0 && dTip > dPip * 0.85;
}

function fingerCurled(tip: Pt, pip: Pt, mcp: Pt): boolean {
  return dist2D(tip, mcp) < dist2D(pip, mcp) * 0.92;
}

/**
 * Classify ASL hand shape from 21 MediaPipe normalized landmarks.
 * Recognises: A B C D E F G I K L O R S U V W X Y
 *             OPEN_PALM  THUMB_UP  THUMB_DOWN  ILY
 */
function classifyASL(
  lms: Pt[],
  handedness: string,
): { label: string | null; confidence: number } {
  if (lms.length !== 21) return { label: null, confidence: 0 };

  const wrist    = lms[0];
  const thumbTip = lms[4], thumbIp = lms[3];
  const idxTip   = lms[8],  idxPip  = lms[6],  idxMcp  = lms[5];
  const midTip   = lms[12], midPip  = lms[10], midMcp  = lms[9];
  const rngTip   = lms[16], rngPip  = lms[14], rngMcp  = lms[13];
  const pkyTip   = lms[20], pkyPip  = lms[18], pkyMcp  = lms[17];

  const palmSz = Math.max(1e-6, dist2D(wrist, lms[9]));

  const idxUp = fingerExtended(idxTip, idxPip, idxMcp);
  const midUp = fingerExtended(midTip, midPip, midMcp);
  const rngUp = fingerExtended(rngTip, rngPip, rngMcp);
  const pkyUp = fingerExtended(pkyTip, pkyPip, pkyMcp);

  // Thumb state
  const thumbSpread   = dist2D(thumbTip, idxMcp) / palmSz > 0.42;
  const thumbDirOpen  = handedness === "Right"
    ? thumbTip.x < thumbIp.x
    : thumbTip.x > thumbIp.x;
  const thumbOpen = thumbSpread || thumbDirOpen;

  // Curl checks
  const idxCurl = fingerCurled(idxTip, idxPip, idxMcp);
  const midCurl = fingerCurled(midTip, midPip, midMcp);
  const rngCurl = fingerCurled(rngTip, rngPip, rngMcp);
  const pkyCurl = fingerCurled(pkyTip, pkyPip, pkyMcp);

  const nUp = [idxUp, midUp, rngUp, pkyUp].filter(Boolean).length;

  // Direction: is the index finger pointing more horizontal than vertical?
  // Used to distinguish H (sideways) from U / V / K (upward).
  const idxDx = Math.abs(idxTip.x - idxMcp.x);
  const idxDy = Math.abs(idxTip.y - idxMcp.y);
  const fingersHorizontal = idxUp && (idxDx > idxDy * 1.1);

  // Normalized distances
  const d = (a: Pt, b: Pt) => dist2D(a, b) / palmSz;
  const dTI = d(thumbTip, idxTip);
  const dTM = d(thumbTip, midTip);
  const dTR = d(thumbTip, rngTip);
  const dIM = d(idxTip,   midTip);
  const dMR = d(midTip,   rngTip);

  // ── C: early detection — must come before nUp-based logic ───────────────
  // C: fingers curve forward (not pointing up), thumb spread, NOT fisted
  // Uses avgTipY vs avgMcpY: in OPEN_PALM tips are clearly ABOVE MCPs;
  // in C they are at the same height or lower (curled forward).
  {
    const avgTipY = (idxTip.y + midTip.y + rngTip.y + pkyTip.y) / 4;
    const avgMcpY = (idxMcp.y + midMcp.y + rngMcp.y + pkyMcp.y) / 4;
    const tipsNotAboveMcps = avgTipY >= avgMcpY - palmSz * 0.12;
    if (thumbOpen && tipsNotAboveMcps && !idxUp && !pkyUp && dTI > 0.26 && !idxCurl)
      return { label: "C", confidence: 0.82 };
  }

  // ── 4 fingers up ─────────────────────────────────────────────────────────

  // B: 4 fingers together, thumb tucked
  if (nUp === 4 && !thumbOpen && dIM < 0.22)
    return { label: "B", confidence: 0.88 };

  // OPEN_PALM / 5 / HELLO
  if (nUp === 4 && thumbOpen)
    return { label: "OPEN_PALM", confidence: 0.93 };

  // ── 3 fingers up ─────────────────────────────────────────────────────────

  // W (and WATER)
  if (idxUp && midUp && rngUp && !pkyUp) {
    const conf = (dIM > 0.14 && dMR > 0.14) ? 0.87 : 0.78;
    return { label: "W", confidence: conf };
  }

  // F: middle + ring + pinky up, index-thumb pinch
  if (midUp && rngUp && pkyUp && !idxUp && dTI < 0.30)
    return { label: "F", confidence: 0.86 };

  // ── Index + pinky combos ─────────────────────────────────────────────────

  // ILY: index + pinky + thumb
  if (idxUp && pkyUp && !midUp && !rngUp && thumbOpen)
    return { label: "ILY", confidence: 0.90 };

  // ── Pinky-only ───────────────────────────────────────────────────────────

  // Y: thumb + pinky
  if (pkyUp && !idxUp && !midUp && !rngUp && thumbOpen)
    return { label: "Y", confidence: 0.88 };

  // I: pinky only
  if (pkyUp && !idxUp && !midUp && !rngUp && !thumbOpen)
    return { label: "I", confidence: 0.88 };

  // ── Index + middle ───────────────────────────────────────────────────────

  if (idxUp && midUp && !rngUp && !pkyUp) {
    // H: index + middle extended sideways (horizontal), not upward
    if (fingersHorizontal && dIM < 0.22 && !thumbOpen)
      return { label: "H", confidence: 0.82 };

    // K: thumb open + fingers pointing upward (not horizontal)
    if (thumbOpen && !fingersHorizontal)
      return { label: "K", confidence: 0.83 };

    if (!fingersHorizontal) {
      // R: index crossed over middle
      const crossed = handedness === "Right"
        ? idxTip.x > midTip.x
        : idxTip.x < midTip.x;
      if (crossed && dIM < 0.18) return { label: "R", confidence: 0.82 };

      // V: spread
      if (dIM > 0.26) return { label: "V", confidence: 0.90 };

      // U: together
      return { label: "U", confidence: 0.86 };
    }

    // Horizontal + thumb open — angled H/K variant
    return { label: "H", confidence: 0.78 };
  }

  // ── Index only ───────────────────────────────────────────────────────────

  if (idxUp && !midUp && !rngUp && !pkyUp) {
    // L: index + thumb out = L shape
    if (thumbOpen) return { label: "L", confidence: 0.88 };

    // D: index up, thumb close (forms circle/arch with other fingers)
    if (!thumbOpen && dTI < 0.32) return { label: "D", confidence: 0.83 };

    // X: hooked index (tip bent back below pip)
    if (idxTip.y > idxPip.y + 0.015) return { label: "X", confidence: 0.80 };

    // G: pointing
    return { label: "G", confidence: 0.80 };
  }

  // ── All fingers down (fist variants) ─────────────────────────────────────

  if (nUp === 0) {
    // O: all tips pinch toward thumb
    if (dTI < 0.34 && dTM < 0.44 && dTR < 0.54)
      return { label: "O", confidence: 0.84 };

    // C: partially bent (not curled, not extended), thumb open
    if (thumbOpen && !idxCurl && !midCurl && !rngCurl && !pkyCurl && dTI > 0.34)
      return { label: "C", confidence: 0.78 };

    if (idxCurl && midCurl && rngCurl) {
      // T: thumb tucked between index and middle at pip height
      if (!thumbOpen && pkyCurl) {
        const midKnuckleX = (idxMcp.x + midMcp.x) / 2;
        const knuckleWidth = Math.abs(idxMcp.x - midMcp.x) + palmSz * 0.3;
        const thumbBetween = Math.abs(thumbTip.x - midKnuckleX) < knuckleWidth * 0.7;
        const thumbAtPip   = Math.abs(thumbTip.y - idxPip.y) < palmSz * 0.45;
        if (thumbBetween && thumbAtPip)
          return { label: "T", confidence: 0.78 };
      }

      if (thumbOpen) {
        // THUMB_UP / THUMB_DOWN
        const palmCy = lms[9].y;
        if (thumbTip.y < palmCy - 0.06) return { label: "THUMB_UP",   confidence: 0.88 };
        if (thumbTip.y > palmCy + 0.06) return { label: "THUMB_DOWN", confidence: 0.88 };
        return { label: "A", confidence: 0.78 };
      }
      // S: thumb over knuckles
      if (thumbTip.y < idxPip.y - 0.015) return { label: "S", confidence: 0.83 };
      // E: tips deep-curled toward wrist
      if (d(idxTip, wrist) < d(idxMcp, wrist) * 0.85 &&
          d(midTip, wrist) < d(midMcp, wrist) * 0.85)
        return { label: "E", confidence: 0.78 };
      // A: thumb alongside fist
      return { label: "A", confidence: 0.84 };
    }
  }

  return { label: null, confidence: 0 };
}

// ---------------------------------------------------------------------------
// Motion helpers
// ---------------------------------------------------------------------------

function handOpenRatio(lms: Pt[]): number {
  const wrist = lms[0];
  const palmSz = Math.max(1e-6, dist2D(wrist, lms[9]));
  return ([8, 12, 16, 20] as const)
    .reduce((acc, i) => acc + dist2D(lms[i], wrist) / palmSz, 0) / 4;
}

function pinchDist(lms: Pt[]): number {
  const palmSz = Math.max(1e-6, dist2D(lms[0], lms[9]));
  return dist2D(lms[4], lms[8]) / palmSz;
}

function countReversals(vals: number[], minDelta: number): number {
  let n = 0;
  for (let i = 2; i < vals.length; i++) {
    const d1 = vals[i - 1] - vals[i - 2];
    const d2 = vals[i]     - vals[i - 1];
    if (Math.abs(d1) > minDelta && Math.abs(d2) > minDelta && d1 * d2 < 0) n++;
  }
  return n;
}

// ---------------------------------------------------------------------------
// Browser motion classifier — mirrors backend/vision/motion_classifier.py
// ---------------------------------------------------------------------------

type MotionDetection = { label: string | null; confidence: number };

class BrowserMotionClassifier {
  private frames: Pt[][] = [];
  private readonly WINDOW = 25;
  private cooldowns = new Map<string, number>();
  private _nullStreak = 0;
  private readonly NULL_RESET_AFTER = 8; // only wipe window after 8 consecutive no-hand frames (~0.5s)

  update(landmarks: Pt[] | null): MotionDetection {
    if (!landmarks) {
      this._nullStreak++;
      if (this._nullStreak >= this.NULL_RESET_AFTER) this.frames = [];
      return { label: null, confidence: 0 };
    }
    this._nullStreak = 0;
    this.frames.push(landmarks);
    if (this.frames.length > this.WINDOW) this.frames.shift();

    const now = performance.now();
    const frms = this.frames;

    const candidates: MotionDetection[] = [
      this._wave(frms, now),
      this._thankYou(frms, now),
      this._please(frms, now),
      this._sorry(frms, now),
      this._yesNod(frms, now),
      this._more(frms, now),
      this._eat(frms, now),
      this._come(frms, now),
    ];
    return candidates.reduce(
      (best, c) => (c.label && c.confidence > best.confidence ? c : best),
      { label: null, confidence: 0 },
    );
  }

  reset(): void { this.frames = []; this.cooldowns.clear(); this._nullStreak = 0; }

  private _cd(label: string, ms: number, now: number): boolean {
    return (now - (this.cooldowns.get(label) ?? 0)) < ms;
  }
  private _fire(label: string, conf: number, now: number): MotionDetection {
    this.cooldowns.set(label, now);
    return { label, confidence: conf };
  }

  private _wave(frms: Pt[][], now: number): MotionDetection {
    if (this._cd("WAVE", 1500, now) || frms.length < 12) return { label: null, confidence: 0 };
    const xs = frms.map(f => f[0].x);
    const openness = frms.slice(-8).reduce((a, f) => a + handOpenRatio(f), 0) / 8;
    if (openness < 0.6) return { label: null, confidence: 0 };
    const r = countReversals(xs, 0.015);
    return r >= 3 ? this._fire("WAVE", Math.min(0.90, 0.70 + r * 0.05), now) : { label: null, confidence: 0 };
  }

  private _thankYou(frms: Pt[][], now: number): MotionDetection {
    if (this._cd("THANK_YOU", 1200, now) || frms.length < 10) return { label: null, confidence: 0 };
    const wy0 = frms[0][0].y, wy1 = frms[frms.length - 1][0].y;
    const dy = wy1 - wy0;
    if (handOpenRatio(frms[0]) > 0.6 && wy0 > 0.3 && wy0 < 0.65 && dy > 0.06)
      return this._fire("THANK_YOU", 0.82, now);
    return { label: null, confidence: 0 };
  }

  private _circularArc(frms: Pt[][]): { arc: number; radius: number } {
    const wxs = frms.map(f => f[0].x), wys = frms.map(f => f[0].y);
    const cx = wxs.reduce((a, b) => a + b, 0) / wxs.length;
    const cy = wys.reduce((a, b) => a + b, 0) / wys.length;
    const angles = wxs.map((x, i) => Math.atan2(wys[i] - cy, x - cx));
    const unwrapped = [angles[0]];
    for (let i = 1; i < angles.length; i++) {
      let diff = angles[i] - angles[i - 1];
      diff = ((diff + Math.PI) % (2 * Math.PI)) - Math.PI;
      unwrapped.push(unwrapped[i - 1] + diff);
    }
    const arc = Math.abs(unwrapped[unwrapped.length - 1] - unwrapped[0]);
    const radius = Math.max(...wxs.map((x, i) => Math.sqrt((x - cx) ** 2 + (wys[i] - cy) ** 2)));
    return { arc, radius };
  }

  private _please(frms: Pt[][], now: number): MotionDetection {
    if (this._cd("PLEASE", 1200, now) || frms.length < 14) return { label: null, confidence: 0 };
    if (handOpenRatio(frms[frms.length - 1]) < 0.55) return { label: null, confidence: 0 };
    const { arc, radius } = this._circularArc(frms);
    if (arc > Math.PI * 1.2 && radius > 0.03)
      return this._fire("PLEASE", Math.min(0.88, 0.70 + arc / (2 * Math.PI) * 0.1), now);
    return { label: null, confidence: 0 };
  }

  private _sorry(frms: Pt[][], now: number): MotionDetection {
    if (this._cd("SORRY", 1500, now) || frms.length < 14) return { label: null, confidence: 0 };
    if (handOpenRatio(frms[frms.length - 1]) > 0.40) return { label: null, confidence: 0 };
    const { arc, radius } = this._circularArc(frms);
    if (arc > Math.PI * 1.2 && radius > 0.03)
      return this._fire("SORRY", Math.min(0.88, 0.70 + arc / (2 * Math.PI) * 0.1), now);
    return { label: null, confidence: 0 };
  }

  private _yesNod(frms: Pt[][], now: number): MotionDetection {
    if (this._cd("YES_NOD", 1200, now) || frms.length < 10) return { label: null, confidence: 0 };
    const wys = frms.map(f => f[0].y);
    const openness = frms.reduce((a, f) => a + handOpenRatio(f), 0) / frms.length;
    if (openness > 0.45) return { label: null, confidence: 0 };
    const r = countReversals(wys, 0.02);
    return r >= 3 ? this._fire("YES_NOD", 0.83, now) : { label: null, confidence: 0 };
  }

  private _more(frms: Pt[][], now: number): MotionDetection {
    if (this._cd("MORE", 1200, now) || frms.length < 10) return { label: null, confidence: 0 };
    const pinches = frms.map(f => pinchDist(f));
    const r = countReversals(pinches, 0.08);
    return r >= 3 ? this._fire("MORE", 0.80, now) : { label: null, confidence: 0 };
  }

  private _eat(frms: Pt[][], now: number): MotionDetection {
    if (this._cd("EAT", 1200, now) || frms.length < 12) return { label: null, confidence: 0 };
    const wys = frms.map(f => f[0].y);
    const avgY = wys.reduce((a, b) => a + b, 0) / wys.length;
    if (avgY > 0.65) return { label: null, confidence: 0 };
    const r = countReversals(wys, 0.02);
    return r >= 2 ? this._fire("EAT", 0.78, now) : { label: null, confidence: 0 };
  }

  private _come(frms: Pt[][], now: number): MotionDetection {
    if (this._cd("COME", 1200, now) || frms.length < 10) return { label: null, confidence: 0 };
    const wy0 = frms[0][0].y, wy1 = frms[frms.length - 1][0].y;
    return wy1 - wy0 > 0.08 ? this._fire("COME", 0.76, now) : { label: null, confidence: 0 };
  }
}

// ---------------------------------------------------------------------------
// Calibration helpers — persist confidence threshold in localStorage
// ---------------------------------------------------------------------------

const CALIB_KEY = "humanedge_conf_threshold";
const DEFAULT_THRESHOLD = 0.65;

export function loadCalibrationThreshold(): number {
  if (typeof window === "undefined") return DEFAULT_THRESHOLD;
  const stored = localStorage.getItem(CALIB_KEY);
  const val = stored ? parseFloat(stored) : NaN;
  return isNaN(val) ? DEFAULT_THRESHOLD : Math.max(0.3, Math.min(0.95, val));
}

export function saveCalibrationThreshold(t: number): void {
  if (typeof window !== "undefined")
    localStorage.setItem(CALIB_KEY, String(Math.max(0.3, Math.min(0.95, t))));
}

// Built-in task gesture → clean token mapping
const TASK_LABEL_MAP: Record<string, string> = {
  Open_Palm:    "OPEN_PALM",
  Closed_Fist:  "FIST",
  Pointing_Up:  "POINTING",
  Thumb_Up:     "THUMB_UP",
  Thumb_Down:   "THUMB_DOWN",
  Victory:      "V",
  ILoveYou:     "ILY",
};

// ---------------------------------------------------------------------------
// ClientGestureDetector
// ---------------------------------------------------------------------------

export class ClientGestureDetector {
  private recognizer: unknown = null;
  private _state: DetectorState = "idle";
  private _error: string | null = null;
  private _motion = new BrowserMotionClassifier();
  private _motionCooldownUntil = 0; // suppress static detection after a motion gesture fires

  get state(): DetectorState { return this._state; }
  get ready(): boolean       { return this._state === "ready"; }
  get errorMessage(): string | null { return this._error; }

  async initialize(): Promise<void> {
    if (this._state === "ready" || this._state === "loading") return;
    this._state = "loading";
    this._error = null;

    _installWasmLogFilter();

    try {
      const { GestureRecognizer, FilesetResolver } = await import(
        "@mediapipe/tasks-vision"
      );

      const vision = await FilesetResolver.forVisionTasks(WASM_CDN);

      this.recognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });

      this._state = "ready";
    } catch (err) {
      this._state = "error";
      this._error = err instanceof Error ? err.message : String(err);
      throw err;
    }
  }

  /**
   * Run gesture recognition on the current video frame.
   * Priority: motion classifier (≥0.75) → ASL rule classifier → built-in model.
   */
  detect(video: HTMLVideoElement, timestamp: number): GestureResult {
    if (!this.recognizer || this._state !== "ready") {
      this._motion.update(null);
      return { label: null, confidence: 0, handTracked: false };
    }

    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result = (this.recognizer as any).recognizeForVideo(video, timestamp);

      const handTracked: boolean =
        Array.isArray(result.landmarks) && result.landmarks.length > 0;

      if (!handTracked) {
        // Don't wipe window on a single missed frame — motion classifier tolerates brief drops
        this._motion.update(null);
        return { label: null, confidence: 0, handTracked: false };
      }

      // Use first hand landmarks for motion classifier
      const firstHandLms = result.landmarks[0] as Pt[];
      const motionResult = this._motion.update(firstHandLms);

      // Motion sign wins if confident enough
      if (motionResult.label && motionResult.confidence >= 0.75) {
        // Suppress static letter detection for 1.2 s so we don't add spurious letters
        this._motionCooldownUntil = performance.now() + 1200;
        return { label: motionResult.label, confidence: motionResult.confidence, handTracked };
      }

      // Still within motion cooldown — don't emit static letters right after a motion sign
      if (performance.now() < this._motionCooldownUntil) {
        return { label: null, confidence: 0, handTracked };
      }

      // ── ASL rule classifier on all detected hands ──────────────────
      let bestAslLabel: string | null = null;
      let bestAslConf = 0;
      for (let h = 0; h < result.landmarks.length; h++) {
        const rawLms = result.landmarks[h] as Pt[];
        const handedness: string =
          Array.isArray(result.handednesses) && h < result.handednesses.length
            ? (result.handednesses[h][0]?.categoryName ?? "Right")
            : "Right";
        const asl = classifyASL(rawLms, handedness);
        if (asl.label && asl.confidence > bestAslConf) {
          bestAslConf = asl.confidence;
          bestAslLabel = asl.label;
        }
      }

      if (bestAslLabel && bestAslConf >= 0.75)
        return { label: bestAslLabel, confidence: bestAslConf, handTracked };

      // ── Fall back to built-in task gesture ────────────
      if (!Array.isArray(result.gestures) || result.gestures.length === 0) {
        if (bestAslLabel && bestAslConf >= 0.65)
          return { label: bestAslLabel, confidence: bestAslConf, handTracked };
        return { label: null, confidence: 0, handTracked };
      }

      const top = result.gestures[0][0] as { categoryName: string; score: number };
      const builtInLabel = top.categoryName && top.categoryName !== "None"
        ? (TASK_LABEL_MAP[top.categoryName] ?? top.categoryName)
        : null;

      return { label: builtInLabel, confidence: top.score, handTracked };
    } catch {
      this._motion.update(null);
      return { label: null, confidence: 0, handTracked: false };
    }
  }

  /**
   * Sample the confidence for the currently visible hand shape.
   * Used during calibration to measure per-user detection strength.
   */
  sampleConfidence(video: HTMLVideoElement, timestamp: number): number {
    if (!this.recognizer || this._state !== "ready") return 0;
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result = (this.recognizer as any).recognizeForVideo(video, timestamp);
      if (!Array.isArray(result.landmarks) || result.landmarks.length === 0) return 0;
      const lms = result.landmarks[0] as Pt[];
      const handedness = result.handednesses?.[0]?.[0]?.categoryName ?? "Right";
      return classifyASL(lms, handedness).confidence;
    } catch { return 0; }
  }

  resetMotion(): void { this._motion.reset(); this._motionCooldownUntil = 0; }

  close(): void {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (this.recognizer as any)?.close();
    } catch { /* ignore */ }
    this.recognizer = null;
    this._state = "idle";
    this._motion.reset();
  }
}
