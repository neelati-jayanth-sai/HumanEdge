"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ClientGestureDetector,
  type DetectorState,
  type GestureResult,
  loadCalibrationThreshold,
  saveCalibrationThreshold,
} from "../lib/gestureClassifier";
import type { DebugPayload, HistoryEntry, ScenarioKey, ServerPayload } from "../lib/types";
import {
  API_BASE,
  FALLBACK_CAPTURE_INTERVAL_MS,
  WS_URL,
} from "../lib/constants";
import { useTTS } from "../hooks/useTTS";
import CalibrationModal from "./CalibrationModal";
import CameraPanel from "./CameraPanel";
import ControlsPanel from "./ControlsPanel";
import DebugSection from "./DebugSection";
import GuideSection from "./GuideSection";
import HistorySection from "./HistorySection";
import SentenceSection from "./SentenceSection";
import SettingsStrip from "./SettingsStrip";
import TokensSection from "./TokensSection";

// Module-level mutable threshold (updated by calibration)
let CLIENT_CONFIDENCE_THRESHOLD = 0.65;

export default function SignStreamClient() {
  // ── Shared state ──────────────────────────────────────────────────────────
  const [running, setRunning] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [language, setLanguage] = useState<"en" | "hi">("en");
  const [accent, setAccent] = useState("en-US");
  const [scenario, setScenario] = useState<ScenarioKey>("general");
  const [customScenario, setCustomScenario] = useState("");
  const [tokens, setTokens] = useState<string[]>([]);
  const [sentence, setSentence] = useState("");
  const [debug, setDebug] = useState<DebugPayload | null>(null);
  const [status, setStatus] = useState("Idle");
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [autoClear, setAutoClear] = useState(true);
  const [guideOpen, setGuideOpen] = useState(false);
  const [debugOpen, setDebugOpen] = useState(false);
  const [tokenInput, setTokenInput] = useState("");
  /** Calibration modal state */
  const [calibOpen, setCalibOpen] = useState(false);
  const [calibPhase, setCalibPhase] = useState<"idle" | "sampling" | "done">("idle");
  const [calibProgress, setCalibProgress] = useState(0);
  const [calibThreshold, setCalibThreshold] = useState(0.65);

  // ── Detector state ────────────────────────────────────────────────────────
  const [detectorState, setDetectorState] = useState<DetectorState>("idle");
  const [localGesture, setLocalGesture] = useState<GestureResult | null>(null);
  const [clientDetectionActive, setClientDetectionActive] = useState(false);

  // ── Refs ──────────────────────────────────────────────────────────────────
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const rafRef = useRef<number | null>(null);
  const fallbackTimerRef = useRef<number | null>(null);
  const autoClearTimerRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const autoClearRef = useRef(autoClear);
  const detectorRef = useRef<ClientGestureDetector | null>(null);
  const clientDetectionActiveRef = useRef(false);
  const historyEndRef = useRef<HTMLDivElement | null>(null);

  // ── TTS hook ──────────────────────────────────────────────────────────────
  const {
    speak,
    cancelPendingSpeak,
    speakNow,
    speakPending,
    setSpeakPending,
    hasSent,
    setHasSent,
    speakCountdownPct,
    speakTimerRef,
    lastSpokenRef,
  } = useTTS(speechEnabled, accent);

  // Load calibration threshold from localStorage on mount
  useEffect(() => {
    const t = loadCalibrationThreshold();
    CLIENT_CONFIDENCE_THRESHOLD = t;
    setCalibThreshold(t);
  }, []);

  // Keep autoClearRef in sync so the WS message handler always sees the latest value
  useEffect(() => { autoClearRef.current = autoClear; }, [autoClear]);

  // Build the config payload sent to the backend on every relevant change
  const buildConfigMsg = useCallback(
    (lang: string, sc: ScenarioKey, custom: string) =>
      JSON.stringify({
        type: "config",
        language: lang,
        scenario: sc,
        custom_scenario: sc === "custom" ? custom : "",
      }),
    []
  );

  // When language changes: auto-select accent + notify backend
  useEffect(() => {
    if (language === "hi") {
      setAccent("hi-IN");
    } else {
      setAccent((prev) => (prev === "hi-IN" ? "en-US" : prev));
    }
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(buildConfigMsg(language, scenario, customScenario));
    }
  }, [language]); // eslint-disable-line react-hooks/exhaustive-deps

  // When scenario changes: notify backend
  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(buildConfigMsg(language, scenario, customScenario));
    }
  }, [scenario]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleCustomScenarioCommit = useCallback(
    (text: string) => {
      setCustomScenario(text);
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(buildConfigMsg(language, "custom", text));
      }
    },
    [language, buildConfigMsg]
  );

  // ---------------------------------------------------------------------------
  // Load persisted history on mount + auto-scroll
  // ---------------------------------------------------------------------------

  useEffect(() => {
    fetch(`${API_BASE}/history`)
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data.history)) setHistory(data.history);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  const clearHistory = useCallback(async () => {
    await fetch(`${API_BASE}/history/clear`, { method: "POST" }).catch(() => {});
    setHistory([]);
  }, []);

  // ---------------------------------------------------------------------------
  // WebSocket message handler
  // ---------------------------------------------------------------------------

  const handleSocketMessage = useCallback(
    (event: MessageEvent<ArrayBuffer>) => {
      try {
        const data = JSON.parse(new TextDecoder().decode(event.data)) as ServerPayload;
        if (data.error) {
          setStatus(`Error: ${data.error}`);
          return;
        }
        if (data.debug) setDebug(data.debug);
        if (Array.isArray(data.tokens)) setTokens(data.tokens);

        // Streaming partial — show words as they arrive, no TTS/history yet
        if (data.sentence_streaming && typeof data.sentence === "string") {
          setSentence(data.sentence);
          return;
        }

        if (typeof data.sentence === "string") setSentence(data.sentence);

        // sentence_finalized=true means the LLM completed a full turn
        if (data.sentence_finalized && data.sentence && Array.isArray(data.tokens)) {
          const finalSentence = data.sentence as string;
          const newTokens = data.tokens as string[];
          const tokenCountAtFinalization = newTokens.length;

          setHistory((prev) => {
            if (prev.length > 0) {
              const last = prev[prev.length - 1];
              const isContinuation =
                newTokens.length >= last.tokens.length &&
                last.tokens.every((t, i) => newTokens[i] === t);
              if (isContinuation)
                return [...prev.slice(0, -1), { tokens: newTokens, sentence: finalSentence }];
            }
            return [...prev, { tokens: newTokens, sentence: finalSentence }];
          });

          // Schedule auto-speak with a 2.5s review window
          cancelPendingSpeak();
          setHasSent(false);
          setSpeakPending(true);
          speakTimerRef.current = window.setTimeout(() => {
            speakTimerRef.current = null;
            setSpeakPending(false);
            setHasSent(true);
            if (finalSentence !== lastSpokenRef.current) {
              lastSpokenRef.current = finalSentence;
              speak(finalSentence);
            }
          }, 2500);

          // Auto-clear — only fires if no new tokens were added during the delay
          if (autoClearTimerRef.current !== null) {
            clearTimeout(autoClearTimerRef.current);
            autoClearTimerRef.current = null;
          }
          if (autoClearRef.current) {
            autoClearTimerRef.current = window.setTimeout(() => {
              autoClearTimerRef.current = null;
              // Abort if the signer already started a new sentence
              setTokens((current) => {
                if (current.length > tokenCountAtFinalization) return current;
                if (wsRef.current?.readyState === WebSocket.OPEN)
                  wsRef.current.send(JSON.stringify({ type: "reset" }));
                setSentence("");
                setHasSent(false);
                lastSpokenRef.current = "";
                return [];
              });
            }, 4000);
          }
        }
      } catch {
        /* ignore parse errors */
      }
    },
    [speak, cancelPendingSpeak, setSpeakPending, setHasSent, speakTimerRef, lastSpokenRef]
  );

  // ---------------------------------------------------------------------------
  // Cleanup helpers
  // ---------------------------------------------------------------------------

  const stopDetectionLoop = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (fallbackTimerRef.current !== null) {
      clearInterval(fallbackTimerRef.current);
      fallbackTimerRef.current = null;
    }
  }, []);

  const stopStreaming = useCallback(() => {
    stopDetectionLoop();
    cancelPendingSpeak();
    if (autoClearTimerRef.current !== null) {
      clearTimeout(autoClearTimerRef.current);
      autoClearTimerRef.current = null;
    }
    wsRef.current?.close();
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    clientDetectionActiveRef.current = false;
    setClientDetectionActive(false);
    setLocalGesture(null);
    setRunning(false);
    setStatus("Stopped");
  }, [stopDetectionLoop, cancelPendingSpeak]);

  // ---------------------------------------------------------------------------
  // Client-side detection loop (requestAnimationFrame)
  // ---------------------------------------------------------------------------

  const startClientDetectionLoop = useCallback((ws: WebSocket) => {
    const detector = detectorRef.current;
    if (!detector?.ready) return;

    clientDetectionActiveRef.current = true;
    setClientDetectionActive(true);

    const loop = (timestamp: number) => {
      const video = videoRef.current;
      if (!video || ws.readyState !== WebSocket.OPEN) return;

      const result = detector.detect(video, timestamp);
      setLocalGesture(result);

      if (ws.readyState === WebSocket.OPEN) {
        const label =
          result.label && result.confidence >= CLIENT_CONFIDENCE_THRESHOLD
            ? result.label
            : null;
        ws.send(
          JSON.stringify({
            type: "gesture",
            label,
            confidence: result.confidence,
          })
        );
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
  }, []);

  // ---------------------------------------------------------------------------
  // Image-streaming fallback loop (setInterval)
  // ---------------------------------------------------------------------------

  const startFallbackStreamingLoop = useCallback((ws: WebSocket) => {
    clientDetectionActiveRef.current = false;
    setStatus("Connected (server detection)");

    fallbackTimerRef.current = window.setInterval(() => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video || ws.readyState !== WebSocket.OPEN) return;

      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (!blob || ws.readyState !== WebSocket.OPEN) return;
          blob.arrayBuffer().then((buf) => ws.send(buf));
        },
        "image/jpeg",
        0.82
      );
    }, FALLBACK_CAPTURE_INTERVAL_MS);
  }, []);

  // ---------------------------------------------------------------------------
  // Start streaming
  // ---------------------------------------------------------------------------

  const startStreaming = useCallback(async () => {
    if (running) return;
    setStatus("Starting camera…");

    // 1. Get camera
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    streamRef.current = mediaStream;

    const video = videoRef.current;
    if (!video) return;
    video.srcObject = mediaStream;
    await video.play();

    // 2. Connect WebSocket
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      setRunning(true);
      // Sync current language + scenario to backend
      ws.send(buildConfigMsg(language, scenario, customScenario));

      const detector = detectorRef.current;
      if (detector?.ready) {
        setStatus("Connected (browser detection)");
        startClientDetectionLoop(ws);
      } else {
        startFallbackStreamingLoop(ws);
      }
    };

    ws.onmessage = handleSocketMessage;
    ws.onerror = () => setStatus("WebSocket error");
    ws.onclose = () => {
      stopDetectionLoop();
      setRunning(false);
      setStatus("Disconnected");
    };
  }, [
    running,
    language,
    scenario,
    customScenario,
    buildConfigMsg,
    startClientDetectionLoop,
    startFallbackStreamingLoop,
    handleSocketMessage,
    stopDetectionLoop,
  ]);

  // ---------------------------------------------------------------------------
  // Reset
  // ---------------------------------------------------------------------------

  const resetSession = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "reset" }));
    }
    cancelPendingSpeak();
    if (autoClearTimerRef.current !== null) {
      clearTimeout(autoClearTimerRef.current);
      autoClearTimerRef.current = null;
    }
    setTokens([]);
    setSentence("");
    setDebug(null);
    setLocalGesture(null);
    setHasSent(false);
    lastSpokenRef.current = "";
    detectorRef.current?.resetMotion();
    setStatus(running ? (clientDetectionActiveRef.current ? "Connected (browser detection)" : "Connected (server detection)") : "Reset");
  }, [running, cancelPendingSpeak, setHasSent, lastSpokenRef]);

  // ---------------------------------------------------------------------------
  // Initialize client-side detector once on mount
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const det = new ClientGestureDetector();
    detectorRef.current = det;
    setDetectorState("loading");

    det
      .initialize()
      .then(() => {
        setDetectorState("ready");
        // If streaming already started before the model was ready, switch to
        // client detection now.
        if (wsRef.current?.readyState === WebSocket.OPEN && !clientDetectionActiveRef.current) {
          stopDetectionLoop();
          startClientDetectionLoop(wsRef.current);
          setStatus("Connected (browser detection)");
        }
      })
      .catch(() => {
        setDetectorState("error");
        // Fall back gracefully — server-side image streaming still works.
      });

    return () => {
      det.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => () => stopStreaming(), [stopStreaming]);

  // ---------------------------------------------------------------------------
  // Derived display values
  // ---------------------------------------------------------------------------

  const handTracked =
    clientDetectionActive && localGesture !== null
      ? localGesture.handTracked
      : (debug?.hand_tracked ?? false);

  const currentGestureLabel =
    clientDetectionActive && localGesture !== null
      ? localGesture.label
      : debug?.accepted_label ?? null;

  const currentConfidence =
    clientDetectionActive && localGesture !== null
      ? localGesture.confidence
      : debug?.confidence ?? 0;

  // ---------------------------------------------------------------------------
  // Token correction helpers
  // ---------------------------------------------------------------------------

  const sendTokenUpdate = useCallback((newTokens: string[]) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "set_tokens", tokens: newTokens }));
    }
    setTokens(newTokens);
    setSentence("");
    lastSpokenRef.current = "";
  }, [lastSpokenRef]);

  const handleTokenDelete = useCallback((idx: number) => {
    setTokens((prev) => {
      const next = prev.filter((_, i) => i !== idx);
      sendTokenUpdate(next);
      return next;
    });
  }, [sendTokenUpdate]);

  const handleManualAdd = useCallback(() => {
    const val = tokenInput.trim().toUpperCase();
    if (!val) return;
    setTokens((prev) => {
      const next = [...prev, val];
      sendTokenUpdate(next);
      return next;
    });
    setTokenInput("");
  }, [tokenInput, sendTokenUpdate]);

  const handleRegenerate = useCallback(() => {
    setTokens((prev) => {
      sendTokenUpdate([...prev]);
      return prev;
    });
  }, [sendTokenUpdate]);

  // ---------------------------------------------------------------------------
  // Calibration
  // ---------------------------------------------------------------------------

  const runCalibration = useCallback(() => {
    const detector = detectorRef.current;
    const video = videoRef.current;
    if (!detector?.ready || !video) return;

    setCalibPhase("sampling");
    setCalibProgress(0);

    const SAMPLES = 60;
    const confidences: number[] = [];
    let frame = 0;

    const tick = (ts: number) => {
      const conf = detector.sampleConfidence(video, ts);
      if (conf > 0) confidences.push(conf);
      frame++;
      setCalibProgress(Math.round((frame / SAMPLES) * 100));

      if (frame < SAMPLES) {
        requestAnimationFrame(tick);
      } else {
        if (confidences.length < 10) {
          setCalibPhase("done");
          return;
        }
        confidences.sort((a, b) => a - b);
        const p20 = confidences[Math.floor(confidences.length * 0.20)];
        const newThreshold = Math.max(0.35, Math.min(0.88, p20 * 0.85));
        CLIENT_CONFIDENCE_THRESHOLD = newThreshold;
        saveCalibrationThreshold(newThreshold);
        setCalibThreshold(newThreshold);
        setCalibPhase("done");
      }
    };
    requestAnimationFrame(tick);
  }, []);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="app">
      {/* Background effects */}
      <div className="bg-grid" aria-hidden />
      <div className="bg-orb-1" aria-hidden />
      <div className="bg-orb-2" aria-hidden />

      {/* ── Header ── */}
      <header className="app-header">
        <div className="logo">
          <span className="logo-mark">◈</span>
          <span className="logo-name">Human<span> Edge</span></span>
        </div>
        <div className="header-divider" />
        <div className="status-pill">
          <span className={`status-dot${running ? " live" : ""}`} />
          {status}
        </div>
        <div className="header-right">
          {detectorState === "loading" && (
            <div className="detector-badge loading">Loading model…</div>
          )}
          {detectorState === "error" && (
            <div className="detector-badge error">Server detection</div>
          )}
          {detectorState === "ready" && (
            <button
              type="button"
              className="calib-btn"
              onClick={() => { setCalibOpen(true); setCalibPhase("idle"); }}
              title="Calibrate detection to your hand"
            >
              ◎ Calibrate
            </button>
          )}
        </div>
      </header>

      {/* ── Body ── */}
      <div className="app-body">

        {/* ════ Left column — camera ════ */}
        <div className="cam-col">
          <CameraPanel
            running={running}
            handTracked={handTracked}
            currentGestureLabel={currentGestureLabel}
            currentConfidence={currentConfidence}
            clientDetectionActive={clientDetectionActive}
            debug={debug}
            videoRef={videoRef}
            canvasRef={canvasRef}
          />
          <ControlsPanel
            running={running}
            speechEnabled={speechEnabled}
            autoClear={autoClear}
            guideOpen={guideOpen}
            onStart={startStreaming}
            onStop={stopStreaming}
            onReset={resetSession}
            onToggleSpeech={() => setSpeechEnabled((v) => !v)}
            onToggleAutoClear={() => setAutoClear((v) => !v)}
            onToggleGuide={() => setGuideOpen((v) => !v)}
            onClearHistory={clearHistory}
          />
          <SettingsStrip
            language={language}
            accent={accent}
            scenario={scenario}
            customScenario={customScenario}
            onLanguageChange={setLanguage}
            onAccentChange={setAccent}
            onScenarioChange={setScenario}
            onCustomScenarioCommit={handleCustomScenarioCommit}
          />
        </div>

        {/* ════ Right column — output ════ */}
        <div className="out-col">
          <TokensSection
            tokens={tokens}
            tokenInput={tokenInput}
            onInputChange={setTokenInput}
            onAdd={handleManualAdd}
            onDelete={handleTokenDelete}
            onRegenerate={handleRegenerate}
          />
          <SentenceSection
            sentence={sentence}
            speakPending={speakPending}
            hasSent={hasSent}
            speakCountdownPct={speakCountdownPct}
            onSpeakNow={() => speakNow(sentence)}
            onCancel={cancelPendingSpeak}
            onNewSentence={resetSession}
          />
          {guideOpen && <GuideSection />}
          {history.length > 0 && (
            <HistorySection history={history} historyEndRef={historyEndRef} />
          )}
          <DebugSection
            debug={debug}
            open={debugOpen}
            onToggle={() => setDebugOpen((v) => !v)}
            clientDetectionActive={clientDetectionActive}
            handTracked={handTracked}
            currentGestureLabel={currentGestureLabel}
            currentConfidence={currentConfidence}
          />
        </div>
      </div>

      <CalibrationModal
        open={calibOpen}
        phase={calibPhase}
        progress={calibProgress}
        threshold={calibThreshold}
        running={running}
        onClose={() => setCalibOpen(false)}
        onStart={runCalibration}
      />
    </div>
  );
}
