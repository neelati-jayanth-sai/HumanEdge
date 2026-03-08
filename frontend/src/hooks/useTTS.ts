"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE } from "../lib/constants";

// Map accent code → gTTS (lang, tld) params
function _accentToGtts(ac: string): { lang: string; tld: string } {
  switch (ac) {
    case "en-GB": return { lang: "en", tld: "co.uk" };
    case "en-IN": return { lang: "en", tld: "co.in" };
    case "hi-IN": return { lang: "hi", tld: "com" };
    default:      return { lang: "en", tld: "com" }; // en-US
  }
}

export function useTTS(speechEnabled: boolean, accent: string) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [speakPending, setSpeakPending] = useState(false);
  const [hasSent, setHasSent] = useState(false);
  const [speakCountdownPct, setSpeakCountdownPct] = useState(0);
  const speakTimerRef = useRef<number | null>(null);
  const lastSpokenRef = useRef("");

  const speak = useCallback(
    async (text: string) => {
      if (!speechEnabled || !text) return;
      // Stop any currently playing audio
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      try {
        const { lang, tld } = _accentToGtts(accent);
        const resp = await fetch(`${API_BASE}/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, lang, tld }),
        });
        if (!resp.ok) throw new Error(`TTS ${resp.status}`);
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audioRef.current = audio;
        audio.onended = () => URL.revokeObjectURL(url);
        await audio.play();
      } catch {
        // Fallback to Web Speech API if backend is unavailable
        if (window.speechSynthesis) {
          const utter = new SpeechSynthesisUtterance(text);
          utter.lang = accent;
          window.speechSynthesis.speak(utter);
        }
      }
    },
    [speechEnabled, accent]
  );

  // Animate the speak countdown progress bar
  useEffect(() => {
    if (!speakPending) { setSpeakCountdownPct(0); return; }
    const DURATION = 2500;
    const start = Date.now();
    setSpeakCountdownPct(100);
    const id = window.setInterval(() => {
      const pct = Math.max(0, Math.round(((DURATION - (Date.now() - start)) / DURATION) * 100));
      setSpeakCountdownPct(pct);
    }, 40);
    return () => clearInterval(id);
  }, [speakPending]);

  const cancelPendingSpeak = useCallback(() => {
    if (speakTimerRef.current !== null) {
      clearTimeout(speakTimerRef.current);
      speakTimerRef.current = null;
    }
    setSpeakPending(false);
  }, []);

  const speakNow = useCallback(
    (text: string) => {
      cancelPendingSpeak();
      lastSpokenRef.current = text;
      setHasSent(true);
      speak(text);
    },
    [speak, cancelPendingSpeak]
  );

  return {
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
    audioRef,
  };
}
