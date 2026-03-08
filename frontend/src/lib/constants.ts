export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";

/** Loaded from localStorage; updated by calibration flow. */
export let CLIENT_CONFIDENCE_THRESHOLD = 0.65; // set properly on mount via loadCalibrationThreshold()

/**
 * When falling back to image streaming, capture at this interval.
 * The backend rate-limits to MAX_FPS=15 anyway.
 */
export const FALLBACK_CAPTURE_INTERVAL_MS = 66;

// All recognized tokens — drives datalist autocomplete in the correction input
export const ALL_GESTURE_LABELS = [
  // Static signs
  "OPEN_PALM","THUMB_UP","THUMB_DOWN","V","ILY",
  "A","B","C","D","E","F","G","I","K","L","O","R","S","U","W","X","Y",
  // Dynamic signs (motion classifier)
  "WAVE","THANK_YOU","PLEASE","SORRY","YES_NOD","MORE","EAT","COME",
];

export const SCENARIOS = [
  { key: "general",    label: "General",    icon: "💬", hint: "Everyday conversation" },
  { key: "interview",  label: "Interview",  icon: "🤝", hint: "Job interview" },
  { key: "banking",    label: "Bank",       icon: "🏦", hint: "Banking & finance" },
  { key: "medical",    label: "Medical",    icon: "🏥", hint: "Doctor / healthcare" },
  { key: "restaurant", label: "Restaurant", icon: "🍽️", hint: "Ordering food" },
  { key: "custom",     label: "Custom",     icon: "✏️", hint: "Describe your own setting" },
] as const;

export const GESTURE_GUIDE_CATEGORIES = [
  {
    category: "Common Signs",
    signs: [
      { label: "OPEN_PALM",  emoji: "🖐️", how: "Spread all 5 fingers, face palm to camera",      tip: "Hello / Hi" },
      { label: "THUMB_UP",   emoji: "👍",  how: "Fist with thumb pointing straight up",           tip: "Yes / Good" },
      { label: "THUMB_DOWN", emoji: "👎",  how: "Fist with thumb pointing straight down",         tip: "No / Bad" },
      { label: "V",          emoji: "✌️",  how: "Index + middle up, spread apart",                tip: "Peace / V" },
      { label: "ILY",        emoji: "🤟",  how: "Thumb + index + pinky out, fold middle + ring",  tip: "I Love You" },
    ],
  },
  {
    category: "ASL Letters A – I",
    signs: [
      { label: "A", emoji: "✊", how: "Fist, thumb resting alongside index (not over fingers)",   tip: "Letter A" },
      { label: "B", emoji: "🖐", how: "4 fingers straight up together, thumb tucked across palm", tip: "Letter B" },
      { label: "C", emoji: "🤏", how: "Fingers curved into a C arc, thumb open",                  tip: "Letter C" },
      { label: "D", emoji: "☝️", how: "Index up, thumb + others arch to form a D",               tip: "Letter D" },
      { label: "E", emoji: "✊", how: "All fingertips bent deeply toward palm, thumb tucked",     tip: "Letter E" },
      { label: "F", emoji: "👌", how: "Thumb-index pinch, middle + ring + pinky extended",        tip: "Letter F" },
      { label: "G", emoji: "👉", how: "Index pointing sideways, thumb out (gun shape)",           tip: "Letter G" },
      { label: "I", emoji: "🤙", how: "Only pinky finger up, other four curled",                  tip: "Letter I" },
    ],
  },
  {
    category: "ASL Letters K – Y",
    signs: [
      { label: "K", emoji: "✌️", how: "Index + middle up, thumb pointing up between them",        tip: "Letter K" },
      { label: "L", emoji: "👆", how: "Index pointing up + thumb to the side — L shape",          tip: "Letter L" },
      { label: "O", emoji: "👌", how: "All fingertips pinched together forming a round O",         tip: "Letter O" },
      { label: "R", emoji: "🤞", how: "Index + middle up, index crossed over middle",              tip: "Letter R" },
      { label: "S", emoji: "✊", how: "Fist, thumb wrapped over the front of fingers",             tip: "Letter S" },
      { label: "U", emoji: "🤞", how: "Index + middle up, held close together",                   tip: "Letter U" },
      { label: "W", emoji: "🖖", how: "Index + middle + ring spread upward (3 fingers)",           tip: "W / Water" },
      { label: "X", emoji: "☝️", how: "Index finger hooked/crooked, other fingers curled",        tip: "Letter X" },
      { label: "Y", emoji: "🤙", how: "Pinky + thumb out, index + middle + ring curled",          tip: "Letter Y" },
    ],
  },
];
