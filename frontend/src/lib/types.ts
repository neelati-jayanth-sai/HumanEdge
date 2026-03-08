export type DebugPayload = {
  backend: string;
  hand_tracked: boolean;
  raw_label?: string | null;
  accepted_label?: string | null;
  confidence: number;
  token_updated: boolean;
  pending_label?: string | null;
  pending_elapsed_ms: number;
  stable_window_ms: number;
};

export type ServerPayload = {
  tokens?: string[];
  sentence?: string;
  sentence_finalized?: boolean;
  sentence_streaming?: boolean;
  debug?: DebugPayload;
  error?: string;
};

export type HistoryEntry = { tokens: string[]; sentence: string };

export type ScenarioKey =
  | "general"
  | "interview"
  | "banking"
  | "medical"
  | "restaurant"
  | "custom";
