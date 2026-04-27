/** Mirrors `services/shared/schemas_v1.py` for POST /v1/ask. */

export type Source = {
  source_id: string;
  title?: string | null;
  url?: string | null;
  snippet?: string | null;
  score?: number;
  metadata?: Record<string, unknown> | null;
};

/** @deprecated Use `Source`; kept for incremental refactors. */
export type SourceItem = Source;

export type Citation = {
  source_id: string;
  title?: string | null;
  url?: string | null;
};

export type EntityItem = {
  type: string;
  text: string;
  start: number;
  end: number;
  confidence?: number | null;
};

export type FeatureContribution = {
  feature: string;
  contribution: number;
};

export type RiskBlock = {
  score?: number | null;
  label?: "low" | "medium" | "high" | null;
  explanation?: FeatureContribution[];
  risk_available?: boolean;
  confidence?: number | null;
  rationale?: string | null;
};

export type AskDiagnostics = {
  total_request_time_ms?: number | null;
  retrieval_time_ms?: number | null;
  llm_time_ms?: number | null;
  timings?: Record<string, number>;
  retrieval_diagnostics?: Record<string, unknown> | null;
  planner_decisions?: Record<string, unknown> | null;
  fallback_used?: boolean;
  warnings?: string[];
  scoring_diagnostics?: Record<string, unknown> | null;
  trace_storage?: Record<string, unknown> | null;
};

export type AskRequest = {
  mode: "strict" | "hybrid";
  note_text?: string;
  question: string;
  trace_id?: string;
  user_context?: Record<string, unknown>;
};

export type AskResponse = {
  status?: "ok" | "error";
  trace_id?: string;
  pii_redacted?: boolean;
  answer: string;
  sources: Source[];
  citations: Citation[];
  entities?: EntityItem[] | null;
  risk_block?: RiskBlock | null;
  diagnostics?: AskDiagnostics | null;
  warnings?: string[];
  error?: { code: string; message: string; details?: Record<string, unknown> | null };
};
