export type SourceItem = {
  id?: string;
  title: string;
  snippet?: string;
  url?: string;
  score?: number;
};

export type EntityItem = {
  text: string;
  label: string;
  confidence?: number;
};

export type RiskAssessment = {
  level?: "low" | "medium" | "high" | "unknown";
  rationale?: string;
  recommendations?: string[];
};

export type AskRequest = {
  mode: "strict" | "hybrid";
  note_text?: string;
  question: string;
};

export type AskResponse = {
  answer: string;
  sources: SourceItem[];
  entities: EntityItem[];
  risk?: RiskAssessment;
  trace_id?: string;
  warnings?: string[];
  total_request_time_ms?: number;
  retrieval_time_ms?: number;
  llm_time_ms?: number;
  timings?: {
    total_request_time_ms?: number;
    retrieval_time_ms?: number;
    llm_time_ms?: number;
  };
  retrieval_diagnostics?: unknown;
  planner_decisions?: unknown;
  planner?: unknown;
};
