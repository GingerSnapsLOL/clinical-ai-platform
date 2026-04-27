"use client";
import { useState } from "react";
import { AnswerPanel } from "@/components/answer-panel";
import { AskForm } from "@/components/ask-form";
import { EntitiesPanel } from "@/components/entities-panel";
import { RiskPanel } from "@/components/risk-panel";
import { SourcesList } from "@/components/sources-list";
import { DashboardCard } from "@/components/dashboard-card";
import { askClinicalQuestion } from "@/lib/api";
import type { AskRequest, AskResponse } from "@/lib/types";

type AskStatus = "idle" | "loading" | "success" | "error";

function formatMs(value?: number): string {
  return typeof value === "number" ? `${value.toFixed(0)} ms` : "N/A";
}

function getTiming(result: AskResponse | null, key: "total" | "retrieval" | "llm") {
  if (!result) return undefined;
  const d = result.diagnostics;
  const t = d?.timings;
  if (key === "total") {
    return d?.total_request_time_ms ?? t?.total_request_time_ms;
  }
  if (key === "retrieval") {
    return d?.retrieval_time_ms ?? t?.retrieval_service_duration_ms;
  }
  return d?.llm_time_ms ?? t?.llm_service_duration_ms;
}

function getSourceStats(result: AskResponse | null) {
  const sources = result?.sources ?? [];
  const scores = sources
    .map((source) => source.score)
    .filter((score): score is number => typeof score === "number");
  const avgScore = scores.length
    ? scores.reduce((sum, score) => sum + score, 0) / scores.length
    : undefined;

  return { count: sources.length, avgScore };
}

function getGroundingQuality(result: AskResponse | null): {
  label: "strong grounding" | "weak grounding" | "insufficient data";
  toneClass: string;
  note: string;
  confidenceLabel: "High" | "Medium" | "Low";
  groundingScore: number;
} | undefined {
  if (!result) return undefined;

  const warnings = result.warnings ?? [];
  const scores = (result.sources ?? [])
    .map((source) => source.score)
    .filter((score): score is number => typeof score === "number");

  const normalize = (value: number): number => {
    const v = value > 1 ? value / 10 : value;
    return Math.max(0, Math.min(1, v));
  };

  const confidenceBucket = (
    score: number
  ): { confidenceLabel: "High" | "Medium" | "Low"; toneClass: string } => {
    if (score > 0.7) return { confidenceLabel: "High", toneClass: "bg-green-100 text-green-700" };
    if (score >= 0.4) return { confidenceLabel: "Medium", toneClass: "bg-amber-100 text-amber-700" };
    return { confidenceLabel: "Low", toneClass: "bg-red-100 text-red-700" };
  };

  if (scores.length === 0) {
    const groundingScore = 0;
    const conf = confidenceBucket(groundingScore);
    return {
      label: "insufficient data",
      toneClass: conf.toneClass,
      note: "No retrieval scores available to assess grounding.",
      confidenceLabel: conf.confidenceLabel,
      groundingScore,
    };
  }

  const avgRawScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
  const avgScore = normalize(avgRawScore);
  const hasWarnings = warnings.length > 0;
  const conf = confidenceBucket(avgScore);

  if (avgScore >= 0.75 && !hasWarnings) {
    return {
      label: "strong grounding",
      toneClass: "bg-green-100 text-green-700",
      note: "High retrieval relevance with no warning signals.",
      confidenceLabel: conf.confidenceLabel,
      groundingScore: avgScore,
    };
  }

  if (avgScore < 0.45 || hasWarnings) {
    return {
      label: avgScore < 0.35 ? "insufficient data" : "weak grounding",
      toneClass: avgScore < 0.35 ? "bg-red-100 text-red-700" : "bg-amber-100 text-amber-700",
      note: hasWarnings
        ? "Warning signals present; grounding confidence is reduced."
        : "Retrieval relevance is below target confidence.",
      confidenceLabel: conf.confidenceLabel,
      groundingScore: avgScore,
    };
  }

  return {
    label: "weak grounding",
    toneClass: "bg-amber-100 text-amber-700",
    note: "Moderate retrieval relevance; review evidence before use.",
    confidenceLabel: conf.confidenceLabel,
    groundingScore: avgScore,
  };
}

export default function AskPage() {
  const [result, setResult] = useState<AskResponse | null>(null);
  const [compareResult, setCompareResult] = useState<AskResponse | null>(null);
  const [status, setStatus] = useState<AskStatus>("idle");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [devMode, setDevMode] = useState(false);

  async function handleAsk(data: AskRequest) {
    try {
      setIsSubmitting(true);
      setStatus("loading");
      setError(null);
      setCompareError(null);

      const reqWithMode: AskRequest = devMode
        ? {
            ...data,
            user_context: { ...(data.user_context ?? {}), debug: true },
          }
        : data;

      if (compareMode) {
        const [first, second] = await Promise.allSettled([
          askClinicalQuestion(reqWithMode),
          askClinicalQuestion(reqWithMode),
        ]);

        if (first.status === "fulfilled") {
          setResult(first.value);
        } else {
          setResult(null);
          setError(
            first.reason instanceof Error
              ? first.reason.message
              : "Primary compare run failed."
          );
        }

        if (second.status === "fulfilled") {
          setCompareResult(second.value);
        } else {
          setCompareResult(null);
          setCompareError(
            second.reason instanceof Error
              ? second.reason.message
              : "Secondary compare run failed."
          );
        }

        if (first.status === "rejected" && second.status === "rejected") {
          setStatus("error");
        } else {
          setStatus("success");
        }
        return;
      }

      const response = await askClinicalQuestion(reqWithMode);
      setResult(response);
      setCompareResult(null);
      setStatus("success");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to get response.");
      setStatus("error");
    } finally {
      setIsSubmitting(false);
    }
  }

  const totalRequestTime = getTiming(result, "total");
  const retrievalTime = getTiming(result, "retrieval");
  const llmTime = getTiming(result, "llm");
  const grounding = getGroundingQuality(result);
  const compareGrounding = getGroundingQuality(compareResult);
  const compareTotalRequestTime = getTiming(compareResult, "total");
  const compareRetrievalTime = getTiming(compareResult, "retrieval");
  const compareLlmTime = getTiming(compareResult, "llm");
  const sourceA = getSourceStats(result);
  const sourceB = getSourceStats(compareResult);
  const retrievalDiagnostics = result?.diagnostics?.retrieval_diagnostics;
  const plannerDecisions = result?.diagnostics?.planner_decisions;

  const hasTimings =
    typeof totalRequestTime === "number" ||
    typeof retrievalTime === "number" ||
    typeof llmTime === "number";
  const hasCompareTimings =
    typeof compareTotalRequestTime === "number" ||
    typeof compareRetrievalTime === "number" ||
    typeof compareLlmTime === "number";
  const fallbackUsed =
    Boolean(result?.diagnostics?.fallback_used) ||
    Boolean((result?.warnings ?? []).some((w) => w.includes("fallback_used")));
  const compareFallbackUsed =
    Boolean(compareResult?.diagnostics?.fallback_used) ||
    Boolean((compareResult?.warnings ?? []).some((w) => w.includes("fallback_used")));

  return (
    <main className="mx-auto flex w-full max-w-[1280px] flex-col gap-8 px-4 py-8 lg:px-8 lg:py-10">
      <header className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Clinical Assistant</p>
        <h1 className="text-3xl font-semibold tracking-tight text-slate-800 lg:text-4xl">Clinical Guidance Workspace</h1>
        <p className="max-w-3xl text-sm leading-6 text-slate-600">
          Ask a clinical question and review evidence-backed guidance with clear risk and source context.
        </p>
      </header>
      <div className="grid gap-8 lg:grid-cols-12 lg:items-start">
        <section className="lg:col-span-5 xl:col-span-4">
          <div className="rounded-2xl border border-slate-200/80 bg-white/80 p-4 lg:sticky lg:top-6">
            <h2 className="mb-1 text-base font-semibold text-slate-800">Request Input</h2>
            <p className="mb-4 text-xs text-slate-600">Required: mode, note_text, question</p>
            <label className="mb-4 flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={compareMode}
                onChange={(event) => setCompareMode(event.target.checked)}
              />
              Compare mode (run same request twice)
            </label>
            <label className="mb-4 flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={devMode}
                onChange={(event) => setDevMode(event.target.checked)}
              />
              Dev mode (show diagnostics and raw metadata)
            </label>
            <AskForm onSubmit={handleAsk} isSubmitting={isSubmitting} />
          </div>
        </section>
        <section className="space-y-6 lg:col-span-7 xl:col-span-8">
          {compareMode ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="rounded-2xl border border-slate-200 bg-white p-1 shadow-sm">
                <AnswerPanel
                  title="Answer A"
                  answer={result?.answer}
                  isLoading={isSubmitting}
                  error={error}
                  grounding={grounding}
                  showWeakGroundingWarning={grounding?.label === "weak grounding"}
                  fallbackUsed={fallbackUsed}
                />
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-1 shadow-sm">
                <AnswerPanel
                  title="Answer B"
                  answer={compareResult?.answer}
                  isLoading={isSubmitting}
                  error={compareError}
                  grounding={compareGrounding}
                  showWeakGroundingWarning={compareGrounding?.label === "weak grounding"}
                  fallbackUsed={compareFallbackUsed}
                />
              </div>
            </div>
          ) : (
            <div className="rounded-2xl border border-slate-200 bg-white p-1 shadow-sm">
              <AnswerPanel
                answer={result?.answer}
                isLoading={isSubmitting}
                error={error}
                grounding={grounding}
                showWeakGroundingWarning={grounding?.label === "weak grounding"}
                fallbackUsed={fallbackUsed}
              />
            </div>
          )}
          <SourcesList sources={result?.sources ?? []} devMode={devMode} />
          <RiskPanel riskBlock={result?.risk_block} devMode={devMode} />
          {devMode && <EntitiesPanel entities={result?.entities ?? []} />}
          {devMode && <DashboardCard title="Diagnostics">
            <details className="group">
              <summary className="cursor-pointer list-none text-sm font-medium text-slate-800">
                <span className="inline-flex items-center gap-2">
                  Expand diagnostics
                  <span className="text-xs text-slate-500 group-open:hidden">(collapsed)</span>
                  <span className="hidden text-xs text-slate-500 group-open:inline">(expanded)</span>
                </span>
              </summary>
              <div className="mt-3">
                {!hasTimings ? (
                  <p className="text-sm text-slate-600">
                    Latency timings are not provided by the backend yet.
                  </p>
                ) : (
                  <div className="grid gap-2 sm:grid-cols-3">
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                        Total request
                      </p>
                      <p className="mt-1 text-sm font-semibold text-slate-900">
                        {formatMs(totalRequestTime)}
                      </p>
                    </div>
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                        Retrieval
                      </p>
                      <p className="mt-1 text-sm font-semibold text-slate-900">
                        {formatMs(retrievalTime)}
                      </p>
                    </div>
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                        LLM
                      </p>
                      <p className="mt-1 text-sm font-semibold text-slate-900">
                        {formatMs(llmTime)}
                      </p>
                    </div>
                  </div>
                )}

                {compareMode && (
                  <div className="mt-3 grid gap-2 sm:grid-cols-2">
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
                      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                        Latency differences (B - A)
                      </p>
                      {!hasCompareTimings ? (
                        <p className="mt-1">No timing metrics returned for compare.</p>
                      ) : (
                        <>
                          <p className="mt-1">
                            Total:{" "}
                            {typeof compareTotalRequestTime === "number" &&
                            typeof totalRequestTime === "number"
                              ? `${(compareTotalRequestTime - totalRequestTime).toFixed(0)} ms`
                              : "N/A"}
                          </p>
                          <p>
                            Retrieval:{" "}
                            {typeof compareRetrievalTime === "number" &&
                            typeof retrievalTime === "number"
                              ? `${(compareRetrievalTime - retrievalTime).toFixed(0)} ms`
                              : "N/A"}
                          </p>
                          <p>
                            LLM:{" "}
                            {typeof compareLlmTime === "number" &&
                            typeof llmTime === "number"
                              ? `${(compareLlmTime - llmTime).toFixed(0)} ms`
                              : "N/A"}
                          </p>
                        </>
                      )}
                    </div>
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
                      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                        Source differences
                      </p>
                      <p className="mt-1">Count A/B: {sourceA.count} / {sourceB.count}</p>
                      <p>
                        Count delta: {(sourceB.count - sourceA.count).toString()}
                      </p>
                      <p>
                        Avg score A/B:{" "}
                        {typeof sourceA.avgScore === "number"
                          ? sourceA.avgScore.toFixed(2)
                          : "N/A"}{" "}
                        /{" "}
                        {typeof sourceB.avgScore === "number"
                          ? sourceB.avgScore.toFixed(2)
                          : "N/A"}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </details>
          </DashboardCard>}
          {devMode && <DashboardCard title="Trace / Debug">
            <details className="group">
              <summary className="cursor-pointer list-none text-sm font-medium text-slate-800">
                <span className="inline-flex items-center gap-2">
                  Expand debug details
                  <span className="text-xs text-slate-500 group-open:hidden">(collapsed)</span>
                  <span className="hidden text-xs text-slate-500 group-open:inline">(expanded)</span>
                </span>
              </summary>
              <div className="mt-3 space-y-3">
                <div>
                  <p className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">
                    Trace ID
                  </p>
                  <pre className="meta-block">{result?.trace_id ?? "not_available"}</pre>
                </div>

                <div>
                  <p className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">
                    Warnings
                  </p>
                  {result?.warnings && result.warnings.length > 0 ? (
                    <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">
                      {result.warnings.map((warning, index) => (
                        <li key={`${warning}-${index}`}>{warning}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-slate-600">No warnings.</p>
                  )}
                </div>

                <div>
                  <p className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">
                    Retrieval diagnostics
                  </p>
                  <pre className="meta-block">
                    {retrievalDiagnostics
                      ? JSON.stringify(retrievalDiagnostics, null, 2)
                      : "not_available"}
                  </pre>
                </div>

                <div>
                  <p className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">
                    Planner decisions
                  </p>
                  <pre className="meta-block">
                    {plannerDecisions
                      ? JSON.stringify(plannerDecisions, null, 2)
                      : "not_available"}
                  </pre>
                </div>
              </div>
            </details>
          </DashboardCard>}
        </section>
      </div>
    </main>
  );
}
