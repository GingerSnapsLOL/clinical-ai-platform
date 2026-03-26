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
  if (key === "total") {
    return result.total_request_time_ms ?? result.timings?.total_request_time_ms;
  }
  if (key === "retrieval") {
    return result.retrieval_time_ms ?? result.timings?.retrieval_time_ms;
  }
  return result.llm_time_ms ?? result.timings?.llm_time_ms;
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
} | null {
  if (!result) return null;

  const warnings = result.warnings ?? [];
  const scores = (result.sources ?? [])
    .map((source) => source.score)
    .filter((score): score is number => typeof score === "number");

  if (scores.length === 0) {
    return {
      label: "insufficient data",
      toneClass: "bg-red-100 text-red-700",
      note: "No retrieval scores available to assess grounding.",
    };
  }

  const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
  const hasWarnings = warnings.length > 0;

  if (avgScore >= 0.75 && !hasWarnings) {
    return {
      label: "strong grounding",
      toneClass: "bg-green-100 text-green-700",
      note: "High retrieval relevance with no warning signals.",
    };
  }

  if (avgScore < 0.45 || hasWarnings) {
    return {
      label: avgScore < 0.35 ? "insufficient data" : "weak grounding",
      toneClass: avgScore < 0.35 ? "bg-red-100 text-red-700" : "bg-amber-100 text-amber-700",
      note: hasWarnings
        ? "Warning signals present; grounding confidence is reduced."
        : "Retrieval relevance is below target confidence.",
    };
  }

  return {
    label: "weak grounding",
    toneClass: "bg-amber-100 text-amber-700",
    note: "Moderate retrieval relevance; review evidence before use.",
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

  async function handleAsk(data: AskRequest) {
    try {
      setIsSubmitting(true);
      setStatus("loading");
      setError(null);
      setCompareError(null);

      if (compareMode) {
        const [first, second] = await Promise.allSettled([
          askClinicalQuestion(data),
          askClinicalQuestion(data),
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

      const response = await askClinicalQuestion(data);
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
  const retrievalDiagnostics = result?.retrieval_diagnostics;
  const plannerDecisions = result?.planner_decisions ?? result?.planner;

  return (
    <main className="mx-auto flex w-full max-w-[1280px] flex-col gap-8 px-4 py-8 lg:px-8 lg:py-10">
      <header className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Clinical AI Dashboard</p>
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900 lg:text-4xl">Ask Endpoint Testing Console</h1>
        <p className="max-w-3xl text-sm leading-6 text-slate-600">
          Internal console for validating answer quality, evidence coverage, extracted entities, and risk assessment output.
        </p>
      </header>
      <div className="grid gap-6 lg:grid-cols-12 lg:items-start">
        <section className="lg:col-span-5 xl:col-span-4">
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 lg:sticky lg:top-6">
            <h2 className="mb-1 text-base font-semibold text-slate-900">Request Input</h2>
            <p className="mb-4 text-xs text-slate-600">Required: mode, note_text, question</p>
            <label className="mb-4 flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={compareMode}
                onChange={(event) => setCompareMode(event.target.checked)}
              />
              Compare mode (run same request twice)
            </label>
            <AskForm onSubmit={handleAsk} isSubmitting={isSubmitting} />
          </div>
        </section>
        <section className="space-y-4 lg:col-span-7 xl:col-span-8">
          {compareMode ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <AnswerPanel
                title="Answer A"
                answer={result?.answer}
                isLoading={isSubmitting}
                error={error}
                grounding={grounding}
              />
              <AnswerPanel
                title="Answer B"
                answer={compareResult?.answer}
                isLoading={isSubmitting}
                error={compareError}
                grounding={compareGrounding}
              />
            </div>
          ) : (
            <AnswerPanel
              answer={result?.answer}
              isLoading={isSubmitting}
              error={error}
              grounding={grounding}
            />
          )}
          <SourcesList sources={result?.sources ?? []} />
          <EntitiesPanel entities={result?.entities ?? []} />
          <RiskPanel risk={result?.risk} />
          <DashboardCard title="Diagnostics">
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
            {compareMode && (
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                    Latency differences (B - A)
                  </p>
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
          </DashboardCard>
          <DashboardCard title="Trace / Debug">
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
          </DashboardCard>
        </section>
      </div>
    </main>
  );
}
