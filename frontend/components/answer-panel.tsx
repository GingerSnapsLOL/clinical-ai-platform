import { DashboardCard } from "@/components/dashboard-card";

type GroundingIndicator = {
  label: "strong grounding" | "weak grounding" | "insufficient data";
  toneClass: string;
  note: string;
};

type Props = {
  title?: string;
  answer?: string;
  isLoading?: boolean;
  error?: string | null;
  grounding?: GroundingIndicator;
};

export function AnswerPanel({
  title = "Answer",
  answer,
  isLoading,
  error,
  grounding,
}: Props) {
  return (
    <DashboardCard title={title}>
      {grounding && (
        <div className="mb-3 flex items-start justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Grounding quality
            </p>
            <p className="mt-1 text-xs text-slate-600">{grounding.note}</p>
          </div>
          <span
            className={`rounded-full px-2 py-1 text-xs font-semibold ${grounding.toneClass}`}
          >
            {grounding.label}
          </span>
        </div>
      )}
      {isLoading && <p className="rounded-md bg-gray-50 px-3 py-2 text-sm text-gray-600">Generating answer...</p>}
      {!isLoading && error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {!isLoading && !error && !answer && <p className="text-sm text-gray-600">Submit a question to see model output.</p>}
      {!isLoading && !error && answer && <p className="whitespace-pre-wrap text-sm leading-7 text-gray-800">{answer}</p>}
    </DashboardCard>
  );
}
