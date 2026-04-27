import { DashboardCard } from "@/components/dashboard-card";

type GroundingIndicator = {
  label: "strong grounding" | "weak grounding" | "insufficient data";
  toneClass: string;
  note: string;
  confidenceLabel: "High" | "Medium" | "Low";
  groundingScore: number;
};

type Props = {
  title?: string;
  answer?: string;
  isLoading?: boolean;
  error?: string | null;
  grounding?: GroundingIndicator;
  showWeakGroundingWarning?: boolean;
  fallbackUsed?: boolean;
};

type SectionKey = "summary" | "treatment" | "risks" | "recommendations";

type StructuredSection = {
  key: SectionKey;
  title: string;
  items: string[];
};

const SECTION_ORDER: SectionKey[] = ["summary", "treatment", "risks", "recommendations"];

function normalizeHeading(raw: string): SectionKey | null {
  const heading = raw.trim().toLowerCase();
  if (heading === "summary") return "summary";
  if (heading === "treatment" || heading === "plan" || heading === "management") return "treatment";
  if (heading === "risks" || heading === "key risks" || heading === "risk") return "risks";
  if (
    heading === "recommendations" ||
    heading === "recommended monitoring" ||
    heading === "monitoring" ||
    heading === "next steps"
  ) {
    return "recommendations";
  }
  return null;
}

function sectionTitle(key: SectionKey): string {
  if (key === "summary") return "Summary";
  if (key === "treatment") return "Treatment";
  if (key === "risks") return "Risks";
  return "Recommendations";
}

function normalizeLine(line: string): string {
  return line.replace(/^\s*[-*]\s+/, "").trim();
}

function parseStructuredAnswer(answer: string): StructuredSection[] {
  const lines = answer.split(/\r?\n/);
  const rows: Record<SectionKey, string[]> = {
    summary: [],
    treatment: [],
    risks: [],
    recommendations: [],
  };
  let active: SectionKey | null = null;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // Accept markdown headings like "### Summary" / "## Risks".
    const headingMatch = trimmed.match(/^#{1,6}\s+(.+)$/);
    if (headingMatch) {
      active = normalizeHeading(headingMatch[1]);
      continue;
    }

    // Accept plain heading labels like "Summary:".
    const plainHeadingMatch = trimmed.match(/^([A-Za-z ]+):$/);
    if (plainHeadingMatch) {
      active = normalizeHeading(plainHeadingMatch[1]);
      continue;
    }

    if (active) {
      rows[active].push(normalizeLine(trimmed));
    }
  }

  const structured = SECTION_ORDER
    .map((key) => ({ key, title: sectionTitle(key), items: rows[key] }))
    .filter((section) => section.items.length > 0);

  // Fallback: if no explicit sections were found, render original text as one summary block.
  if (structured.length === 0) {
    return [
      {
        key: "summary",
        title: "Summary",
        items: answer
          .split(/\n+/)
          .map((line) => line.trim())
          .filter(Boolean),
      },
    ];
  }

  return structured;
}

export function AnswerPanel({
  title = "Answer",
  answer,
  isLoading,
  error,
  grounding,
  showWeakGroundingWarning = false,
  fallbackUsed = false,
}: Props) {
  const parsed = answer ? parseStructuredAnswer(answer) : [];

  return (
    <DashboardCard title={title}>
      {grounding && (
        <div className="mb-3 flex items-start justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Grounding quality
            </p>
            <p className="mt-1 text-xs text-slate-600">{grounding.note}</p>
            <p className="mt-2 text-xs text-slate-600">
              Confidence:{" "}
              <span className="font-semibold text-slate-800">{grounding.confidenceLabel}</span>{" "}
              <span className="font-mono">({grounding.groundingScore.toFixed(2)})</span>
            </p>
          </div>
          <span
            className={`rounded-full px-2 py-1 text-xs font-semibold ${grounding.toneClass}`}
          >
            {grounding.label} / {grounding.confidenceLabel}
          </span>
        </div>
      )}
      {showWeakGroundingWarning && (
        <div className="mb-3 inline-flex items-center rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-800">
          Weak grounding
        </div>
      )}
      {fallbackUsed && (
        <p className="mb-3 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-800">
          Warning: system fallback response was used due to upstream generation issues.
        </p>
      )}
      {isLoading && <p className="rounded-md bg-gray-50 px-3 py-2 text-sm text-gray-600">Generating answer...</p>}
      {!isLoading && error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {!isLoading && !error && !answer && <p className="text-sm text-gray-600">Submit a question to see model output.</p>}
      {!isLoading && !error && answer && (
        <div className="space-y-4">
          {parsed.map((section) => (
            <section key={section.key} className="space-y-2">
              <h3 className="text-sm font-semibold text-slate-900">{section.title}</h3>
              <ul className="list-disc space-y-1 pl-5 text-sm leading-7 text-gray-800">
                {section.items.map((item, index) => (
                  <li key={`${section.key}-${index}`}>{item}</li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}
    </DashboardCard>
  );
}
