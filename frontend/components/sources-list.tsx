import type { Source } from "@/lib/types";
import { DashboardCard } from "@/components/dashboard-card";

function buildMetadata(source: Source): Record<string, unknown> {
  const base: Record<string, unknown> = {};
  if (source.metadata && typeof source.metadata === "object") {
    Object.assign(base, source.metadata);
  }
  const record = source as Record<string, unknown>;
  for (const [key, value] of Object.entries(record)) {
    if (["source_id", "title", "snippet", "score", "url", "metadata"].includes(key)) continue;
    if (value !== undefined && value !== null && value !== "") base[key] = value;
  }
  return base;
}

function sourceLabel(source: Source): string {
  const meta = buildMetadata(source);
  const candidates = [
    typeof meta.source === "string" ? meta.source : null,
    typeof meta.publisher === "string" ? meta.publisher : null,
    typeof meta.organization === "string" ? meta.organization : null,
    typeof meta.guideline === "string" ? meta.guideline : null,
    typeof meta.journal === "string" ? meta.journal : null,
  ].filter((v): v is string => Boolean(v && v.trim()));
  return candidates[0] ?? "Unknown source";
}

type SourcesListProps = {
  sources: Source[];
  devMode?: boolean;
};

export function SourcesList({ sources, devMode = false }: SourcesListProps) {
  return (
    <DashboardCard title="Sources">
      {sources.length === 0 ? <p className="text-sm text-gray-600">No sources returned.</p> : (
        <ul className="space-y-3">
          {sources.map((source, index) => {
            const metadata = buildMetadata(source);
            return (
              <li
                key={source.source_id ?? `${source.title}-${index}`}
                className="rounded-xl border border-gray-200 bg-white p-4"
              >
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm font-semibold text-gray-900">
                  {source.title || `Evidence #${index + 1}`}
                </p>
                <span className="rounded-full bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700">
                  Relevance: {typeof source.score === "number" ? source.score.toFixed(2) : "N/A"}
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-600">
                Source: <span className="font-medium text-slate-700">{sourceLabel(source)}</span>
              </p>

              <div className="mt-3">
                <p className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">
                  Snippet
                </p>
                <p className="rounded-md bg-gray-50 px-3 py-2 text-sm leading-6 text-gray-700">
                  {source.snippet || "No snippet text provided."}
                </p>
              </div>
              {!devMode && source.url && (
                <p className="mt-2 text-xs text-slate-500">
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noreferrer"
                    className="underline"
                  >
                    Open source
                  </a>
                </p>
              )}
              {Object.keys(metadata).length > 0 && (
                <details className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-2">
                  <summary className="cursor-pointer text-xs font-medium text-slate-700">
                    View raw metadata
                  </summary>
                  <pre className="meta-block mt-2">{JSON.stringify(metadata, null, 2)}</pre>
                </details>
              )}
              </li>
            );
          })}
        </ul>
      )}
    </DashboardCard>
  );
}
