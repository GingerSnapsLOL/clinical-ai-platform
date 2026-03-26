import type { SourceItem } from "@/lib/types";
import { DashboardCard } from "@/components/dashboard-card";

function buildMetadata(source: SourceItem): Record<string, unknown> {
  const record = source as Record<string, unknown>;
  const metadata: Record<string, unknown> = {};

  if (source.id) metadata.id = source.id;
  if (source.url) metadata.url = source.url;

  for (const [key, value] of Object.entries(record)) {
    if (["id", "title", "snippet", "score", "url"].includes(key)) continue;
    if (value !== undefined && value !== null && value !== "") metadata[key] = value;
  }

  return metadata;
}

export function SourcesList({ sources }: { sources: SourceItem[] }) {
  return (
    <DashboardCard title="Sources">
      {sources.length === 0 ? <p className="text-sm text-gray-600">No sources returned.</p> : (
        <ul className="space-y-3">
          {sources.map((source, index) => (
            <li
              key={source.id ?? `${source.title}-${index}`}
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

              <div className="mt-3">
                <p className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">
                  Metadata
                </p>
                <pre className="meta-block">{JSON.stringify(buildMetadata(source), null, 2)}</pre>
              </div>

              <div className="mt-3">
                <p className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">
                  Snippet
                </p>
                <p className="rounded-md bg-gray-50 px-3 py-2 text-sm leading-6 text-gray-700">
                  {source.snippet || "No snippet text provided."}
                </p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </DashboardCard>
  );
}
