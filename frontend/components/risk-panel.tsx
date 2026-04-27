import type { RiskBlock } from "@/lib/types";
import { DashboardCard } from "@/components/dashboard-card";

const riskToneClass: Record<"low" | "medium" | "high", string> = {
  low: "border border-green-200 bg-green-50 text-green-800",
  medium: "border border-amber-200 bg-amber-50 text-amber-800",
  high: "border border-red-200 bg-red-50 text-red-800",
};

const riskLabel: Record<"low" | "medium" | "high", string> = {
  low: "Low Risk",
  medium: "Medium Risk",
  high: "High Risk",
};

export function RiskPanel({
  riskBlock,
  devMode = false,
}: {
  riskBlock?: RiskBlock | null;
  devMode?: boolean;
}) {
  return (
    <DashboardCard title="Risk Assessment">
      {!riskBlock || riskBlock.risk_available === false ? (
        <p className="text-sm text-gray-600">
          {riskBlock?.rationale || "Risk assessment unavailable (insufficient data)"}
        </p>
      ) : !riskBlock.label ? (
        <p className="text-sm text-gray-600">Risk assessment unavailable (insufficient data)</p>
      ) : (
        <>
          {typeof riskBlock.score === "number" && (
            <p className="text-xs text-gray-600">
              Score: <span className="font-mono font-medium">{riskBlock.score.toFixed(3)}</span>
            </p>
          )}
          <span
            className={`mt-2 inline-flex rounded-full px-3 py-1 text-xs font-semibold ${riskToneClass[riskBlock.label]}`}
          >
            {riskLabel[riskBlock.label]}
          </span>
          {typeof riskBlock.confidence === "number" && (
            <p className="mt-2 text-xs text-gray-600">
              Confidence:{" "}
              <span className="font-mono font-medium">{riskBlock.confidence.toFixed(2)}</span>
            </p>
          )}
          {riskBlock.rationale && (
            <p className="mt-2 text-sm text-gray-700">{riskBlock.rationale}</p>
          )}

          {(devMode || (riskBlock.explanation && riskBlock.explanation.length > 0)) && (
            <div className="mt-3">
              <p className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">
                Explanation
              </p>
              {riskBlock.explanation && riskBlock.explanation.length > 0 ? (
                <ul className="list-disc space-y-1 pl-5 text-sm text-gray-700">
                  {riskBlock.explanation.map((feat, index) => (
                    <li key={`${feat.feature}-${index}`}>
                      {feat.feature}: {feat.contribution.toFixed(4)}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-gray-600">No structured explanation provided.</p>
              )}
            </div>
          )}
        </>
      )}
    </DashboardCard>
  );
}
