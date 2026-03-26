import type { RiskAssessment } from "@/lib/types";
import { DashboardCard } from "@/components/dashboard-card";

const riskToneClass: Record<NonNullable<RiskAssessment["level"]>, string> = {
  low: "border border-green-200 bg-green-50 text-green-800",
  medium: "border border-amber-200 bg-amber-50 text-amber-800",
  high: "border border-red-200 bg-red-50 text-red-800",
  unknown: "border border-gray-200 bg-gray-50 text-gray-700",
};

const riskLabel: Record<NonNullable<RiskAssessment["level"]>, string> = {
  low: "Low Risk",
  medium: "Medium Risk",
  high: "High Risk",
  unknown: "Unknown Risk",
};

export function RiskPanel({ risk }: { risk?: RiskAssessment }) {
  const level: NonNullable<RiskAssessment["level"]> = risk?.level ?? "unknown";
  const details: string[] = [];

  if (risk?.rationale) details.push(risk.rationale);
  if (risk?.recommendations?.length) details.push(...risk.recommendations);

  return (
    <DashboardCard title="Risk Assessment">
      {!risk ? (
        <p className="text-sm text-gray-600">No risk assessment yet.</p>
      ) : (
        <>
          <span
            className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${riskToneClass[level]}`}
          >
            {riskLabel[level]}
          </span>

          <div className="mt-3">
            <p className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">
              Explanation
            </p>
            {details.length > 0 ? (
              <ul className="list-disc space-y-1 pl-5 text-sm text-gray-700">
                {details.map((item, index) => (
                  <li key={`${item}-${index}`}>{item}</li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-gray-600">
                No structured explanation provided.
              </p>
            )}
          </div>
        </>
      )}
    </DashboardCard>
  );
}
