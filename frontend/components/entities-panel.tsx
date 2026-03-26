import type { EntityItem } from "@/lib/types";
import { DashboardCard } from "@/components/dashboard-card";
export function EntitiesPanel({ entities }: { entities: EntityItem[] }) {
  return (
    <DashboardCard title="Entities">
      {entities.length === 0 ? <p className="text-sm text-gray-600">No entities extracted.</p> : (
        <div className="flex flex-wrap gap-2">
          {entities.map((entity, index) => (
            <div key={`${entity.text}-${index}`} className="rounded-full border border-gray-300 bg-gray-50 px-3 py-1 text-xs">
              <span className="font-medium">{entity.text}</span>
              <span className="ml-2 text-gray-500">{entity.label}</span>
            </div>
          ))}
        </div>
      )}
    </DashboardCard>
  );
}
