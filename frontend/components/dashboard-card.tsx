import type { ReactNode } from "react";

type DashboardCardProps = {
  title: string;
  children: ReactNode;
  className?: string;
};

export function DashboardCard({ title, children, className }: DashboardCardProps) {
  return (
    <section className={`dashboard-card ${className ?? ""}`.trim()}>
      <h2 className="mb-3 text-lg font-semibold tracking-tight text-slate-900">{title}</h2>
      {children}
    </section>
  );
}
