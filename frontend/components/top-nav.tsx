"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

function isDashboardActive(pathname: string) {
  return pathname === "/" || pathname === "/ask";
}

function isChatActive(pathname: string) {
  return pathname === "/chat";
}

export function TopNav() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-[1280px] items-center justify-between px-4 py-3 lg:px-8">
        <div className="text-sm font-semibold text-slate-900">
          Clinical AI
        </div>

        <div className="flex items-center gap-2">
          <Link
            href="/ask"
            className={
              isDashboardActive(pathname)
                ? "rounded-xl bg-slate-900 px-3 py-2 text-sm font-semibold text-white"
                : "rounded-xl px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
            }
          >
            Dashboard
          </Link>
          <Link
            href="/chat"
            className={
              isChatActive(pathname)
                ? "rounded-xl bg-slate-900 px-3 py-2 text-sm font-semibold text-white"
                : "rounded-xl px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
            }
          >
            Chat
          </Link>
        </div>
      </div>
    </nav>
  );
}

