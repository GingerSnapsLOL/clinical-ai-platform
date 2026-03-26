import { NextResponse } from "next/server";
const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL ?? "http://localhost:8000";

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }
  try {
    const upstream = await fetch(`${BACKEND_BASE_URL}/v1/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    const text = await upstream.text();
    const ct = upstream.headers.get("content-type") ?? "";
    if (ct.includes("application/json")) {
      try {
        return NextResponse.json(JSON.parse(text), { status: upstream.status });
      } catch {
        return new NextResponse(text, { status: upstream.status, headers: { "Content-Type": "application/json" } });
      }
    }
    return new NextResponse(text, { status: upstream.status, headers: { "Content-Type": ct || "text/plain; charset=utf-8" } });
  } catch (error) {
    const details = error instanceof Error ? error.message : "Upstream request failed.";
    return NextResponse.json({ error: "Failed to reach backend gateway.", details }, { status: 502 });
  }
}
