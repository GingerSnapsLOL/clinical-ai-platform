import type { AskRequest, AskResponse } from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

export async function askClinicalQuestion(
  payload: AskRequest
): Promise<AskResponse> {
  if (!API_BASE_URL) {
    throw new Error("NEXT_PUBLIC_API_URL is not configured.");
  }

  const response = await fetch(`${API_BASE_URL}/v1/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const details = await response.text().catch(() => "");
    throw new Error(
      details
        ? `Request failed (${response.status}): ${details}`
        : `Request failed with status ${response.status}`
    );
  }

  return (await response.json()) as AskResponse;
}
