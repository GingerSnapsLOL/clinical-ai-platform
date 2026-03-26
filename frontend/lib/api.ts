import type { AskRequest, AskResponse } from "@/lib/types";

const ASK_ENDPOINT = "/api/ask";

export async function askClinicalQuestion(
  payload: AskRequest
): Promise<AskResponse> {
  const response = await fetch(ASK_ENDPOINT, {
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
