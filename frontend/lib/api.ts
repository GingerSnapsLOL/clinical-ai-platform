import type { AskRequest, AskResponse } from "@/lib/types";

const ASK_PROXY_ENDPOINT = "/api/ask";

export async function askClinicalQuestion(
  payload: AskRequest
): Promise<AskResponse> {
  // Backend validation rejects empty strings for note_text.
  // If it's empty/whitespace, omit the field from the request body.
  const requestBody: Record<string, unknown> = { ...payload };
  if (
    typeof requestBody.note_text === "string" &&
    requestBody.note_text.trim().length === 0
  ) {
    delete requestBody.note_text;
  }

  const response = await fetch(ASK_PROXY_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
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
