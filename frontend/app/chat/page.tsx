"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { askClinicalQuestion } from "@/lib/api";
import type { SourceItem } from "@/lib/types";

type ChatRole = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  sources?: SourceItem[];
};

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

const demoPrompts: Array<{ label: string; question: string }> = [
  {
    label: "Hypertension treatment",
    question: "What is a first-line treatment approach for hypertension in adults?",
  },
  {
    label: "Thiazide diuretics",
    question: "When are thiazide diuretics recommended for hypertension management?",
  },
  {
    label: "Calcium channel blockers",
    question:
      "In which scenarios are calcium channel blockers preferred for blood pressure control?",
  },
  {
    label: "Unknown query",
    question: "What is the recommended treatment for condition XYZ-9999?",
  },
];

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: uid(),
      role: "assistant",
      content:
        "Hi. Add a short “About myself” note (symptoms, history, context), then ask your question below.",
    },
  ]);
  /** Sent as `note_text` to `/v1/ask` (required by the API). */
  const [aboutMyself, setAboutMyself] = useState("");
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [assistantLoadingText, setAssistantLoadingText] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  const bottomRef = useRef<HTMLDivElement | null>(null);

  const canSend = useMemo(
    () =>
      aboutMyself.trim().length > 0 &&
      draft.trim().length > 0 &&
      !isSending,
    [aboutMyself, draft, isSending]
  );

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
  }, [messages, assistantLoadingText]);

  async function handleSend() {
    const note = aboutMyself.trim();
    const text = draft.trim();
    if (!note || !text || isSending) return;

    setIsSending(true);
    setDraft("");
    setAssistantLoadingText("Generating answer...");

    const userMsg: ChatMessage = { id: uid(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const response = await askClinicalQuestion({
        mode: "strict",
        note_text: note,
        question: text,
      });

      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "assistant",
          content: response.answer,
          sources: response.sources,
        },
      ]);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to get response.";
      setMessages((prev) => [...prev, { id: uid(), role: "assistant", content: message }]);
    } finally {
      setIsSending(false);
      setAssistantLoadingText(null);
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-6 lg:px-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
          Chat
        </h1>
        <p className="text-sm text-slate-600">
          Questions go to the gateway as <code className="text-xs">question</code>; your context
          note is sent as <code className="text-xs">note_text</code> (required by the API).
        </p>
        <label className="flex items-center gap-2 text-sm text-slate-700">
          <input
            type="checkbox"
            checked={showDetails}
            onChange={(e) => setShowDetails(e.target.checked)}
          />
          Show details
        </label>

        <div className="pt-2">
          <p className="mb-2 text-sm font-medium text-slate-800">Demo prompts</p>
          <div className="flex flex-wrap gap-2">
            {demoPrompts.map((prompt) => (
              <button
                key={prompt.label}
                type="button"
                className="rounded-full border border-slate-300 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100"
                onClick={() => setDraft(prompt.question)}
              >
                {prompt.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <section className="flex min-h-[60vh] flex-1 flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white">
        <div className="flex-1 overflow-auto p-4">
          <div className="space-y-3">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={
                  msg.role === "user"
                    ? "flex justify-end"
                    : "flex justify-start"
                }
              >
                <div
                  className={
                    msg.role === "user"
                      ? "max-w-[78%] rounded-2xl bg-slate-900 px-4 py-3 text-sm text-white"
                      : "max-w-[78%] rounded-2xl bg-slate-50 px-4 py-3 text-sm text-slate-900"
                  }
                >
                  <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                    {msg.role}
                  </div>
                  <div className="whitespace-pre-wrap leading-6">
                    {msg.content}
                  </div>

                  {msg.role === "assistant" &&
                    showDetails &&
                    msg.sources &&
                    msg.sources.length > 0 && (
                      <div className="mt-3 border-t border-slate-200 pt-3">
                        <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                          Sources
                        </p>
                        <div className="space-y-2">
                          {msg.sources.map((source, index) => (
                            <div
                              key={source.source_id ?? `${source.title}-${index}`}
                              className="rounded-lg border border-slate-200 bg-white/60 p-3"
                            >
                              <div className="text-sm font-semibold text-slate-900">
                                {source.title}
                              </div>
                              {typeof source.score === "number" && (
                                <div className="text-xs text-slate-600">
                                  Relevance: {source.score.toFixed(2)}
                                </div>
                              )}
                              {source.snippet && (
                                <div className="mt-1 whitespace-pre-wrap text-sm leading-6 text-slate-700">
                                  {source.snippet}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                </div>
              </div>
            ))}
            {assistantLoadingText && (
              <div className="flex justify-start">
                <div className="max-w-[78%] rounded-2xl bg-slate-50 px-4 py-3 text-sm text-slate-900">
                  <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                    assistant
                  </div>
                  <div className="whitespace-pre-wrap leading-6">
                    {assistantLoadingText}
                  </div>
                </div>
              </div>
            )}
          </div>
          <div ref={bottomRef} />
        </div>

        <form
          className="border-t border-slate-200 bg-white p-4"
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
        >
          <div className="mb-3 space-y-1.5">
            <label
              htmlFor="chat-about"
              className="block text-xs font-semibold uppercase tracking-wide text-slate-600"
            >
              About myself
            </label>
            <textarea
              id="chat-about"
              value={aboutMyself}
              onChange={(e) => setAboutMyself(e.target.value)}
              placeholder="e.g. Headache for 2 days, temperature 37.5°C, no prior migraines…"
              rows={3}
              className="w-full resize-y rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-slate-400"
            />
            <p className="text-xs text-slate-500">
              Used for every message in this session as the clinical note (<code>note_text</code>).
            </p>
          </div>
          <div className="flex items-end gap-3">
            <div className="flex-1">
              <label className="sr-only" htmlFor="chat-input">
                Message
              </label>
              <input
                id="chat-input"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder="Ask a clinical question..."
                className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-slate-400"
                autoComplete="off"
              />
            </div>
            <button
              type="submit"
              disabled={!canSend}
              className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white disabled:opacity-60"
            >
              {isSending ? "Sending..." : "Send"}
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}

