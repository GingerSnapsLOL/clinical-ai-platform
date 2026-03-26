"use client";
import { useState } from "react";
import type { AskRequest } from "@/lib/types";

type Props = { onSubmit: (data: AskRequest) => void; isSubmitting: boolean };

const demoPrompts = [
  {
    label: "Hypertension treatment",
    question: "What is a first-line treatment approach for hypertension in adults?",
    note_text:
      "Adult patient with persistent elevated blood pressure readings over multiple visits. No acute end-organ symptoms reported.",
  },
  {
    label: "Thiazide diuretics",
    question: "When are thiazide diuretics recommended for hypertension management?",
    note_text:
      "Patient has stage 1 hypertension and is reviewing medication options. Basic metabolic panel available; renal function is being monitored.",
  },
  {
    label: "Calcium channel blockers",
    question: "In which scenarios are calcium channel blockers preferred for blood pressure control?",
    note_text:
      "Patient with hypertension reports prior cough on ACE inhibitor and asks about alternative classes for ongoing blood pressure control.",
  },
  {
    label: "Unknown query",
    question: "What is the recommended treatment for condition XYZ-9999?",
    note_text:
      "Clinical note text is limited and does not clearly describe a known diagnosis, creating an intentionally ambiguous test case.",
  },
] as const;

export function AskForm({ onSubmit, isSubmitting }: Props) {
  const [mode, setMode] = useState<AskRequest["mode"]>("strict");
  const [noteText, setNoteText] = useState("");
  const [question, setQuestion] = useState("");

  return (
    <form
      className="space-y-5 rounded-2xl border border-slate-200 bg-white p-5"
      onSubmit={(e) => {
        e.preventDefault();
        if (!mode.trim() || !noteText.trim() || !question.trim()) return;
        onSubmit({ mode, note_text: noteText.trim(), question: question.trim() });
      }}
    >
      <div>
        <p className="mb-2 block text-sm font-medium text-slate-800">Demo prompts</p>
        <div className="flex flex-wrap gap-2">
          {demoPrompts.map((prompt) => (
            <button
              key={prompt.label}
              type="button"
              className="rounded-full border border-slate-300 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100"
              onClick={() => {
                setQuestion(prompt.question);
                setNoteText(prompt.note_text);
              }}
            >
              {prompt.label}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="mb-1 block text-sm font-medium text-slate-800">Question</label>
        <textarea className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm leading-6 text-slate-800" rows={3} value={question} onChange={(e) => setQuestion(e.target.value)} required />
      </div>
      <div>
        <label className="mb-1 block text-sm font-medium text-slate-800">Note text</label>
        <textarea className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm leading-6 text-slate-800" rows={7} value={noteText} onChange={(e) => setNoteText(e.target.value)} required />
      </div>
      <div>
        <label className="mb-1 block text-sm font-medium text-slate-800">Mode</label>
        <select className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800" value={mode} onChange={(e) => setMode(e.target.value as AskRequest["mode"])}>
          <option value="strict">strict</option>
          <option value="hybrid">hybrid</option>
        </select>
      </div>
      <button disabled={isSubmitting} className="w-full rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-medium text-white disabled:opacity-60">
        {isSubmitting ? "Running..." : "Ask"}
      </button>
    </form>
  );
}
