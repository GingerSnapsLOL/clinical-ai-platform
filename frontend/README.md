# Frontend

Standalone Next.js frontend for the Clinical AI Platform monorepo.

## Stack

- Next.js (App Router)
- TypeScript
- ESLint
- Tailwind CSS

## What is implemented

### App structure

- `app/page.tsx` - landing page
- `app/ask/page.tsx` - main clinical testing console
- `app/api/ask/route.ts` - frontend proxy route to backend `/v1/ask`
- `components/*` - reusable dashboard UI components
- `lib/api.ts` - typed frontend API client
- `lib/types.ts` - shared request/response types

### Ask flow and API

- Typed request payload for `/v1/ask`:
  - `mode` (`strict` | `hybrid`)
  - `note_text`
  - `question`
- Frontend calls `/api/ask` (not backend URL directly)
- Route handler forwards to `${BACKEND_BASE_URL}/v1/ask`
- Clean error forwarding with status preservation

### Dashboard UI

- Portfolio-style internal clinical console layout
- Responsive two-column structure
- Sticky request form rail on desktop
- Consistent reusable card styling across panels
- Clear typography hierarchy and spacing

### Panels and UX

- Answer panel with loading/error handling
- Grounding quality indicator:
  - strong grounding (green)
  - weak grounding (yellow)
  - insufficient data (red)
- Sources panel redesigned as evidence cards with:
  - source title
  - source metadata
  - relevance badge
  - snippet text
- Entities panel
- Risk assessment panel with:
  - colored status badge (low/medium/high/unknown)
  - structured explanation list
- Diagnostics panel with latency metrics:
  - total request time
  - retrieval time
  - llm time
- Trace/Debug collapsible panel with:
  - `trace_id`
  - warnings
  - retrieval diagnostics
  - planner decisions (if available)

### Compare mode

- Toggleable compare mode in request section
- Sends the same request twice
- Displays Answer A and Answer B side-by-side
- Shows latency differences and source differences

### Demo prompts

Request form includes one-click prefill buttons for:

- Hypertension treatment
- Thiazide diuretics
- Calcium channel blockers
- Unknown query

## Environment

Create `frontend/.env.local`:

```env
BACKEND_BASE_URL=http://localhost:8000
```

## Run locally

```bash
cd frontend
npm install
node ./node_modules/next/dist/bin/next dev
```

Open:

- `http://localhost:3000`
- `http://localhost:3000/ask`

## Notes

- If backend rejects mode values, ensure mode is `strict` or `hybrid`.
- If `npm run dev` behaves inconsistently in mixed Windows/WSL setups, use:
  - `node ./node_modules/next/dist/bin/next dev`
