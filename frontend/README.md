# SatQuery AI — Frontend (SIH26167)

Frontend-only build of the SatQuery AI interface: a chat-style query panel next
to a pan/zoom satellite image viewer that highlights the regions an answer is
grounded in. The vision model, VLM fusion, and RAG grounding layer described
in the project brief are **not** in this repo — they're being built
separately and will be wired in later through one file (`src/api/client.js`).

Until that's ready, the app runs entirely against a mock backend
(`src/api/mockBackend.js`) so it's fully demoable on its own: load the demo
scene, ask one of the sample questions, watch the answer highlight a region.

## Running it

```bash
npm install
npm run dev       # http://localhost:5173
```

```bash
npm run build     # production build -> dist/
npm run preview   # serve the production build locally
```

No API keys or backend required to run or demo this.

Note: the team's root `requirements.txt` / Python venv setup is for the
backend and ML modules — this is a Node/npm project, so `package.json` is
its equivalent. Nothing Python-related needs to be installed here.

## What's here

```
src/
  App.jsx                 — top-level state: scene(s), query type, log
  components/
    Header.jsx             — title bar, single/compare mode toggle
    QueryTypeTabs.jsx       — count / locate / compare / classify selector
    SceneLoader.jsx         — upload real images or load a synthetic demo scene
    ImageViewer.jsx         — Leaflet (CRS.Simple) pan/zoom viewer + bbox overlays
    CompareViewer.jsx       — two ImageViewers side by side with optional view sync
    StatusBar.jsx           — center/zoom/sensor readout, like a GIS tool
    QueryPanel.jsx          — the query log + input box
    QueryLogEntry.jsx       — one log row (question, pending, answer, error)
  api/
    client.js               — the ONE file to change to point at the real backend
    mockBackend.js           — canned responses standing in for the model today
  hooks/
    useSceneImage.js         — normalizes an uploaded file or a synthetic scene
  utils/
    syntheticScene.js        — draws a believable satellite-style tile on <canvas>,
                                so there's always something to demo without real imagery
  data/
    sampleQueries.js          — the example questions from the project brief
```

### Why a synthetic demo scene instead of a sample image file

There's no real ISRO tile to ship in this repo yet, and bundling a random
stock photo would be misleading in a demo about satellite imagery. Instead,
"Load demo scene" draws a small procedural scene (fields, a river, a town) on
a canvas, with a matching "after" variant for the before/after compare
queries. It's not a substitute for real RSVQA/DOTA/iSAID imagery — it's just
enough for the interaction to be demoable end to end today. Real images work
too: the "Upload image" buttons take any file.

## Integration contract for the ML/agent team

This app speaks the **team's shared I/O contract** (images / natural-language
query / metadata in, success + result + confidence + visual output +
metadata + error out) on the wire in `src/api/client.js`, and normalizes it
internally to a plain shape the UI components consume:

```
// Wire request (POST {VITE_API_BASE_URL}/query):
{
  images: { before: {...}, after: {...} | null },
  query: string,
  metadata: { queryType: "count" | "locate" | "compare" | "classify" }
}

// Wire response, per the team contract:
{
  success: boolean,
  result?: string,             // textual answer, present when success
  confidence?: number,         // 0..1, if available
  visual_output?: { regions: Region[] },  // see open question below
  metadata?: { queryType?: string, ... },
  error?: string                // present when success is false
}
```

**Open question for whoever owns the contract:** "visual output path/data"
doesn't say whether that's a rendered image path or structured data. This
app assumes structured region data (`visual_output.regions`, pixel bounding
boxes) because the UI needs coordinates to draw interactive highlight boxes
and support "click a region to fly to it" — a static image path would cover
display but not that interaction. Worth confirming before the real backend
is built around a different assumption.

This is defined once in `src/api/mockBackend.js` (see the JSDoc typedefs at
the top) and consumed once in `src/api/client.js`, which normalizes it to
`{ answer, confidence, regions, queryType }` for every component downstream.
To go live:

1. Stand up the FastAPI endpoint from the tech-stack doc, speaking the
   contract above.
2. Set `VITE_API_BASE_URL` (copy `.env.example` to `.env`).
3. That's it — `client.js` already calls the real endpoint once that env
   var is set. Nothing else in the app needs to change, since every
   component consumes `sendQuery()`, not the mock directly.

Note in `client.js`: sending full base64 images on every query will be slow
once real tiles are involved — prefer an `/images` upload endpoint that
returns an `imageId` to reference in `/query`, instead of re-sending pixels
each time. That's a backend-side change; the frontend contract above doesn't
need to change to support it (`before`/`after` can carry an `imageId` field
alongside `width`/`height` whenever that's ready).

## Design notes

Built to read like an analyst's instrument panel (GIS-tool conventions —
coordinate/zoom/sensor readouts, hairline dividers, a query *log* rather than
chat bubbles) rather than a generic chatbot skin, since the real audience is
someone cross-referencing bands and tiles, not chatting casually. Two
accents only: teal for active/system state, coral/amber for detected regions
on the imagery itself.

## Known gaps / next steps

- No real backend yet — see integration contract above.
- No auth/session handling — add if the deployed demo needs to be gated.
- `CompareViewer`'s view-sync assumes both images are roughly the same
  geographic extent; for wildly different-sized before/after tiles you may
  want independent zoom with only pan synced, not full view sync.
- No automated tests yet (no test runner is configured). If this grows past
  the hackathon, Vitest + React Testing Library would be the natural fit
  given the Vite setup already here.
