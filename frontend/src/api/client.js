/**
 * SINGLE INTEGRATION POINT for the real backend.
 *
 * The wire format here matches the team's shared I/O contract exactly
 * (see mockBackend.js for the full shape and one open question flagged
 * for the team). This file's job is to speak that contract on the wire,
 * then normalize the response into the plain shape the rest of the app
 * already uses ({ answer, confidence, regions, queryType }) — so UI
 * components never need to know about `success`/`result`/`visual_output`
 * at all, and swapping mock for real backend needs no changes anywhere
 * else in the app.
 *
 * NOTE: there is no query-type selector in the UI. The question's type
 * (count / locate / compare / classify) is classified by the model
 * itself from the natural-language query — the frontend never sends one.
 * The classified type comes back in `metadata.queryType` on the response
 * and is only used for display (e.g. showing what the model inferred).
 *
 * To go live:
 *   1. Set VITE_API_BASE_URL in .env (see .env.example).
 *   2. That's it — sendQuery() below already calls the real endpoint
 *      whenever that env var is set, using the exact contract shape.
 *
 * Expected real endpoint: POST {VITE_API_BASE_URL}/query
 *   Request JSON (per team contract — images / natural-language query / metadata):
 *     {
 *       images: {
 *         before: { dataUrl?: string, width: number, height: number },
 *         after:  { dataUrl?: string, width: number, height: number } | null
 *       },
 *       query: string,
 *       metadata: {}   // reserved for future use — no queryType is sent
 *     }
 *   Response JSON (per team contract):
 *     {
 *       success: boolean,
 *       result?: string,            // textual answer, present when success
 *       confidence?: number,        // 0..1, if available
 *       visual_output?: { regions: Region[] },  // see open question in mockBackend.js
 *       metadata?: { queryType?: string, ... },  // model's own classification of the query
 *       error?: string              // present when success is false
 *     }
 *
 * Sending full base64 dataUrls to a real backend will be heavy for large
 * tiles — once real infra exists, prefer uploading the image once (e.g.
 * POST /images) and sending back an imageId to reference here instead.
 */

import { runQuery as mockRunQuery } from './mockBackend';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

// The frontend's internal "scene" shape (see useSceneImage.js) uses `url`
// for local rendering. The wire contract uses `dataUrl`. This is the one
// place that difference gets bridged — everything else on either side
// only ever sees its own field name.
function toWireImage(scene) {
  if (!scene) return null;
  return {
    dataUrl: scene.url,
    width: scene.width,
    height: scene.height,
    modality: scene.modality,
  };
}

export async function sendQuery({ question, before, after }) {
  const requestBody = {
    images: { before: toWireImage(before), after: toWireImage(after) },
    query: question,
    metadata: {},
  };

  const response = API_BASE_URL
    ? await callRealBackend(requestBody)
    : await mockRunQuery(requestBody);

  if (!response.success) {
    throw new Error(response.error || 'Query failed with no error message.');
  }

  // Normalize the contract shape into what the UI components consume.
  return {
    answer: response.result,
    confidence: response.confidence,
    regions: response.visual_output?.regions || [],
    queryType: response.metadata?.queryType || 'unknown',
    visualOutput: response.metadata?.visualOutput
      ? `${API_BASE_URL}${response.metadata.visualOutput.replaceAll('\\', '/')}?t=${Date.now()}`
      : null,
  };
}

async function callRealBackend(requestBody) {
  const res = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
  });
  if (!res.ok) {
    // HTTP-level failure (backend down, 500, etc.) — not the same as a
    // contract-level { success: false }, but the caller shouldn't need
    // to tell the two apart, so normalize into the same shape here.
    return { success: false, error: `Request failed: ${res.status} ${res.statusText}` };
  }
  return res.json();
}
