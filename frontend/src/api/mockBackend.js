/**
 * MOCK BACKEND — stands in for the real inference service until the
 * ML/agent team's endpoint is ready.
 *
 * Shaped to match the team's shared I/O contract exactly:
 *
 *   Input:  images, natural-language query, metadata (if required)
 *   Output: success status, textual answer/result, confidence (if available),
 *           visual output path/data (if applicable), model/task metadata,
 *           error message (if execution fails)
 *
 * @typedef {Object} Region
 * @property {string} id
 * @property {string} label
 * @property {[number, number, number, number]} bbox - [x, y, w, h] in the ORIGINAL image's pixel space, origin top-left
 * @property {number} score            - 0..1 model confidence for this region
 * @property {'before'|'after'} [image] - which image this box belongs to (compare mode only)
 *
 * @typedef {Object} ContractResponse
 * @property {boolean} success
 * @property {string} [result]           - textual answer (present when success)
 * @property {number} [confidence]       - 0..1 overall confidence, if available
 * @property {Object} [visual_output]    - visual output, if applicable — see note below
 * @property {Region[]} [visual_output.regions]
 * @property {Object} [metadata]         - model/task metadata
 * @property {string} [error]            - present when success is false
 *
 * OPEN QUESTION FOR THE TEAM (flag this before backend implementation locks in):
 * the doc says "visual output path/data" without saying which — a rendered
 * image path, or raw structured data. This mock assumes structured region
 * data (bounding boxes) under `visual_output.regions`, since the frontend
 * needs pixel coordinates to draw interactive highlight boxes on the map —
 * a static image path would work for *display* but not for the
 * click-a-region-to-locate-it interaction already built. Confirm this with
 * whoever owns the contract before the real backend is built around a
 * different assumption.
 */

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function clampBox([x, y, w, h], imgW, imgH) {
  const cx = Math.max(0, Math.min(x, imgW - 4));
  const cy = Math.max(0, Math.min(y, imgH - 4));
  const cw = Math.max(4, Math.min(w, imgW - cx));
  const ch = Math.max(4, Math.min(h, imgH - cy));
  return [cx, cy, cw, ch];
}

function seededBoxes(seed, count, imgW, imgH) {
  let s = seed;
  const rand = () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
  return Array.from({ length: count }, (_, i) => {
    const w = imgW * (0.1 + rand() * 0.12);
    const h = imgH * (0.08 + rand() * 0.1);
    const x = rand() * (imgW - w);
    const y = rand() * (imgH - h);
    return {
      id: `r${seed}-${i}`,
      bbox: clampBox([x, y, w, h], imgW, imgH),
      score: 0.7 + rand() * 0.28,
    };
  });
}

const KEYWORD_RULES = [
  {
    match: /waterlog|flood|inundat/i,
    queryType: 'compare',
    answer:
      'Approximately 14.6 hectares of cropland along the riverbank are newly waterlogged compared to the earlier pass — mostly the low-lying parcels on the southern bend.',
    confidence: 0.84,
    makeRegions: (before, after) => [
      ...seededBoxes(11, 2, before.width, before.height).map((r) => ({ ...r, label: 'cropland (baseline)', image: 'before' })),
      ...seededBoxes(12, 3, after.width, after.height).map((r) => ({ ...r, label: 'waterlogged parcel', image: 'after' })),
    ],
  },
  {
    match: /construction|new building|riverbank/i,
    queryType: 'compare',
    answer:
      'One new built-up cluster appears near the riverbank that was not present in the earlier image — roughly 3 structures across 0.4 hectares.',
    confidence: 0.79,
    makeRegions: (before, after) => [
      ...seededBoxes(21, 1, after.width, after.height).map((r) => ({ ...r, label: 'new construction', image: 'after' })),
    ],
  },
  {
    match: /landslide|shadow|scar/i,
    queryType: 'classify',
    answer:
      'This is most consistent with a landslide scar rather than a shadow — the boundary follows the slope contour and shows disturbed-soil backscatter rather than a hard illumination edge.',
    confidence: 0.71,
    makeRegions: (before) => seededBoxes(31, 1, before.width, before.height).map((r) => ({ ...r, label: 'landslide scar (candidate)' })),
  },
  {
    match: /how many|count/i,
    queryType: 'count',
    answer: 'Detected 4 distinct built-up structures in the selected region.',
    confidence: 0.88,
    makeRegions: (before) => seededBoxes(41, 4, before.width, before.height).map((r, i) => ({ ...r, label: `structure ${i + 1}` })),
  },
];

const FALLBACK = {
  queryType: 'locate',
  answer:
    'Here is the region most relevant to that query, based on the visual features detected in the current scene.',
  confidence: 0.62,
  makeRegions: (before) => seededBoxes(1, 1, before.width, before.height).map((r) => ({ ...r, label: 'region of interest' })),
};

/**
 * @param {Object} args
 * @param {string} args.query               - natural-language query
 * @param {Object} args.images              - { before, after }
 * @param {Object} [args.metadata]          - reserved for future use; not used for classification
 * @returns {Promise<ContractResponse>}
 */
export async function runQuery({ query, images }) {
  await delay(650 + Math.random() * 550);

  const { before, after } = images;
  if (!before) {
    return { success: false, error: 'No image supplied for this query.' };
  }

  const rule = KEYWORD_RULES.find((r) => r.match.test(query)) || FALLBACK;
  const regions = rule.makeRegions(before, after || before);

  return {
    success: true,
    result: rule.answer,
    confidence: rule.confidence,
    visual_output: { regions },
    metadata: {
      queryType: rule.queryType,
      model: 'mock-v0',
    },
  };
}
