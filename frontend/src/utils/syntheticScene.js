/**
 * Generates a believable "satellite tile" entirely on <canvas>, client-side.
 *
 * Why this exists: the real imagery (Cartosat / Resourcesat / RISAT tiles)
 * doesn't exist yet on the frontend side — it either comes from a judge's
 * uploaded file, or later from whatever dataset the ML team wires up. This
 * generator gives the UI something real to pan/zoom/highlight today, and a
 * matched before/after pair for the multi-temporal demo query, without
 * bundling any external image asset or hitting the network.
 *
 * Swap-out point: once real tiles are available, `loadImageFile()` in
 * useSceneImage.js already handles arbitrary uploaded images the same way —
 * this generator is only used for the "Load demo scene" button.
 */

// Deterministic PRNG so a given seed always draws the same fields/river,
// which matters for producing a believable "after" image from a "before" one.
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const FIELD_GREENS = ['#3E5C3A', '#4C6B3F', '#5B7A46', '#35502F', '#6B8A4E'];
const SOIL_TONES = ['#7A6248', '#8C7355', '#6E5A40'];
const WATER = '#2B5C73';
const WATER_FLOOD = '#3E7FA0';
const URBAN = '#8A8680';
const URBAN_ROOF = '#B0ABA0';

function drawFields(ctx, rng, w, h) {
  const cols = 6;
  const rows = 5;
  const cw = w / cols;
  const ch = h / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const isSoil = rng() < 0.25;
      const palette = isSoil ? SOIL_TONES : FIELD_GREENS;
      ctx.fillStyle = palette[Math.floor(rng() * palette.length)];
      const jitter = 6;
      ctx.fillRect(
        c * cw - jitter / 2,
        r * ch - jitter / 2,
        cw + jitter,
        ch + jitter
      );
      // row texture
      ctx.strokeStyle = 'rgba(0,0,0,0.08)';
      ctx.lineWidth = 1;
      for (let i = 0; i < 6; i++) {
        ctx.beginPath();
        ctx.moveTo(c * cw, r * ch + (i * ch) / 6);
        ctx.lineTo(c * cw + cw, r * ch + (i * ch) / 6);
        ctx.stroke();
      }
    }
  }
}

function drawRiver(ctx, rng, w, h, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = h * 0.09;
  ctx.lineCap = 'round';
  ctx.beginPath();
  const points = [];
  const n = 6;
  for (let i = 0; i <= n; i++) {
    const x = (i / n) * w;
    const y = h * 0.35 + Math.sin(i * 1.3 + rng() * 2) * h * 0.12 + (rng() - 0.5) * h * 0.06;
    points.push([x, y]);
  }
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length - 1; i++) {
    const midX = (points[i][0] + points[i + 1][0]) / 2;
    const midY = (points[i][1] + points[i + 1][1]) / 2;
    ctx.quadraticCurveTo(points[i][0], points[i][1], midX, midY);
  }
  ctx.stroke();
  return points;
}

function drawUrbanBlock(ctx, x, y, w, h, rng) {
  ctx.fillStyle = URBAN;
  ctx.fillRect(x, y, w, h);
  const rows = Math.max(2, Math.round(h / 14));
  const cols = Math.max(2, Math.round(w / 14));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (rng() < 0.7) {
        ctx.fillStyle = URBAN_ROOF;
        ctx.fillRect(x + c * (w / cols) + 1, y + r * (h / rows) + 1, w / cols - 2, h / rows - 2);
      }
    }
  }
}

/**
 * @param {Object} opts
 * @param {number} opts.seed
 * @param {number} [opts.width=1024]
 * @param {number} [opts.height=768]
 * @param {'baseline'|'flood'|'construction'} [opts.variant='baseline']
 * @returns {{ dataUrl: string, width: number, height: number, capturedOn: string, sensor: string }}
 */
export function generateSyntheticScene({ seed, width = 1024, height = 768, variant = 'baseline' }) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const rng = mulberry32(seed);

  drawFields(ctx, rng, width, height);
  const riverColor = variant === 'flood' ? WATER_FLOOD : WATER;
  const riverPoints = drawRiver(ctx, rng, width, height, riverColor);

  if (variant === 'flood') {
    // widen the water body to simulate waterlogged cropland along the bank
    ctx.strokeStyle = 'rgba(62,127,160,0.55)';
    ctx.lineWidth = height * 0.22;
    ctx.beginPath();
    ctx.moveTo(riverPoints[0][0], riverPoints[0][1]);
    for (let i = 1; i < riverPoints.length; i++) {
      ctx.lineTo(riverPoints[i][0], riverPoints[i][1]);
    }
    ctx.stroke();
  }

  // fixed urban cluster, bottom-right
  const townX = width * 0.62;
  const townY = height * 0.66;
  drawUrbanBlock(ctx, townX, townY, width * 0.22, height * 0.2, rng);

  if (variant === 'construction') {
    // a new cluster near the riverbank that wasn't there in baseline
    drawUrbanBlock(ctx, width * 0.28, height * 0.42, width * 0.14, height * 0.12, rng);
  }

  // light vignette + grain to sell the "sensor" feel
  const grad = ctx.createRadialGradient(
    width / 2,
    height / 2,
    height * 0.2,
    width / 2,
    height / 2,
    height * 0.9
  );
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(1, 'rgba(0,0,0,0.25)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  return {
    dataUrl: canvas.toDataURL('image/png'),
    width,
    height,
    capturedOn: variant === 'baseline' ? '2026-08-01' : '2026-08-14',
    sensor: variant === 'flood' || variant === 'baseline' ? 'Cartosat-3 (optical)' : 'Resourcesat-2A (optical)',
  };
}
