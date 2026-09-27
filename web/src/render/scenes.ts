import { mulberry32, type Vec2 } from '../math';

/**
 * The backdrop behind a fight. Practice always happens in the training yard; a real campaign
 * fight shows the place it happens in; the waves keep the night temple.
 */
export type Scene = 'night' | 'training' | 'courtyard' | 'stairs' | 'bridge' | 'garden' | 'gate' | 'boss';

/** A layer being painted: screen size, margin around it (for parallax), pixels per unit, horizon. */
export interface Frame { g: CanvasRenderingContext2D; W: number; H: number; M: number; u: number; hz: number }

export interface FloorStyle { top: string; bottom: string; line: string; kind: 'grid' | 'tiles' | 'planks' | 'sand' | 'dirt' | 'steps' }

export const FLOORS: Record<Scene, FloorStyle> = {
  night: { top: '#24182b', bottom: '#0a080f', line: 'rgba(255,200,160,.06)', kind: 'grid' },
  training: { top: '#7a5a3c', bottom: '#2a1b10', line: 'rgba(255,225,170,.13)', kind: 'sand' },
  courtyard: { top: '#4a3a4a', bottom: '#140e16', line: 'rgba(255,210,190,.10)', kind: 'tiles' },
  stairs: { top: '#3a3448', bottom: '#0e0c14', line: 'rgba(200,200,255,.10)', kind: 'steps' },
  bridge: { top: '#4a3a26', bottom: '#150f08', line: 'rgba(20,10,4,.55)', kind: 'planks' },
  garden: { top: '#4c4a52', bottom: '#121116', line: 'rgba(230,230,240,.12)', kind: 'sand' },
  gate: { top: '#3e2a1e', bottom: '#120a08', line: 'rgba(255,170,110,.07)', kind: 'dirt' },
  boss: { top: '#40241a', bottom: '#120806', line: 'rgba(255,150,90,.08)', kind: 'dirt' },
};

/** Torches and lanterns: where they glow, and in what colour. */
export interface Glows { pts: Vec2[]; color: [number, number, number] }

const WARM: [number, number, number] = [255, 160, 80];

// ---------- shared pieces ----------

function skyGradient(f: Frame, stops: [number, string][]): void {
  const { g, W, M, hz } = f, gr = g.createLinearGradient(0, -M, 0, hz);
  for (const [o, c] of stops) gr.addColorStop(o, c);
  g.fillStyle = gr;
  g.fillRect(-M, -M, W + 2 * M, hz + M + 1);
}

function stars(f: Frame, n: number, seed: number, alpha = 1): void {
  const { g, W, M, hz } = f, R = mulberry32(seed);
  g.fillStyle = '#fff';
  for (let i = 0; i < n; i++) {
    const x = R() * (W + 2 * M) - M, y = R() * hz * 0.75 - M, r = R() * 1.3 + 0.3;
    g.globalAlpha = (0.2 + R() * 0.8) * alpha;
    g.fillRect(x, y, r, r);
  }
  g.globalAlpha = 1;
}

function moon(f: Frame, x: number, y: number, r: number, glow = 0.28, col = '#f3ead8'): void {
  const { g } = f;
  const gr = g.createRadialGradient(x, y, 0, x, y, r * 6);
  gr.addColorStop(0, `rgba(255,238,215,${glow})`); gr.addColorStop(1, 'rgba(255,238,215,0)');
  g.fillStyle = gr;
  g.fillRect(x - r * 6, y - r * 6, r * 12, r * 12);
  g.fillStyle = col;
  g.beginPath(); g.arc(x, y, r, 0, 7); g.fill();
  g.fillStyle = 'rgba(120,100,90,.12)';
  for (const [a, b, s] of [[-0.3, -0.2, 0.28], [0.25, 0.2, 0.22], [0.1, -0.4, 0.12]]) {
    g.beginPath(); g.arc(x + a * r, y + b * r, s * r, 0, 7); g.fill();
  }
}

/** A mountain ridge silhouette with its base below the horizon. */
function ridge(f: Frame, base: number, amp: number, col: string, seed: number, f1: number, f2: number, sharp = 0): void {
  const { g, W, M, u, hz } = f;
  g.fillStyle = col;
  g.beginPath();
  g.moveTo(-M, hz + 2 * u);
  for (let x = -M; x <= W + M + 8; x += 5) {
    let h = 0.55 + 0.3 * Math.sin(x * f1 + seed) + 0.15 * Math.sin(x * f2 + seed * 3);
    if (sharp) h += sharp * Math.abs(Math.sin(x * f1 * 2.3 + seed * 5)) ** 3;
    g.lineTo(x, base - amp * h);
  }
  g.lineTo(W + M, hz + 2 * u);
  g.closePath();
  g.fill();
}

/** A soft band of mist at height y. */
function mist(f: Frame, y: number, h: number, rgb: string, a: number): void {
  const { g, W, M } = f, gr = g.createLinearGradient(0, y - h, 0, y + h);
  gr.addColorStop(0, `rgba(${rgb},0)`); gr.addColorStop(0.5, `rgba(${rgb},${a})`); gr.addColorStop(1, `rgba(${rgb},0)`);
  g.fillStyle = gr;
  g.fillRect(-M, y - h, W + 2 * M, 2 * h);
}

/** A curved, flared temple roof between l and r with its eave at y. */
function roof(g: CanvasRenderingContext2D, l: number, r: number, y: number, h: number, k: number): void {
  const w = r - l;
  g.beginPath();
  g.moveTo(l - 1.5 * k, y - 2 * k);
  g.quadraticCurveTo(l + w * 0.06, y + 0.6 * k, l + w * 0.16, y + 0.6 * k);
  g.lineTo(r - w * 0.16, y + 0.6 * k);
  g.quadraticCurveTo(r - w * 0.06, y + 0.6 * k, r + 1.5 * k, y - 2 * k);
  g.lineTo(r - w * 0.24, y - h);
  g.lineTo(l + w * 0.24, y - h);
  g.closePath();
  g.fill();
}

/** A pine tree silhouette standing at (x, y), h tall. */
function pine(g: CanvasRenderingContext2D, x: number, y: number, h: number, col: string): void {
  g.fillStyle = col;
  g.fillRect(x - h * 0.03, y - h * 0.25, h * 0.06, h * 0.25);
  for (let i = 0; i < 4; i++) {
    const t = y - h * (0.18 + i * 0.2), w = h * (0.34 - i * 0.07);
    g.beginPath(); g.moveTo(x - w, t); g.lineTo(x, t - h * 0.32); g.lineTo(x + w, t); g.closePath(); g.fill();
  }
}

/** A stalk of bamboo from the ground (y0) up past the top of the screen, with leaves. */
function bamboo(g: CanvasRenderingContext2D, x: number, y0: number, top: number, w: number, lean: number, col: string, leaf: string, R: () => number): void {
  const seg = w * 7;
  g.strokeStyle = col;
  g.lineWidth = w;
  g.lineCap = 'butt';
  for (let y = y0; y > top; y -= seg) {
    const k0 = (y0 - y) / (y0 - top), k1 = (y0 - Math.max(top, y - seg + w * 0.4)) / (y0 - top);
    g.beginPath(); g.moveTo(x + lean * k0, y); g.lineTo(x + lean * k1, Math.max(top, y - seg + w * 0.4)); g.stroke();
    if (R() < 0.35) {
      g.fillStyle = leaf;
      const lx = x + lean * k1, ly = y - seg, s = R() < 0.5 ? -1 : 1;
      for (let j = 0; j < 3; j++) {
        g.beginPath();
        g.ellipse(lx + s * w * (2 + j * 1.2), ly + j * w * 0.8, w * 2.6, w * 0.45, s * (0.4 + j * 0.25), 0, 7);
        g.fill();
      }
    }
  }
}

function banner(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, cloth: string, mark: string, k: number): void {
  g.fillStyle = '#1a120c';
  g.fillRect(x - 0.3 * k, y - 2 * k, 0.6 * k, h + 3 * k);
  g.fillRect(x - w * 0.6, y - 1 * k, w * 1.2, 0.8 * k);
  g.fillStyle = cloth;
  g.beginPath();
  g.moveTo(x - w / 2, y); g.lineTo(x + w / 2, y); g.lineTo(x + w / 2, y + h); g.lineTo(x, y + h - w * 0.35); g.lineTo(x - w / 2, y + h);
  g.closePath(); g.fill();
  g.strokeStyle = mark;
  g.lineWidth = Math.max(1, w * 0.1);
  g.beginPath(); g.arc(x, y + h * 0.35, w * 0.25, 0, 7); g.stroke();
  g.beginPath(); g.moveTo(x - w * 0.25, y + h * 0.35); g.lineTo(x + w * 0.25, y + h * 0.35); g.stroke();
}

function horizonGlow(f: Frame, rgb: string, a: number): void {
  const { g, W, M, u, hz } = f, gr = g.createLinearGradient(0, hz - 10 * u, 0, hz + 3 * u);
  gr.addColorStop(0, `rgba(${rgb},0)`); gr.addColorStop(0.7, `rgba(${rgb},${a})`); gr.addColorStop(1, `rgba(${rgb},0)`);
  g.fillStyle = gr;
  g.fillRect(-M, hz - 10 * u, W + 2 * M, 13 * u);
}

// ---------- skies (far layer) ----------

export function paintSky(f: Frame, scene: Scene): void {
  const { W, H, u, hz } = f;
  switch (scene) {
    case 'night':
      skyGradient(f, [[0, '#06061a'], [0.55, '#151131'], [1, '#3c1d33']]);
      stars(f, 180, 11);
      moon(f, W * 0.8, H * 0.13, 4.2 * u);
      ridge(f, hz - 2 * u, 14 * u, '#231838', 1.3, 0.006, 0.021);
      ridge(f, hz, 9 * u, '#170f27', 4.1, 0.009, 0.03);
      horizonGlow(f, '255,110,80', 0.12);
      break;
    case 'training': {
      // late afternoon: a big low sun behind soft hills
      skyGradient(f, [[0, '#3a2c52'], [0.45, '#9a5a6a'], [0.8, '#e89868'], [1, '#f6c483']]);
      const sx = W * 0.3, sy = hz - 7 * u, sr = 7 * u, g = f.g;
      const gr = g.createRadialGradient(sx, sy, 0, sx, sy, sr * 5);
      gr.addColorStop(0, 'rgba(255,230,170,.55)'); gr.addColorStop(1, 'rgba(255,200,140,0)');
      g.fillStyle = gr; g.fillRect(sx - sr * 5, sy - sr * 5, sr * 10, sr * 10);
      g.fillStyle = '#ffe6b0'; g.beginPath(); g.arc(sx, sy, sr, 0, 7); g.fill();
      g.fillStyle = 'rgba(255,240,220,.35)';
      for (const [cx, cy, w] of [[0.62, 0.12, 16], [0.78, 0.2, 10], [0.12, 0.18, 12]]) {
        g.beginPath(); g.ellipse(W * cx, H * cy, w * u, 1.4 * u, 0, 0, 7); g.fill();
        g.beginPath(); g.ellipse(W * cx + 4 * u, H * cy - 1.2 * u, w * 0.5 * u, 1.3 * u, 0, 0, 7); g.fill();
      }
      ridge(f, hz - 3 * u, 12 * u, '#8a5e70', 2.2, 0.005, 0.017);
      ridge(f, hz, 7 * u, '#5e4054', 5.3, 0.008, 0.026);
      break;
    }
    case 'courtyard':
      // dawn over the temple
      skyGradient(f, [[0, '#15163a'], [0.5, '#4c2e5c'], [0.85, '#c8606a'], [1, '#f4a070']]);
      stars(f, 60, 21, 0.5);
      moon(f, W * 0.82, H * 0.1, 3.4 * u, 0.15, '#f5e6e0');
      ridge(f, hz - 2 * u, 16 * u, '#43294e', 3.3, 0.005, 0.02, 0.4);
      ridge(f, hz, 8 * u, '#2a1a36', 1.1, 0.01, 0.03);
      horizonGlow(f, '255,150,100', 0.25);
      break;
    case 'stairs':
      skyGradient(f, [[0, '#050818'], [0.6, '#16183a'], [1, '#2e2448']]);
      stars(f, 220, 31);
      moon(f, W * 0.62, H * 0.1, 3.8 * u);
      // sharp peaks all around: you are high on the mountain
      ridge(f, hz - 4 * u, 17 * u, '#1e1d3a', 7.7, 0.006, 0.019, 0.6);
      ridge(f, hz - 1 * u, 12 * u, '#15142a', 2.9, 0.008, 0.024, 0.5);
      mist(f, hz - 1 * u, 3 * u, '120,130,190', 0.35);
      break;
    case 'bridge':
      skyGradient(f, [[0, '#061020'], [0.6, '#12233a'], [1, '#2a3a50']]);
      stars(f, 160, 41);
      moon(f, W * 0.24, H * 0.12, 4 * u, 0.32);
      ridge(f, hz - 3 * u, 15 * u, '#162338', 4.4, 0.006, 0.02, 0.4);
      // the valley below the bridge is full of mist
      mist(f, hz, 5 * u, '150,180,210', 0.5);
      break;
    case 'garden':
      skyGradient(f, [[0, '#0a0c18'], [0.6, '#1c2236'], [1, '#3a3a4e']]);
      stars(f, 140, 51);
      moon(f, W * 0.72, H * 0.14, 6 * u, 0.35);
      ridge(f, hz - 2 * u, 11 * u, '#232a3c', 6.1, 0.006, 0.02);
      ridge(f, hz, 6 * u, '#171b28', 2.4, 0.01, 0.03);
      break;
    case 'gate':
    case 'boss': {
      // the sky over the village, lit from below by the earth clan's fires
      const boss = scene === 'boss';
      skyGradient(f, boss
        ? [[0, '#12060a'], [0.5, '#3a1216'], [0.85, '#8a2a1a'], [1, '#d0582a']]
        : [[0, '#0c0814'], [0.55, '#2e1626'], [0.9, '#7a3024'], [1, '#b8542e']]);
      stars(f, 60, 61, 0.4);
      const g = f.g;
      g.fillStyle = boss ? 'rgba(40,10,10,.5)' : 'rgba(40,20,26,.5)';
      for (let i = 0; i < 7; i++) {
        const x = W * (0.1 + i * 0.14), y = hz - (12 + (i % 3) * 5) * u;
        g.beginPath(); g.ellipse(x, y, 14 * u, 3 * u, 0.1, 0, 7); g.fill();
      }
      ridge(f, hz - 2 * u, 12 * u, boss ? '#2a1014' : '#26141e', 8.8, 0.006, 0.02, 0.3);
      horizonGlow(f, '255,110,50', boss ? 0.4 : 0.28);
      break;
    }
  }
}

// ---------- the middle layer: what stands around the arena ----------

export function paintMid(f: Frame, scene: Scene): Glows {
  switch (scene) {
    case 'night': return midNight(f);
    case 'training': return midTraining(f);
    case 'courtyard': return midCourtyard(f);
    case 'stairs': return midStairs(f);
    case 'bridge': return midBridge(f);
    case 'garden': return midGarden(f);
    case 'gate': case 'boss': return midGate(f, scene === 'boss');
  }
}

function midNight(f: Frame): Glows {
  const { g, W, M, u, hz } = f, k = u * 0.8, pts: Vec2[] = [];
  g.fillStyle = '#110c19';
  g.fillRect(-M, hz - 2.2 * u, W + 2 * M, 2.6 * u);
  for (let x = -M; x < W + M; x += 9 * u) g.fillRect(x, hz - 3.4 * u, 1.6 * u, 1.4 * u);
  const temple = (x: number, w: number) => {
    const base = hz - u, bodyH = 12 * k, b1 = base - 3 * k - bodyH, b2 = b1 - 12 * k;
    g.fillStyle = '#0e0a16';
    g.fillRect(x - w * 0.04, base - 3 * k, w * 1.08, 3 * k);
    g.fillRect(x + w * 0.1, b1, w * 0.8, bodyH);
    roof(g, x - w * 0.02, x + w * 1.02, b1, 5 * k, k);
    g.fillRect(x + w * 0.3, b2, w * 0.4, 7 * k);
    roof(g, x + w * 0.18, x + w * 0.82, b2, 4.5 * k, k);
    g.fillStyle = 'rgba(255,165,90,.22)';
    for (let i = 0; i < 3; i++) g.fillRect(x + w * (0.22 + i * 0.22), b1 + bodyH * 0.35, w * 0.1, bodyH * 0.4);
    g.fillRect(x + w * 0.44, b2 + 2 * k, w * 0.12, 3.5 * k);
    pts.push({ x: x + w * 0.12, y: b1 + 2.4 * k }, { x: x + w * 0.88, y: b1 + 2.4 * k });
  };
  temple(W * 0.02, W * 0.22);
  temple(W * 0.76, W * 0.22);
  return { pts, color: WARM };
}

/** The training yard: a sandy ring with a wooden fence, racks, straw targets and a dojo. */
function midTraining(f: Frame): Glows {
  const { g, W, M, u, hz } = f, pts: Vec2[] = [], wood = '#4a3020', dark = '#2e1c12';
  // dojo hall on the left, low and wide
  const dx = W * 0.02, dw = W * 0.26, base = hz + 0.5 * u, body = 9 * u;
  g.fillStyle = dark; g.fillRect(dx, base - body, dw, body);
  g.fillStyle = '#6a2a22'; roof(g, dx - 2 * u, dx + dw + 2 * u, base - body, 6 * u, u);
  g.fillStyle = 'rgba(255,200,130,.35)';
  for (let i = 0; i < 4; i++) g.fillRect(dx + dw * (0.1 + i * 0.22), base - body * 0.8, dw * 0.12, body * 0.55);
  g.strokeStyle = dark; g.lineWidth = 0.3 * u;
  for (let i = 0; i < 4; i++) { const x = dx + dw * (0.16 + i * 0.22); g.beginPath(); g.moveTo(x, base - body * 0.8); g.lineTo(x, base - body * 0.25); g.stroke(); }
  pts.push({ x: dx + dw * 0.05, y: base - body * 0.9 }, { x: dx + dw * 0.95, y: base - body * 0.9 });
  // the fence all the way round the yard
  g.fillStyle = wood;
  g.fillRect(-M, hz - 3.2 * u, W + 2 * M, 0.7 * u);
  g.fillRect(-M, hz - 1.4 * u, W + 2 * M, 0.7 * u);
  for (let x = -M; x < W + M; x += 6 * u) g.fillRect(x, hz - 4.4 * u, 0.9 * u, 5 * u);
  // weapon rack with staffs, right of centre
  const rx = W * 0.62, ry = hz + 0.2 * u;
  g.fillStyle = dark;
  g.fillRect(rx, ry - 7 * u, 0.7 * u, 7 * u); g.fillRect(rx + 9 * u, ry - 7 * u, 0.7 * u, 7 * u);
  g.fillRect(rx - 0.5 * u, ry - 6 * u, 10.7 * u, 0.6 * u);
  g.strokeStyle = '#8a6a44'; g.lineWidth = 0.35 * u;
  for (let i = 0; i < 6; i++) { const x = rx + (1.2 + i * 1.4) * u; g.beginPath(); g.moveTo(x, ry - 9.5 * u); g.lineTo(x + 0.3 * u, ry); g.stroke(); }
  // straw target bales with painted rings
  for (const [x, s] of [[W * 0.84, 1], [W * 0.93, 0.8], [W * 0.46, 0.7]] as const) {
    const r = 3.2 * u * s, y = hz - r * 0.2;
    g.fillStyle = '#b89a5a'; g.beginPath(); g.ellipse(x, y, r, r * 1.05, 0, 0, 7); g.fill();
    g.strokeStyle = '#8a6a3a'; g.lineWidth = 0.2 * u;
    for (let i = 1; i < 4; i++) { g.beginPath(); g.ellipse(x, y, r * i / 4, r * i / 4 * 1.05, 0, 0, 7); g.stroke(); }
    g.fillStyle = '#b03a2a'; g.beginPath(); g.arc(x, y, r * 0.22, 0, 7); g.fill();
    g.fillStyle = dark; g.fillRect(x - r * 0.9, y + r * 0.8, 0.5 * u, r * 0.6); g.fillRect(x + r * 0.9 - 0.5 * u, y + r * 0.8, 0.5 * u, r * 0.6);
  }
  // red practice banners
  banner(g, W * 0.36, hz - 13 * u, 2.6 * u, 8 * u, '#a8302a', '#f0c070', u);
  banner(g, W * 0.72, hz - 13 * u, 2.6 * u, 8 * u, '#a8302a', '#f0c070', u);
  return { pts, color: [255, 190, 120] };
}

/** The temple courtyard: the great hall, red gate pillars, stone lanterns and braziers. */
function midCourtyard(f: Frame): Glows {
  const { g, W, M, u, hz } = f, pts: Vec2[] = [];
  // courtyard wall
  g.fillStyle = '#2a1a26';
  g.fillRect(-M, hz - 4 * u, W + 2 * M, 4.5 * u);
  g.fillStyle = '#4a2a30';
  g.fillRect(-M, hz - 4.6 * u, W + 2 * M, 0.8 * u);
  // the great hall, centred behind the arena
  const cx = W / 2, hw = W * 0.2, base = hz - 3.5 * u, body = 10 * u;
  g.fillStyle = '#3a1a22';
  g.fillRect(cx - hw * 0.8, base - body, hw * 1.6, body);
  g.fillStyle = '#8a2a24';
  for (let i = 0; i < 6; i++) g.fillRect(cx - hw * 0.75 + i * hw * 0.29, base - body, 0.9 * u, body);
  g.fillStyle = 'rgba(255,190,110,.4)';
  g.fillRect(cx - hw * 0.12, base - body * 0.75, hw * 0.24, body * 0.75);
  g.fillStyle = '#241018';
  roof(g, cx - hw, cx + hw, base - body, 7 * u, u);
  g.fillRect(cx - hw * 0.45, base - body - 12 * u, hw * 0.9, 5.5 * u);
  roof(g, cx - hw * 0.62, cx + hw * 0.62, base - body - 6.5 * u, 6 * u, u);
  g.fillStyle = '#e0a040';
  g.beginPath(); g.arc(cx, base - body - 14.5 * u, 0.9 * u, 0, 7); g.fill();
  pts.push({ x: cx - hw * 0.55, y: base - body + 2 * u }, { x: cx + hw * 0.55, y: base - body + 2 * u });
  // red gate pillars framing the arena
  for (const x of [W * 0.12, W * 0.88]) {
    g.fillStyle = '#9a2a22';
    g.fillRect(x - 5 * u, hz - 20 * u, 1.6 * u, 21 * u);
    g.fillRect(x + 3.4 * u, hz - 20 * u, 1.6 * u, 21 * u);
    g.fillStyle = '#1e1016';
    g.fillRect(x - 7 * u, hz - 21.5 * u, 14 * u, 1.6 * u);
    g.fillStyle = '#9a2a22';
    g.fillRect(x - 6 * u, hz - 17 * u, 12 * u, 1 * u);
  }
  // stone lanterns and braziers
  for (const x of [W * 0.28, W * 0.72]) {
    g.fillStyle = '#5a5058';
    g.fillRect(x - 0.6 * u, hz - 4 * u, 1.2 * u, 4.5 * u);
    g.fillRect(x - 1.8 * u, hz - 6.5 * u, 3.6 * u, 2.5 * u);
    g.beginPath(); g.moveTo(x - 2.6 * u, hz - 6.5 * u); g.lineTo(x, hz - 8.6 * u); g.lineTo(x + 2.6 * u, hz - 6.5 * u); g.fill();
    g.fillStyle = 'rgba(255,190,110,.8)'; g.fillRect(x - 0.9 * u, hz - 6 * u, 1.8 * u, 1.4 * u);
    pts.push({ x, y: hz - 5.3 * u });
  }
  return { pts, color: [255, 170, 90] };
}

/** The long stairs: cliff faces on both sides, pines clinging on, lantern posts down the steps. */
function midStairs(f: Frame): Glows {
  const { g, W, H, M, u, hz } = f, pts: Vec2[] = [], R = mulberry32(33);
  // the steps climbing away in the distance, between the cliffs
  g.fillStyle = '#2a2638';
  g.beginPath(); g.moveTo(W * 0.42, hz + 0.3 * u); g.lineTo(W * 0.47, hz - 9 * u); g.lineTo(W * 0.53, hz - 9 * u); g.lineTo(W * 0.58, hz + 0.3 * u); g.fill();
  g.strokeStyle = 'rgba(0,0,0,.35)'; g.lineWidth = 0.2 * u;
  for (let i = 1; i < 10; i++) { const y = hz + 0.3 * u - i * 0.95 * u, t = i / 10; g.beginPath(); g.moveTo(W * (0.42 + 0.05 * t), y); g.lineTo(W * (0.58 - 0.05 * t), y); g.stroke(); }
  // cliffs
  for (const side of [-1, 1]) {
    const edge = side < 0 ? W * 0.2 : W * 0.8, out = side < 0 ? -M : W + M;
    g.fillStyle = '#18162a';
    g.beginPath();
    g.moveTo(out, -M);
    let x = edge;
    for (let y = -M; y < H * 0.62; y += 3 * u) { x = edge + side * (R() * 4 - 2) * u - side * (y / H) * 6 * u; g.lineTo(x, y); }
    g.lineTo(out, H * 0.62);
    g.closePath(); g.fill();
    g.strokeStyle = 'rgba(140,140,200,.12)'; g.lineWidth = 0.3 * u;
    for (let i = 0; i < 9; i++) {
      const y = R() * H * 0.55, x0 = edge + side * R() * 12 * u;
      g.beginPath(); g.moveTo(x0, y); g.lineTo(x0 + side * (4 + R() * 8) * u, y + R() * 3 * u); g.stroke();
    }
    for (let i = 0; i < 3; i++) pine(g, edge + side * (2 + i * 6) * u, H * (0.12 + i * 0.13), (9 + R() * 5) * u, '#0e0d1c');
  }
  // lantern posts either side of the path
  for (const x of [W * 0.3, W * 0.7, W * 0.4, W * 0.6]) {
    const near = Math.abs(x - W / 2) > W * 0.15, h = near ? 7 * u : 4 * u, y = near ? hz + 1 * u : hz - 1 * u;
    g.fillStyle = '#100e1a'; g.fillRect(x - 0.3 * u, y - h, 0.6 * u, h);
    g.fillStyle = 'rgba(255,180,100,.85)'; g.fillRect(x - 0.7 * u, y - h - 1.2 * u, 1.4 * u, 1.4 * u);
    pts.push({ x, y: y - h - 0.5 * u });
  }
  return { pts, color: WARM };
}

/** The bamboo bridge: bamboo walls both sides, rope rails, and mist far below. */
function midBridge(f: Frame): Glows {
  const { g, W, H, M, u, hz } = f, pts: Vec2[] = [], R = mulberry32(44);
  // the far end of the bridge and its posts
  g.fillStyle = '#2a2014';
  g.fillRect(W * 0.44, hz - 0.8 * u, W * 0.12, 1 * u);
  g.fillRect(W * 0.44, hz - 5 * u, 0.6 * u, 4.5 * u); g.fillRect(W * 0.56 - 0.6 * u, hz - 5 * u, 0.6 * u, 4.5 * u);
  // rope rails running from the far end out to either side of you
  g.strokeStyle = '#6a5436'; g.lineWidth = 0.35 * u;
  for (const side of [-1, 1]) {
    const fx = W / 2 + side * W * 0.06;
    for (const [fy, ny] of [[hz - 4.5 * u, H * 0.52], [hz - 2.5 * u, H * 0.66]]) {
      g.beginPath(); g.moveTo(fx, fy); g.quadraticCurveTo(W / 2 + side * W * 0.3, fy + 3 * u, W / 2 + side * (W / 2 + M), ny); g.stroke();
    }
  }
  // bamboo groves on both banks, darker and bigger nearer the edges
  for (const side of [-1, 1]) {
    for (let i = 0; i < 16; i++) {
      const t = R(), x = W / 2 + side * (W * 0.2 + t * (W * 0.34 + M)), near = t;
      const w = (0.5 + near * 1.3) * u, y0 = hz + (near * 8 + R() * 2) * u;
      const col = near > 0.6 ? '#1e3020' : '#2c4430', leaf = near > 0.6 ? '#1a2a1c' : '#2a3e2c';
      bamboo(g, x, y0, -M, w, side * R() * 3 * u, col, leaf, R);
    }
  }
  // paper lanterns hung on the bridge posts
  pts.push({ x: W * 0.44, y: hz - 5.5 * u }, { x: W * 0.56, y: hz - 5.5 * u });
  g.fillStyle = 'rgba(255,200,120,.9)';
  for (const p of pts) { g.beginPath(); g.ellipse(p.x, p.y, 0.6 * u, 0.8 * u, 0, 0, 7); g.fill(); }
  return { pts, color: [255, 200, 130] };
}

/** The stone garden: boulders standing in raked sand, a twisted pine, stone lanterns. */
function midGarden(f: Frame): Glows {
  const { g, W, M, u, hz } = f, pts: Vec2[] = [];
  // low garden wall with a tiled cap
  g.fillStyle = '#26242e'; g.fillRect(-M, hz - 3.4 * u, W + 2 * M, 3.8 * u);
  g.fillStyle = '#16151c'; g.fillRect(-M, hz - 4.2 * u, W + 2 * M, 1 * u);
  // boulders
  const rock = (x: number, y: number, w: number, h: number) => {
    g.fillStyle = '#4a4852';
    g.beginPath();
    g.moveTo(x - w, y);
    g.bezierCurveTo(x - w * 1.05, y - h * 0.7, x - w * 0.3, y - h * 1.05, x + w * 0.1, y - h);
    g.bezierCurveTo(x + w * 0.7, y - h * 0.95, x + w * 1.05, y - h * 0.4, x + w, y);
    g.closePath(); g.fill();
    g.fillStyle = 'rgba(200,210,240,.14)';
    g.beginPath(); g.ellipse(x - w * 0.3, y - h * 0.7, w * 0.35, h * 0.2, -0.4, 0, 7); g.fill();
    g.strokeStyle = 'rgba(220,220,235,.18)'; g.lineWidth = 0.2 * u;
    for (let i = 1; i <= 3; i++) { g.beginPath(); g.ellipse(x, y, w * (1 + i * 0.35), h * 0.12 * (1 + i * 0.4), 0, Math.PI, 0); g.stroke(); }
  };
  rock(W * 0.16, hz + 1 * u, 5 * u, 6 * u);
  rock(W * 0.3, hz, 2.5 * u, 3 * u);
  rock(W * 0.78, hz + 1 * u, 6 * u, 8 * u);
  rock(W * 0.9, hz + 0.5 * u, 3 * u, 3.5 * u);
  // a twisted pine leaning over the wall
  const px = W * 0.64, py = hz - 3 * u;
  g.strokeStyle = '#141218'; g.lineWidth = 1.4 * u; g.lineCap = 'round';
  g.beginPath(); g.moveTo(px, py); g.bezierCurveTo(px - 3 * u, py - 6 * u, px + 4 * u, py - 9 * u, px - 2 * u, py - 14 * u); g.stroke();
  g.lineWidth = 0.7 * u;
  g.beginPath(); g.moveTo(px - 1 * u, py - 8 * u); g.quadraticCurveTo(px + 5 * u, py - 10 * u, px + 9 * u, py - 9 * u); g.stroke();
  g.fillStyle = '#1a2622';
  for (const [x, y, w] of [[-2, -15, 5], [2, -12, 4], [9, -9.5, 4.5], [-5, -11, 3.5]]) {
    g.beginPath(); g.ellipse(px + x * u, py + y * u, w * u, 1.4 * u, 0, 0, 7); g.fill();
  }
  // stone lanterns
  for (const x of [W * 0.42, W * 0.56]) {
    g.fillStyle = '#5a5862';
    g.fillRect(x - 0.4 * u, hz - 3 * u, 0.8 * u, 3 * u);
    g.fillRect(x - 1.2 * u, hz - 5 * u, 2.4 * u, 2 * u);
    g.beginPath(); g.moveTo(x - 1.8 * u, hz - 5 * u); g.lineTo(x, hz - 6.6 * u); g.lineTo(x + 1.8 * u, hz - 5 * u); g.fill();
    g.fillStyle = 'rgba(255,200,120,.8)'; g.fillRect(x - 0.6 * u, hz - 4.6 * u, 1.2 * u, 1.1 * u);
    pts.push({ x, y: hz - 4 * u });
  }
  return { pts, color: [255, 190, 110] };
}

/** The village gate: a timber palisade, gate towers with torches, earth-clan banners. */
function midGate(f: Frame, boss: boolean): Glows {
  const { g, W, M, u, hz } = f, pts: Vec2[] = [], R = mulberry32(boss ? 77 : 66);
  // village roofs peeking over the wall
  g.fillStyle = '#1a0e10';
  for (let i = 0; i < 6; i++) {
    const x = W * (0.08 + i * 0.17) + R() * 3 * u;
    roof(g, x - 4 * u, x + 4 * u, hz - 9 * u - R() * 2 * u, 3 * u, u * 0.6);
  }
  // palisade of sharpened logs
  for (let x = -M; x < W + M; x += 1.8 * u) {
    const h = (8 + R() * 1.2) * u;
    g.fillStyle = R() < 0.5 ? '#3a2416' : '#301c10';
    g.beginPath(); g.moveTo(x, hz + 0.5 * u); g.lineTo(x, hz - h); g.lineTo(x + 0.9 * u, hz - h - 1.4 * u); g.lineTo(x + 1.8 * u, hz - h); g.lineTo(x + 1.8 * u, hz + 0.5 * u); g.fill();
  }
  g.fillStyle = '#20140c';
  g.fillRect(-M, hz - 6 * u, W + 2 * M, 0.6 * u); g.fillRect(-M, hz - 2.5 * u, W + 2 * M, 0.6 * u);
  // the gate itself, behind the arena
  const gx = W / 2, gw = 9 * u;
  g.fillStyle = '#1c100a'; g.fillRect(gx - gw, hz - 10 * u, gw * 2, 10.5 * u);
  g.fillStyle = '#4a2c18'; g.fillRect(gx - gw * 0.9, hz - 9 * u, gw * 0.88, 9.5 * u); g.fillRect(gx + gw * 0.02, hz - 9 * u, gw * 0.88, 9.5 * u);
  g.fillStyle = '#20140c';
  for (const y of [-7, -3]) g.fillRect(gx - gw * 0.9, hz + y * u, gw * 1.8, 0.7 * u);
  // gate towers with torches
  for (const x of [gx - gw - 3 * u, gx + gw + 3 * u, W * 0.08, W * 0.92]) {
    const big = Math.abs(x - gx) < W * 0.3, h = big ? 16 * u : 12 * u;
    g.fillStyle = '#26160c';
    g.fillRect(x - 2.2 * u, hz - h, 4.4 * u, h + 0.5 * u);
    g.fillStyle = '#140a06';
    roof(g, x - 3.2 * u, x + 3.2 * u, hz - h, 2.6 * u, u * 0.6);
    g.fillStyle = '#ffb060';
    g.beginPath(); g.ellipse(x, hz - h + 2.2 * u, 0.6 * u, 1 * u, 0, 0, 7); g.fill();
    pts.push({ x, y: hz - h + 2 * u });
  }
  // earth clan banners: green with a mountain sigil
  const cloth = boss ? '#3e5a2a' : '#34502a';
  for (const x of boss ? [W * 0.22, W * 0.34, W * 0.66, W * 0.78] : [W * 0.28, W * 0.72]) {
    banner(g, x, hz - 17 * u, 2.8 * u, 10 * u, cloth, '#c8a860', u);
  }
  if (boss) {
    // boulders he has already torn up
    for (const [x, r] of [[0.12, 4], [0.86, 5], [0.95, 3]]) {
      g.fillStyle = '#2c2018'; g.strokeStyle = '#120a06'; g.lineWidth = 0.3 * u;
      g.beginPath(); g.ellipse(W * x, hz + 1 * u, r * u, r * 0.8 * u, 0, Math.PI, 0); g.closePath(); g.fill(); g.stroke();
      g.fillStyle = 'rgba(255,150,90,.15)';
      g.beginPath(); g.ellipse(W * x - r * 0.3 * u, hz + 1 * u - r * 0.5 * u, r * 0.4 * u, r * 0.15 * u, -0.3, 0, 7); g.fill();
    }
  }
  return { pts, color: [255, 140, 60] };
}
