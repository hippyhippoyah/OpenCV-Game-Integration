import { STOPS } from '../campaign/chapter1';
import type { ScrollId } from '../campaign/progress';
import { mulberry32 } from '../math';
import { MAP_H, MAP_W, pointAt, type MapPt } from './path';

/** What the map needs to know each frame. */
export interface MapView {
  /** Where you are along the path. */
  d: number;
  walking: boolean;
  /** Standing at something to do: a scroll or an arena. */
  waiting: boolean;
  isDone: (stop: number) => boolean;
  flames: (stop: number) => number;
  hasScroll: (id: ScrollId) => boolean;
  /** The first stop not yet done (or -1). */
  next: number;
}

const INK = '#3a2616', INK_SOFT = 'rgba(58,38,22,.55)', RED = '#b3321f', GOLD = '#f2b640';
const PAPER = '#ecdcb6';

/**
 * The campaign's explore view: an ink-and-parchment map of the mountain, the path drawn on it, the
 * stops as lanterns, and you — a flame token walking the path. The land is painted once (per
 * resize); the path, stops and you every frame.
 */
export class PathMap {
  private ctx: CanvasRenderingContext2D;
  private land = document.createElement('canvas');
  private dpr = 1;
  private W = 0;
  private H = 0;
  /** Map space → screen: scale and offset. */
  private s = 1;
  private ox = 0;
  private oy = 0;
  private t = 0;
  private trail: MapPt[] = [];

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    this.resize();
  }

  resize(): void {
    this.dpr = Math.min(devicePixelRatio || 1, 2);
    this.W = innerWidth;
    this.H = innerHeight;
    for (const c of [this.canvas, this.land]) {
      c.width = Math.round(this.W * this.dpr);
      c.height = Math.round(this.H * this.dpr);
    }
    // the whole map fits, leaving room for the controls and subtitles below
    this.s = Math.min(this.W / (MAP_W + 60), (this.H - 40) / (MAP_H + 60));
    this.ox = (this.W - MAP_W * this.s) / 2;
    this.oy = Math.max(10, (this.H - MAP_H * this.s) / 2 - 20);
    this.paintLand();
  }

  /** The stop whose lantern is under a screen point, if any. */
  stopAt(clientX: number, clientY: number): number | null {
    const r = this.canvas.getBoundingClientRect();
    const x = (clientX - r.left - this.ox) / this.s, y = (clientY - r.top - this.oy) / this.s;
    const i = STOPS.findIndex(st => { const p = pointAt(st.pathAt); return Math.hypot(p.x - x, p.y - y) < 34; });
    return i < 0 ? null : i;
  }

  render(v: MapView, dt: number): void {
    this.t += dt;
    const c = this.ctx;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.drawImage(this.land, 0, 0, this.W, this.H);
    c.setTransform(this.dpr * this.s, 0, 0, this.dpr * this.s, this.dpr * this.ox, this.dpr * this.oy);
    this.drawPath(c, v);
    this.drawScrolls(c, v);
    this.drawStops(c, v);
    this.drawYou(c, v);
  }

  // ---------- the land (painted once) ----------

  private paintLand(): void {
    const g = this.land.getContext('2d')!, { W, H } = this;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    // the table the map lies on
    let gr = g.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.75);
    gr.addColorStop(0, '#2a1a14'); gr.addColorStop(1, '#0c0706');
    g.fillStyle = gr;
    g.fillRect(0, 0, W, H);
    g.setTransform(this.dpr * this.s, 0, 0, this.dpr * this.s, this.dpr * this.ox, this.dpr * this.oy);
    const R = mulberry32(7);
    // parchment with a soft ragged edge
    g.save();
    g.shadowColor = 'rgba(0,0,0,.6)'; g.shadowBlur = 40 * this.s * 3;
    g.fillStyle = PAPER;
    g.beginPath();
    const edge = (x: number, y: number) => g.lineTo(x + (R() - 0.5) * 8, y + (R() - 0.5) * 8);
    g.moveTo(0, 0);
    for (let x = 0; x <= MAP_W; x += 40) edge(x, 0);
    for (let y = 0; y <= MAP_H; y += 40) edge(MAP_W, y);
    for (let x = MAP_W; x >= 0; x -= 40) edge(x, MAP_H);
    for (let y = MAP_H; y >= 0; y -= 40) edge(0, y);
    g.closePath(); g.fill();
    g.restore();
    g.save();
    g.clip();
    gr = g.createRadialGradient(MAP_W / 2, MAP_H / 2, MAP_H * 0.3, MAP_W / 2, MAP_H / 2, MAP_W * 0.62);
    gr.addColorStop(0, 'rgba(255,245,220,.25)'); gr.addColorStop(1, 'rgba(120,80,30,.45)');
    g.fillStyle = gr; g.fillRect(0, 0, MAP_W, MAP_H);
    for (let i = 0; i < 2200; i++) {
      g.fillStyle = R() < 0.5 ? 'rgba(120,85,40,.07)' : 'rgba(255,250,235,.08)';
      g.fillRect(R() * MAP_W, R() * MAP_H, 1 + R() * 3, 1 + R() * 3);
    }
    for (let i = 0; i < 6; i++) {
      g.fillStyle = 'rgba(140,95,40,.06)';
      g.beginPath(); g.ellipse(R() * MAP_W, R() * MAP_H, 60 + R() * 120, 40 + R() * 80, R() * 3, 0, 7); g.fill();
    }

    this.paintRiver(g);
    this.paintMountains(g, R);
    this.paintForests(g, R);
    this.paintGarden(g);
    this.paintVillage(g, R);
    this.paintTemple(g);
    this.paintBridge(g);
    this.paintStairs(g);
    this.paintClouds(g);
    this.paintCompass(g, 110, 880);
    this.paintTitle(g);
    g.restore();
    // a double ink border
    g.strokeStyle = INK; g.lineWidth = 4; g.strokeRect(18, 18, MAP_W - 36, MAP_H - 36);
    g.lineWidth = 1.5; g.strokeRect(28, 28, MAP_W - 56, MAP_H - 56);
    for (const [x, y] of [[28, 28], [MAP_W - 28, 28], [28, MAP_H - 28], [MAP_W - 28, MAP_H - 28]]) {
      g.fillStyle = RED; g.beginPath(); g.arc(x, y, 7, 0, 7); g.fill();
      g.strokeStyle = INK; g.lineWidth = 2; g.stroke();
    }
  }

  private paintRiver(g: CanvasRenderingContext2D): void {
    const pts: [number, number][] = [[1620, 120], [1420, 190], [1260, 300], [1080, 350], [900, 420], [760, 500], [690, 548], [610, 640], [470, 700], [330, 790], [210, 900], [150, 1020]];
    const path = () => {
      g.beginPath();
      g.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length - 1; i++) {
        const mx = (pts[i][0] + pts[i + 1][0]) / 2, my = (pts[i][1] + pts[i + 1][1]) / 2;
        g.quadraticCurveTo(pts[i][0], pts[i][1], mx, my);
      }
      g.lineTo(pts[pts.length - 1][0], pts[pts.length - 1][1]);
    };
    g.lineCap = 'round'; g.lineJoin = 'round';
    path(); g.strokeStyle = 'rgba(58,38,22,.7)'; g.lineWidth = 42; g.stroke();
    path(); g.strokeStyle = '#7fa6b4'; g.lineWidth = 36; g.stroke();
    path(); g.strokeStyle = '#a4c4cc'; g.lineWidth = 18; g.stroke();
    g.setLineDash([14, 22]);
    path(); g.strokeStyle = 'rgba(255,255,255,.55)'; g.lineWidth = 2; g.stroke();
    g.setLineDash([]);
    g.font = 'italic 600 20px Cinzel, serif';
    g.fillStyle = 'rgba(40,70,90,.8)';
    g.save(); g.translate(1150, 318); g.rotate(-0.3); g.fillText('River Tsui', 0, 0); g.restore();
  }

  /** An inked mountain: a light face, a hatched shadow face, a snow cap. */
  private mountain(g: CanvasRenderingContext2D, x: number, base: number, w: number, h: number, R: () => number): void {
    const peak = { x: x + (R() - 0.5) * w * 0.2, y: base - h };
    g.fillStyle = '#d9c49a';
    g.beginPath(); g.moveTo(x - w / 2, base); g.lineTo(peak.x, peak.y); g.lineTo(x + w / 2, base); g.closePath(); g.fill();
    g.fillStyle = '#b89c6c';
    g.beginPath(); g.moveTo(peak.x, peak.y); g.lineTo(x + w / 2, base); g.lineTo(peak.x + w * 0.05, base); g.closePath(); g.fill();
    g.strokeStyle = 'rgba(58,38,22,.45)'; g.lineWidth = 1.2;
    for (let i = 1; i < 9; i++) {
      const k = i / 9, sx = peak.x + (w / 2) * k * 0.95, sy = peak.y + h * k;
      g.beginPath(); g.moveTo(sx, sy); g.lineTo(sx - w * 0.12 * (1 - k * 0.5), sy + h * 0.12); g.stroke();
    }
    g.fillStyle = '#f7f1e4';
    g.beginPath();
    g.moveTo(peak.x, peak.y);
    g.lineTo(peak.x - w * 0.12, peak.y + h * 0.22); g.lineTo(peak.x - w * 0.04, peak.y + h * 0.17);
    g.lineTo(peak.x + w * 0.02, peak.y + h * 0.25); g.lineTo(peak.x + w * 0.1, peak.y + h * 0.2);
    g.closePath(); g.fill();
    g.strokeStyle = INK; g.lineWidth = 2.2; g.lineJoin = 'round';
    g.beginPath(); g.moveTo(x - w / 2, base); g.lineTo(peak.x, peak.y); g.lineTo(x + w / 2, base); g.stroke();
  }

  private paintMountains(g: CanvasRenderingContext2D, R: () => number): void {
    // back to front so nearer peaks overlap farther ones
    const ms: [number, number, number, number][] = [
      [60, 150, 200, 150], [330, 120, 180, 110], [470, 150, 200, 130], [640, 130, 220, 110], [820, 150, 180, 120],
      [980, 200, 200, 120], [1560, 330, 190, 130], [1470, 440, 220, 150], [1560, 560, 160, 110],
      [60, 470, 170, 120], [180, 560, 180, 110], [40, 650, 150, 90],
      [240, 150, 150, 90], [720, 240, 140, 80], [380, 440, 150, 90],
    ];
    ms.sort((a, b) => a[1] - b[1]).forEach(([x, b, w, h]) => this.mountain(g, x, b, w, h, R));
    // the temple's own peak
    this.mountain(g, 150, 250, 300, 190, R);
    // cliffs along the stairs
    g.strokeStyle = INK_SOFT; g.lineWidth = 1.5;
    for (let i = 0; i < 18; i++) {
      const y = 300 + i * 9, x = 600 + Math.sin(i) * 6;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + 16, y + 8); g.stroke();
    }
  }

  private tree(g: CanvasRenderingContext2D, x: number, y: number, s: number): void {
    g.fillStyle = '#5e7a4a'; g.strokeStyle = INK; g.lineWidth = 1.3;
    g.beginPath(); g.moveTo(x - 8 * s, y); g.lineTo(x, y - 22 * s); g.lineTo(x + 8 * s, y); g.closePath(); g.fill(); g.stroke();
    g.beginPath(); g.moveTo(x, y); g.lineTo(x, y + 4 * s); g.stroke();
  }

  private bambooClump(g: CanvasRenderingContext2D, x: number, y: number, R: () => number): void {
    g.lineCap = 'round';
    for (let i = 0; i < 7; i++) {
      const bx = x + (R() - 0.5) * 30, h = 26 + R() * 18;
      g.strokeStyle = '#6f8a4c'; g.lineWidth = 2.6;
      g.beginPath(); g.moveTo(bx, y); g.lineTo(bx + (R() - 0.5) * 6, y - h); g.stroke();
      g.strokeStyle = INK; g.lineWidth = 1;
      for (let k = 8; k < h; k += 8) { g.beginPath(); g.moveTo(bx - 1.5, y - k); g.lineTo(bx + 1.5, y - k); g.stroke(); }
      g.fillStyle = '#4e6e3a';
      g.beginPath(); g.ellipse(bx + 5, y - h + 4, 7, 2, -0.5, 0, 7); g.fill();
      g.beginPath(); g.ellipse(bx - 5, y - h + 8, 7, 2, 0.5, 0, 7); g.fill();
    }
  }

  private paintForests(g: CanvasRenderingContext2D, R: () => number): void {
    const clusters: [number, number, number, number][] = [
      [310, 360, 70, 12], [120, 360, 60, 8], [780, 330, 70, 9], [1180, 480, 90, 12], [1320, 640, 80, 10],
      [430, 830, 110, 16], [640, 860, 90, 12], [820, 760, 70, 9], [1290, 180, 70, 8], [230, 700, 60, 8],
    ];
    for (const [cx, cy, r, n] of clusters) {
      const trees = Array.from({ length: n }, () => [cx + (R() - 0.5) * 2 * r, cy + (R() - 0.5) * r] as const).sort((a, b) => a[1] - b[1]);
      for (const [x, y] of trees) this.tree(g, x, y, 0.8 + R() * 0.5);
    }
    // bamboo groves on both banks by the bridge
    for (const [x, y] of [[640, 470], [600, 600], [760, 610], [740, 470], [820, 590], [560, 520]]) this.bambooClump(g, x, y, R);
  }

  private paintGarden(g: CanvasRenderingContext2D): void {
    const cx = 1000, cy = 640;
    g.fillStyle = '#e8dcc0';
    g.beginPath(); g.ellipse(cx, cy, 120, 62, 0, 0, 7); g.fill();
    g.strokeStyle = INK; g.lineWidth = 2; g.stroke();
    g.strokeStyle = 'rgba(58,38,22,.35)'; g.lineWidth = 1;
    for (let i = 1; i < 6; i++) { g.beginPath(); g.ellipse(cx, cy, 120 - i * 10, 62 - i * 5, 0, 0, 7); g.stroke(); }
    for (const [x, y, r] of [[-60, -10, 14], [55, 18, 18], [30, -30, 9], [-20, 30, 10]]) {
      g.strokeStyle = 'rgba(58,38,22,.4)';
      for (let k = 1; k <= 3; k++) { g.beginPath(); g.ellipse(cx + x, cy + y, r + k * 6, (r + k * 6) * 0.55, 0, 0, 7); g.stroke(); }
      g.fillStyle = '#8f8878'; g.strokeStyle = INK; g.lineWidth = 1.6;
      g.beginPath(); g.ellipse(cx + x, cy + y - r * 0.2, r, r * 0.7, 0, 0, 7); g.fill(); g.stroke();
    }
  }

  private house(g: CanvasRenderingContext2D, x: number, y: number, s: number): void {
    g.fillStyle = '#d6c09a'; g.strokeStyle = INK; g.lineWidth = 1.4;
    g.fillRect(x - 12 * s, y - 10 * s, 24 * s, 10 * s); g.strokeRect(x - 12 * s, y - 10 * s, 24 * s, 10 * s);
    g.fillStyle = '#6a8a4a';
    g.beginPath(); g.moveTo(x - 17 * s, y - 9 * s); g.lineTo(x - 10 * s, y - 20 * s); g.lineTo(x + 10 * s, y - 20 * s); g.lineTo(x + 17 * s, y - 9 * s); g.closePath(); g.fill(); g.stroke();
  }

  private paintVillage(g: CanvasRenderingContext2D, R: () => number): void {
    const cx = 1420, cy = 890, r = 225;
    const hs = Array.from({ length: 16 }, () => { const a = R() * Math.PI * 2, d = R() * r * 0.8; return [cx + Math.cos(a) * d, cy + Math.sin(a) * d * 0.6] as const; }).sort((a, b) => a[1] - b[1]);
    for (const [x, y] of hs) this.house(g, x, y, 0.9 + R() * 0.3);
    // the palisade, open at the gate where the path comes in
    const gateA = Math.atan2((822 - cy) / 0.6, 1195 - cx);
    g.strokeStyle = '#6a4424'; g.lineWidth = 6; g.lineCap = 'butt';
    g.beginPath(); g.ellipse(cx, cy, r, r * 0.6, 0, gateA + 0.12, gateA + Math.PI * 2 - 0.12); g.stroke();
    g.strokeStyle = INK; g.lineWidth = 1.2;
    for (let a = gateA + 0.14; a < gateA + Math.PI * 2 - 0.12; a += 0.07) {
      const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r * 0.6;
      g.beginPath(); g.moveTo(x, y + 3); g.lineTo(x, y - 7); g.stroke();
    }
    g.font = '700 26px Cinzel, serif'; g.fillStyle = INK; g.textAlign = 'center';
    g.fillText('Hollow Pine Village', cx + 20, cy + 70);
    // earth clan camp flags by the gate
    for (const [x, y] of [[1150, 760], [1245, 770], [1180, 880]]) {
      g.strokeStyle = INK; g.lineWidth = 1.6;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x, y - 30); g.stroke();
      g.fillStyle = '#4e6e32'; g.beginPath(); g.moveTo(x, y - 30); g.lineTo(x + 18, y - 25); g.lineTo(x, y - 19); g.closePath(); g.fill(); g.stroke();
    }
  }

  private paintTemple(g: CanvasRenderingContext2D): void {
    const x = 150, y = 150;
    g.fillStyle = '#d6c09a'; g.strokeStyle = INK; g.lineWidth = 1.8;
    g.fillRect(x - 40, y - 4, 80, 10); g.strokeRect(x - 40, y - 4, 80, 10);
    const tier = (w: number, top: number, h: number) => {
      g.fillStyle = '#e4d2ac'; g.fillRect(x - w * 0.35, top, w * 0.7, h); g.strokeRect(x - w * 0.35, top, w * 0.7, h);
      g.fillStyle = RED;
      g.beginPath(); g.moveTo(x - w / 2 - 6, top + 2); g.quadraticCurveTo(x - w / 2, top - 2, x - w * 0.3, top - 10);
      g.lineTo(x + w * 0.3, top - 10); g.quadraticCurveTo(x + w / 2, top - 2, x + w / 2 + 6, top + 2); g.closePath(); g.fill(); g.stroke();
    };
    tier(76, y - 20, 16); tier(58, y - 42, 12); tier(40, y - 62, 10);
    g.beginPath(); g.moveTo(x, y - 72); g.lineTo(x, y - 86); g.stroke();
    g.fillStyle = GOLD; g.beginPath(); g.arc(x, y - 88, 3.5, 0, 7); g.fill(); g.stroke();
    g.font = '700 24px Cinzel, serif'; g.fillStyle = INK; g.textAlign = 'center';
    g.fillText('Ember Temple', x + 10, y + 40);
  }

  private paintBridge(g: CanvasRenderingContext2D): void {
    const a = pointAt(129), b = pointAt(141), ang = Math.atan2(b.y - a.y, b.x - a.x), len = Math.hypot(b.x - a.x, b.y - a.y) + 30;
    g.save();
    g.translate((a.x + b.x) / 2, (a.y + b.y) / 2); g.rotate(ang);
    g.fillStyle = '#b08850'; g.strokeStyle = INK; g.lineWidth = 1.6;
    g.fillRect(-len / 2, -9, len, 18); g.strokeRect(-len / 2, -9, len, 18);
    for (let x = -len / 2 + 5; x < len / 2; x += 6) { g.beginPath(); g.moveTo(x, -9); g.lineTo(x, 9); g.stroke(); }
    g.restore();
  }

  private paintStairs(g: CanvasRenderingContext2D): void {
    g.strokeStyle = INK; g.lineWidth = 1.6;
    for (let d = 72; d <= 108; d += 2.4) {
      const p = pointAt(d), q = pointAt(d + 0.5), ang = Math.atan2(q.y - p.y, q.x - p.x) + Math.PI / 2;
      g.beginPath(); g.moveTo(p.x - Math.cos(ang) * 9, p.y - Math.sin(ang) * 9); g.lineTo(p.x + Math.cos(ang) * 9, p.y + Math.sin(ang) * 9); g.stroke();
    }
  }

  /** Curling cloud scrolls in the old style. */
  private paintClouds(g: CanvasRenderingContext2D): void {
    const cloud = (x: number, y: number, s: number) => {
      g.save(); g.translate(x, y); g.scale(s, s);
      g.fillStyle = 'rgba(250,244,230,.85)'; g.strokeStyle = INK_SOFT; g.lineWidth = 2;
      g.beginPath();
      g.moveTo(-60, 10);
      g.bezierCurveTo(-70, -10, -40, -24, -26, -10);
      g.bezierCurveTo(-24, -34, 12, -36, 14, -12);
      g.bezierCurveTo(26, -28, 58, -20, 52, 0);
      g.bezierCurveTo(70, 2, 70, 16, 56, 16);
      g.lineTo(-60, 16);
      g.closePath(); g.fill(); g.stroke();
      g.beginPath(); g.arc(-26, -2, 8, Math.PI * 0.2, Math.PI * 1.7); g.stroke();
      g.beginPath(); g.arc(14, -4, 9, Math.PI * 0.1, Math.PI * 1.6); g.stroke();
      g.restore();
    };
    cloud(1000, 110, 1.1); cloud(420, 610, 0.8); cloud(1300, 420, 0.9); cloud(700, 930, 0.8);
  }

  private paintCompass(g: CanvasRenderingContext2D, x: number, y: number): void {
    g.save(); g.translate(x, y);
    g.strokeStyle = INK; g.lineWidth = 1.5;
    g.beginPath(); g.arc(0, 0, 44, 0, 7); g.stroke();
    g.beginPath(); g.arc(0, 0, 38, 0, 7); g.stroke();
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI) / 4, long = i % 2 === 0, r = long ? 56 : 34;
      g.fillStyle = i === 0 ? RED : long ? INK : '#8a6a44';
      g.beginPath();
      g.moveTo(Math.cos(a - Math.PI / 2) * r, Math.sin(a - Math.PI / 2) * r);
      g.lineTo(Math.cos(a - Math.PI / 2 + 0.35) * 8, Math.sin(a - Math.PI / 2 + 0.35) * 8);
      g.lineTo(Math.cos(a - Math.PI / 2 - 0.35) * 8, Math.sin(a - Math.PI / 2 - 0.35) * 8);
      g.closePath(); g.fill();
    }
    g.font = '700 16px Cinzel, serif'; g.fillStyle = INK; g.textAlign = 'center';
    g.fillText('N', 0, -62);
    g.restore();
  }

  private paintTitle(g: CanvasRenderingContext2D): void {
    const x = 1190, y = 70;
    g.fillStyle = 'rgba(245,234,206,.92)'; g.strokeStyle = INK; g.lineWidth = 2;
    g.beginPath(); g.moveTo(x - 190, y - 32); g.lineTo(x + 190, y - 32); g.lineTo(x + 210, y); g.lineTo(x + 190, y + 32); g.lineTo(x - 190, y + 32); g.lineTo(x - 210, y); g.closePath();
    g.fill(); g.stroke();
    g.font = '700 34px Cinzel, serif'; g.fillStyle = RED; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText('The Ember Path', x, y - 2);
    g.font = '600 13px Cinzel, serif'; g.fillStyle = INK;
    g.fillText('CHAPTER ONE', x, y + 22);
    g.textBaseline = 'alphabetic';
  }

  // ---------- live: path, stops, scrolls, you ----------

  private drawPath(c: CanvasRenderingContext2D, v: MapView): void {
    const line = (from: number, to: number) => {
      c.beginPath();
      for (let d = from; d <= to; d += 1.5) { const p = pointAt(d); d === from ? c.moveTo(p.x, p.y) : c.lineTo(p.x, p.y); }
      const p = pointAt(to); c.lineTo(p.x, p.y);
    };
    c.lineCap = 'round'; c.lineJoin = 'round';
    // the way ahead: an inked dotted trail
    c.setLineDash([2, 12]); c.strokeStyle = INK; c.lineWidth = 5;
    line(v.d, 300); c.stroke();
    c.setLineDash([]);
    // the way you've come, in red
    if (v.d > 0.5) {
      c.strokeStyle = 'rgba(179,50,31,.25)'; c.lineWidth = 16; line(0, v.d); c.stroke();
      c.strokeStyle = RED; c.lineWidth = 6; line(0, v.d); c.stroke();
    }
  }

  private drawScrolls(c: CanvasRenderingContext2D, v: MapView): void {
    STOPS.forEach(s => {
      if (!s.scroll || s.scrollAt === undefined || v.hasScroll(s.scroll)) return;
      const p = pointAt(s.scrollAt), bob = Math.sin(this.t * 2.5 + s.scrollAt) * 3, y = p.y - 26 + bob;
      const gr = c.createRadialGradient(p.x, y, 0, p.x, y, 34);
      gr.addColorStop(0, 'rgba(255,210,110,.7)'); gr.addColorStop(1, 'rgba(255,210,110,0)');
      c.fillStyle = gr; c.fillRect(p.x - 34, y - 34, 68, 68);
      c.strokeStyle = INK; c.lineWidth = 1.6;
      c.fillStyle = '#f6e7c2'; c.fillRect(p.x - 14, y - 7, 28, 14); c.strokeRect(p.x - 14, y - 7, 28, 14);
      c.fillStyle = '#c9a868';
      for (const sx of [-1, 1]) { c.beginPath(); c.ellipse(p.x + sx * 15, y, 4, 9, 0, 0, 7); c.fill(); c.stroke(); }
      c.fillStyle = RED; c.fillRect(p.x - 2, y - 7, 4, 14);
      c.beginPath(); c.moveTo(p.x, p.y - 12 + bob); c.lineTo(p.x, p.y - 2); c.strokeStyle = 'rgba(58,38,22,.4)'; c.stroke();
    });
  }

  private flame(c: CanvasRenderingContext2D, x: number, y: number, r: number, col: string, inner = '#ffe9a8'): void {
    c.fillStyle = col;
    c.beginPath();
    c.moveTo(x, y - r * 1.4);
    c.bezierCurveTo(x + r * 0.9, y - r * 0.4, x + r * 0.9, y + r * 0.8, x, y + r * 0.9);
    c.bezierCurveTo(x - r * 0.9, y + r * 0.8, x - r * 0.9, y - r * 0.4, x, y - r * 1.4);
    c.fill();
    c.fillStyle = inner;
    c.beginPath();
    c.moveTo(x, y - r * 0.5);
    c.bezierCurveTo(x + r * 0.45, y, x + r * 0.4, y + r * 0.7, x, y + r * 0.7);
    c.bezierCurveTo(x - r * 0.4, y + r * 0.7, x - r * 0.45, y, x, y - r * 0.5);
    c.fill();
  }

  private drawStops(c: CanvasRenderingContext2D, v: MapView): void {
    STOPS.forEach((s, i) => {
      const p = pointAt(s.pathAt), done = v.isDone(i), next = i === v.next, boss = i === STOPS.length - 1;
      const r = boss ? 26 : 21;
      if (done || next) {
        const pulse = next ? 0.6 + 0.4 * Math.sin(this.t * 3) : 0.7;
        const gr = c.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 2.6);
        gr.addColorStop(0, `rgba(255,${done ? 150 : 200},70,${0.55 * pulse})`); gr.addColorStop(1, 'rgba(255,150,70,0)');
        c.fillStyle = gr; c.fillRect(p.x - r * 3, p.y - r * 3, r * 6, r * 6);
      }
      c.fillStyle = done ? '#e88a3a' : next ? '#f7d58a' : '#b8a888';
      c.strokeStyle = INK; c.lineWidth = 3;
      c.beginPath(); c.arc(p.x, p.y, r, 0, 7); c.fill(); c.stroke();
      c.lineWidth = 1.2; c.beginPath(); c.arc(p.x, p.y, r - 5, 0, 7); c.stroke();
      if (boss && !done) {
        // Daro's mark: a stone fist
        c.fillStyle = next ? '#5a4a36' : '#7a6a56';
        c.fillRect(p.x - 9, p.y - 6, 18, 14);
        for (let k = 0; k < 4; k++) { c.beginPath(); c.arc(p.x - 6.75 + k * 4.5, p.y - 6, 2.4, Math.PI, 0); c.fill(); }
        c.fillRect(p.x - 12, p.y - 2, 5, 8);
      } else if (done) this.flame(c, p.x, p.y + 2, 9, RED);
      else if (next) this.flame(c, p.x, p.y + 2, 9, '#d8752c');
      else { c.fillStyle = '#6a5a48'; c.beginPath(); c.arc(p.x, p.y, 5, 0, 7); c.fill(); }
      // name and flames earned
      c.font = `700 ${boss ? 22 : 19}px Cinzel, serif`; c.textAlign = 'left'; c.textBaseline = 'middle';
      const lx = p.x + r + 10, ly = p.y - (done ? 8 : 0);
      c.lineWidth = 6; c.strokeStyle = 'rgba(236,220,182,.9)'; c.strokeText(s.place, lx, ly);
      c.fillStyle = boss ? RED : INK; c.fillText(s.place, lx, ly);
      if (done) {
        const n = v.flames(i);
        for (let k = 0; k < 3; k++) this.flame(c, lx + 8 + k * 18, ly + 22, 6, k < n ? RED : 'rgba(58,38,22,.25)', k < n ? '#ffd27a' : 'rgba(0,0,0,0)');
      }
      c.textBaseline = 'alphabetic';
    });
  }

  private drawYou(c: CanvasRenderingContext2D, v: MapView): void {
    const p = pointAt(v.d), step = v.walking ? Math.abs(Math.sin(this.t * 7)) * 5 : 0, y = p.y - 20 - step;
    // footprints behind you
    const last = this.trail[this.trail.length - 1];
    if (v.walking && (!last || Math.hypot(last.x - p.x, last.y - p.y) > 9)) {
      this.trail.push({ x: p.x, y: p.y });
      if (this.trail.length > 14) this.trail.shift();
    }
    this.trail.forEach((q, i) => {
      c.fillStyle = `rgba(58,38,22,${0.35 * ((i + 1) / this.trail.length)})`;
      c.beginPath(); c.ellipse(q.x + (i % 2 ? 3 : -3), q.y, 2.4, 1.6, 0, 0, 7); c.fill();
    });
    c.fillStyle = 'rgba(40,20,10,.35)';
    c.beginPath(); c.ellipse(p.x, p.y, 13 - step * 0.6, 4.5, 0, 0, 7); c.fill();
    if (v.waiting) {
      const k = (this.t * 0.8) % 1;
      c.strokeStyle = `rgba(242,182,64,${1 - k})`; c.lineWidth = 3;
      c.beginPath(); c.arc(p.x, y, 20 + k * 26, 0, 7); c.stroke();
    }
    const gr = c.createRadialGradient(p.x, y, 0, p.x, y, 42);
    gr.addColorStop(0, 'rgba(255,170,60,.55)'); gr.addColorStop(1, 'rgba(255,120,40,0)');
    c.fillStyle = gr; c.fillRect(p.x - 42, y - 42, 84, 84);
    c.fillStyle = '#2a140c'; c.strokeStyle = '#fff4e0'; c.lineWidth = 3;
    c.beginPath(); c.arc(p.x, y, 17, 0, 7); c.fill(); c.stroke();
    const flick = 1 + Math.sin(this.t * 13) * 0.06;
    this.flame(c, p.x, y + 2, 10 * flick, '#ff7a2a', '#ffe38a');
  }
}
