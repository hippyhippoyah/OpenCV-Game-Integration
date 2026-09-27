import { arrival, FLOOR_Y, FOCAL, type Enemy, type Game, type GameEvent } from '../game/game';
import { clamp, lerp, mulberry32, type Vec2 } from '../math';

type Pal = 'fire' | 'spirit';
interface Particle {
  x: number; y: number; z: number; vx: number; vy: number; vz: number;
  life: number; max: number; size: number; pal: Pal; rise: number;
}

/** How far each background layer shifts when your head moves (1 = as much as the floor at your feet). */
const PAR_SKY = 0.03, PAR_MID = 0.16;
const MAX_PARTICLES = 2600;
const rnd = (a: number, b: number) => a + Math.random() * (b - a);
const nOf = (rate: number, dt: number) => { const x = rate * dt; return Math.floor(x) + (Math.random() < x % 1 ? 1 : 0); };

function sprite(stops: [number, string][]): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d')!, gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  for (const [o, col] of stops) gr.addColorStop(o, col);
  g.fillStyle = gr;
  g.fillRect(0, 0, 64, 64);
  return c;
}

const SPR: Record<Pal, HTMLCanvasElement[]> = {
  fire: [
    sprite([[0, 'rgba(255,255,235,1)'], [0.3, 'rgba(255,220,130,.85)'], [1, 'rgba(255,130,30,0)']]),
    sprite([[0, 'rgba(255,190,90,.95)'], [0.4, 'rgba(255,110,30,.55)'], [1, 'rgba(200,40,10,0)']]),
    sprite([[0, 'rgba(220,60,20,.6)'], [0.5, 'rgba(120,20,10,.25)'], [1, 'rgba(60,0,0,0)']]),
  ],
  spirit: [
    sprite([[0, 'rgba(240,255,255,1)'], [0.3, 'rgba(150,235,255,.8)'], [1, 'rgba(60,160,255,0)']]),
    sprite([[0, 'rgba(120,220,255,.9)'], [0.4, 'rgba(60,140,255,.5)'], [1, 'rgba(40,40,200,0)']]),
    sprite([[0, 'rgba(120,80,255,.5)'], [0.5, 'rgba(70,40,180,.2)'], [1, 'rgba(30,0,80,0)']]),
  ],
};

/** Draws the first-person world. World units: eyes at the origin, x right, y down, z into the screen. */
export class Renderer {
  W = 0;
  H = 0;
  /** Pixels per world unit at the player's plane. */
  u = 8;
  VP: Vec2 = { x: 0, y: 0 };

  private ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private M = 0;
  private t = 0;
  private shake = 0;
  private flash = 0;
  private cam: Vec2 = { x: 0, y: 0 };
  private sky = document.createElement('canvas');
  private mid = document.createElement('canvas');
  private vig = document.createElement('canvas');
  private handLayer = document.createElement('canvas');
  private parts: Particle[] = [];
  private lanterns: Vec2[] = [];

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    this.resize();
  }

  /** Half the screen width in world units at the player's plane. */
  get viewHalfW(): number { return this.W / 2 / this.u; }

  screenToView(x: number, y: number): Vec2 { return { x: (x - this.VP.x) / this.u, y: (y - this.VP.y) / this.u }; }

  private viewToScreen(p: Vec2): Vec2 { return { x: this.VP.x + p.x * this.u, y: this.VP.y + p.y * this.u }; }

  private project(x: number, y: number, z: number): { x: number; y: number; s: number } {
    const s = FOCAL / (FOCAL + Math.max(z, -1.8));
    return { x: this.VP.x + (x - this.cam.x) * s * this.u, y: this.VP.y + (y - this.cam.y) * s * this.u, s };
  }

  resize(): void {
    this.dpr = Math.min(devicePixelRatio || 1, 2);
    this.W = innerWidth;
    this.H = innerHeight;
    this.u = Math.min(this.H, this.W * 1.25) / 100;
    this.M = 12 * this.u;
    this.VP = { x: this.W / 2, y: this.H * 0.45 };
    for (const c of [this.canvas, this.vig, this.handLayer]) {
      c.width = Math.round(this.W * this.dpr);
      c.height = Math.round(this.H * this.dpr);
    }
    for (const c of [this.sky, this.mid]) {
      c.width = Math.round((this.W + 2 * this.M) * this.dpr);
      c.height = Math.round((this.H + 2 * this.M) * this.dpr);
    }
    this.drawSky();
    this.drawMid();
    this.drawVignette();
  }

  onEvent(e: GameEvent): void {
    switch (e.type) {
      case 'punch': this.burst(e.x, e.y, e.z, 'fire', 26, 30); this.shake = Math.max(this.shake, 0.12); break;
      case 'hitEnemy':
      case 'killEnemy': this.burst(e.x, e.y, e.z, 'fire', 30, 40); break;
      case 'clash': this.burst(e.x, e.y, e.z, 'spirit', 24, 34); break;
      case 'blocked':
        this.burst(e.x, e.y, 0, 'spirit', 20, 30);
        this.burst(e.x, e.y, 0, 'fire', 14, 24);
        this.shake = Math.max(this.shake, 0.25);
        break;
      case 'playerHit': this.burst(e.x, e.y, 0, 'spirit', 26, 36); this.shake = 1; this.flash = 1; break;
    }
  }

  render(g: Game | null, dt: number): void {
    this.t += dt;
    this.cam = g ? g.cam : { x: 0, y: 0 };
    if (g) this.emitFromState(g, dt);
    this.updateParticles(dt);
    this.shake = Math.max(0, this.shake - dt * 2.5);
    this.flash = Math.max(0, this.flash - dt * 2);

    const c = this.ctx, { W, H, M, u } = this;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    if (this.shake > 0) c.translate(rnd(-1, 1) * this.shake * 1.2 * u, rnd(-1, 1) * this.shake * 1.2 * u);

    c.drawImage(this.sky, -M - this.cam.x * u * PAR_SKY, -M - this.cam.y * u * PAR_SKY, W + 2 * M, H + 2 * M);
    this.drawFloor();
    const mx = -M - this.cam.x * u * PAR_MID, my = -M - this.cam.y * u * PAR_MID;
    c.drawImage(this.mid, mx, my, W + 2 * M, H + 2 * M);
    this.drawLanterns(mx + M, my + M);

    if (g) [...g.enemies].sort((a, b) => b.z - a.z).forEach(e => this.drawEnemy(e));
    this.drawParticles(true);
    if (g) {
      this.drawProjectiles(g, true);
      this.drawLandingMarkers(g);
      this.drawHandLight(g);
      this.drawProjectiles(g, false);
      this.drawHands(g);
      this.drawOffscreenHands(g);
    }
    this.drawParticles(false);
    if (g && !g.shield.on) {
      // a bright core in each open palm
      c.globalCompositeOperation = 'lighter';
      for (const h of [g.hands.l, g.hands.r]) {
        if (!h?.open || !h.inView) continue;
        const p = this.viewToScreen(h.pos), r = 3 * u * (0.95 + 0.08 * Math.sin(this.t * 25));
        c.drawImage(SPR.fire[0], p.x - r, p.y - r, r * 2, r * 2);
      }
      c.globalCompositeOperation = 'source-over';
    }

    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.drawImage(this.vig, 0, 0, W, H);
    if (this.flash > 0) {
      const gr = c.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.2, W / 2, H / 2, Math.max(W, H) * 0.7);
      gr.addColorStop(0, 'rgba(255,0,0,0)');
      gr.addColorStop(1, `rgba(255,30,30,${0.5 * this.flash})`);
      c.fillStyle = gr;
      c.fillRect(0, 0, W, H);
    }
  }

  // ---------- static layers ----------

  private drawSky(): void {
    const { W, H, M, u } = this, g = this.sky.getContext('2d')!, hz = this.VP.y, R = mulberry32(11);
    g.setTransform(this.dpr, 0, 0, this.dpr, this.dpr * M, this.dpr * M);
    g.clearRect(-M, -M, W + 2 * M, H + 2 * M);
    let gr = g.createLinearGradient(0, -M, 0, hz);
    gr.addColorStop(0, '#06061a'); gr.addColorStop(0.55, '#151131'); gr.addColorStop(1, '#3c1d33');
    g.fillStyle = gr;
    g.fillRect(-M, -M, W + 2 * M, hz + M + 1);
    for (let i = 0; i < 180; i++) {
      const x = R() * (W + 2 * M) - M, y = R() * hz * 0.8 - M, r = R() * 1.3 + 0.3;
      g.globalAlpha = 0.25 + R() * 0.75;
      g.fillStyle = '#fff';
      g.fillRect(x, y, r, r);
    }
    g.globalAlpha = 1;
    const mx = W * 0.8, my = H * 0.13, mr = 4.2 * u;
    gr = g.createRadialGradient(mx, my, 0, mx, my, mr * 6);
    gr.addColorStop(0, 'rgba(255,238,215,.28)'); gr.addColorStop(1, 'rgba(255,238,215,0)');
    g.fillStyle = gr;
    g.fillRect(mx - mr * 6, my - mr * 6, mr * 12, mr * 12);
    g.fillStyle = '#f3ead8';
    g.beginPath(); g.arc(mx, my, mr, 0, 7); g.fill();
    g.fillStyle = 'rgba(120,100,90,.12)';
    for (const [a, b, r] of [[-0.3, -0.2, 0.28], [0.25, 0.2, 0.22], [0.1, -0.4, 0.12]]) {
      g.beginPath(); g.arc(mx + a * mr, my + b * mr, r * mr, 0, 7); g.fill();
    }
    const ridge = (base: number, amp: number, col: string, seed: number, f1: number, f2: number) => {
      g.fillStyle = col;
      g.beginPath();
      g.moveTo(-M, hz + 2 * u);
      for (let x = -M; x <= W + M + 8; x += 6) {
        g.lineTo(x, base - amp * (0.55 + 0.3 * Math.sin(x * f1 + seed) + 0.15 * Math.sin(x * f2 + seed * 3)));
      }
      g.lineTo(W + M, hz + 2 * u);
      g.closePath();
      g.fill();
    };
    ridge(hz - 2 * u, 14 * u, '#231838', 1.3, 0.006, 0.021);
    ridge(hz, 9 * u, '#170f27', 4.1, 0.009, 0.03);
    gr = g.createLinearGradient(0, hz - 8 * u, 0, hz + 3 * u);
    gr.addColorStop(0, 'rgba(255,110,80,0)'); gr.addColorStop(0.7, 'rgba(255,110,80,.12)'); gr.addColorStop(1, 'rgba(255,110,80,0)');
    g.fillStyle = gr;
    g.fillRect(-M, hz - 8 * u, W + 2 * M, 11 * u);
  }

  private drawMid(): void {
    const { W, M, u } = this, g = this.mid.getContext('2d')!, hz = this.VP.y, k = u * 0.8;
    g.setTransform(this.dpr, 0, 0, this.dpr, this.dpr * M, this.dpr * M);
    g.clearRect(-M, -M, W + 2 * M, this.H + 2 * M);
    g.fillStyle = '#110c19';
    g.fillRect(-M, hz - 2.2 * u, W + 2 * M, 2.6 * u);
    for (let x = -M; x < W + M; x += 9 * u) g.fillRect(x, hz - 3.4 * u, 1.6 * u, 1.4 * u);
    this.lanterns = [];
    const roof = (l: number, r: number, y: number, h: number) => {
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
    };
    const temple = (x: number, w: number) => {
      const base = hz - u, bodyH = 12 * k, b1 = base - 3 * k - bodyH, b2 = b1 - 12 * k;
      g.fillStyle = '#0e0a16';
      g.fillRect(x - w * 0.04, base - 3 * k, w * 1.08, 3 * k);
      g.fillRect(x + w * 0.1, b1, w * 0.8, bodyH);
      roof(x - w * 0.02, x + w * 1.02, b1, 5 * k);
      g.fillRect(x + w * 0.3, b2, w * 0.4, 7 * k);
      roof(x + w * 0.18, x + w * 0.82, b2, 4.5 * k);
      g.fillStyle = 'rgba(255,165,90,.22)';
      for (let i = 0; i < 3; i++) g.fillRect(x + w * (0.22 + i * 0.22), b1 + bodyH * 0.35, w * 0.1, bodyH * 0.4);
      g.fillRect(x + w * 0.44, b2 + 2 * k, w * 0.12, 3.5 * k);
      this.lanterns.push({ x: x + w * 0.12, y: b1 + 2.4 * k }, { x: x + w * 0.88, y: b1 + 2.4 * k });
    };
    temple(W * 0.02, W * 0.22);
    temple(W * 0.76, W * 0.22);
  }

  private drawVignette(): void {
    const g = this.vig.getContext('2d')!, { W, H } = this;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const gr = g.createRadialGradient(W / 2, H * 0.5, Math.min(W, H) * 0.35, W / 2, H * 0.5, Math.max(W, H) * 0.75);
    gr.addColorStop(0, 'rgba(0,0,0,0)'); gr.addColorStop(1, 'rgba(0,0,0,.7)');
    g.fillStyle = gr;
    g.fillRect(0, 0, W, H);
  }

  // ---------- live world ----------

  /** The floor is drawn every frame so its perspective follows your head. */
  private drawFloor(): void {
    const c = this.ctx, hz = this.VP.y;
    const gr = c.createLinearGradient(0, hz, 0, this.H);
    gr.addColorStop(0, '#24182b'); gr.addColorStop(1, '#0a080f');
    c.fillStyle = gr;
    c.fillRect(0, hz, this.W, this.H - hz);
    c.strokeStyle = 'rgba(255,200,160,.06)';
    c.lineWidth = 1;
    const step = (this.W * 0.09) / this.u;
    for (let i = -18; i <= 18; i++) {
      const a = this.project(i * step, FLOOR_Y, 16), b = this.project(i * step, FLOOR_Y, -0.4);
      c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
    }
    for (const z of [0, 0.5, 1.1, 1.9, 2.9, 4.2, 5.9, 8.1, 11, 15]) {
      const y = this.project(0, FLOOR_Y, z).y;
      c.beginPath(); c.moveTo(0, y); c.lineTo(this.W, y); c.stroke();
    }
  }

  private drawLanterns(ox: number, oy: number): void {
    const c = this.ctx, u = this.u;
    c.globalCompositeOperation = 'lighter';
    this.lanterns.forEach((l, i) => {
      const x = l.x + ox, y = l.y + oy, r = 7 * u * (0.75 + 0.25 * Math.sin(this.t * 9 + i * 2) * Math.sin(this.t * 5.3 + i));
      const gr = c.createRadialGradient(x, y, 0, x, y, r);
      gr.addColorStop(0, 'rgba(255,170,80,.55)'); gr.addColorStop(1, 'rgba(255,120,40,0)');
      c.fillStyle = gr;
      c.fillRect(x - r, y - r, r * 2, r * 2);
      c.fillStyle = '#ffb45e';
      c.fillRect(x - 0.6 * u, y - 0.9 * u, 1.2 * u, 1.8 * u);
    });
    c.globalCompositeOperation = 'source-over';
  }

  private drawEnemy(e: Enemy): void {
    if (e.dummy) {
      this.drawDummy(e);
      return;
    }
    const c = this.ctx, u = this.u, p = this.project(e.x, e.y, e.z), s = p.s, sx = p.x;
    const bob = Math.sin(e.t * 2 + e.phase) * 1.5 * u * s;
    const cy = p.y + bob, hgt = 52 * u * s, feet = this.project(e.x, FLOOR_Y, e.z).y;
    const alpha = e.appear * (1 - Math.min(1, e.dying));
    if (alpha <= 0) return;
    c.globalAlpha = alpha * 0.5;
    c.fillStyle = '#000';
    c.beginPath(); c.ellipse(sx, feet, 9 * u * s, 2 * u * s, 0, 0, 7); c.fill();
    c.globalCompositeOperation = 'lighter';
    c.globalAlpha = alpha;
    let gr = c.createRadialGradient(sx, cy, 0, sx, cy, hgt * 0.75);
    gr.addColorStop(0, `rgba(80,190,255,${0.22 + e.flash * 0.4})`); gr.addColorStop(1, 'rgba(80,190,255,0)');
    c.fillStyle = gr;
    c.fillRect(sx - hgt, cy - hgt, hgt * 2, hgt * 2);
    c.globalCompositeOperation = 'source-over';
    // robe
    const top = cy - hgt * 0.42, hem = cy + hgt * 0.45, w = 11 * u * s;
    gr = c.createLinearGradient(0, top, 0, hem);
    gr.addColorStop(0, e.flash > 0 ? 'rgba(255,230,200,.95)' : 'rgba(175,235,255,.88)');
    gr.addColorStop(0.6, 'rgba(70,130,210,.55)'); gr.addColorStop(1, 'rgba(40,70,160,0)');
    c.fillStyle = gr;
    c.beginPath();
    c.moveTo(sx - w * 0.45, top + hgt * 0.12);
    c.quadraticCurveTo(sx, top - hgt * 0.08, sx + w * 0.45, top + hgt * 0.12);
    c.lineTo(sx + w * 0.8, top + hgt * 0.28);
    c.lineTo(sx + w * 0.55, top + hgt * 0.4);
    for (let i = 0; i <= 8; i++) {
      const k = i / 8;
      c.lineTo(sx + w * (0.6 - 1.2 * k), hem + Math.sin(k * 12 + e.t * 5) * 1.6 * u * s);
    }
    c.lineTo(sx - w * 0.55, top + hgt * 0.4);
    c.lineTo(sx - w * 0.8, top + hgt * 0.28);
    c.closePath();
    c.fill();
    // mask
    const hy = top + hgt * 0.03, hr = 4.6 * u * s;
    c.fillStyle = '#eaf4f6';
    c.beginPath(); c.ellipse(sx, hy, hr * 0.85, hr, 0, 0, 7); c.fill();
    c.fillStyle = '#0a1a28';
    c.beginPath(); c.ellipse(sx - hr * 0.35, hy - hr * 0.1, hr * 0.22, hr * 0.1, 0.35, 0, 7); c.fill();
    c.beginPath(); c.ellipse(sx + hr * 0.35, hy - hr * 0.1, hr * 0.22, hr * 0.1, -0.35, 0, 7); c.fill();
    c.fillStyle = '#c0392b';
    c.fillRect(sx - hr * 0.08, hy + hr * 0.3, hr * 0.16, hr * 0.35);
    // wind-up telegraph
    if (e.winding) {
      const o = this.project(e.x + e.side * 11, e.y - 14, e.z), r = (1.2 + e.wind * 3.5) * u * s;
      const ox = o.x, oy = o.y + bob;
      c.globalCompositeOperation = 'lighter';
      c.drawImage(SPR.spirit[1], ox - r * 2.2, oy - r * 2.2, r * 4.4, r * 4.4);
      c.drawImage(SPR.spirit[0], ox - r, oy - r, r * 2, r * 2);
      c.strokeStyle = `rgba(140,230,255,${0.6 * (1 - e.wind)})`;
      c.lineWidth = 2;
      c.beginPath(); c.arc(ox, oy, r * (4 - e.wind * 2.5), 0, 7); c.stroke();
      c.globalCompositeOperation = 'source-over';
    }
    c.globalAlpha = 1;
  }

  /** Wooden training post with a straw body and a target for a head. */
  private drawDummy(e: Enemy): void {
    const c = this.ctx, u = this.u, p = this.project(e.x, e.y, e.z), k = u * p.s, x = p.x;
    const feet = this.project(e.x, FLOOR_Y, e.z).y, top = p.y - 24 * k;
    const alpha = e.appear * (1 - Math.min(1, e.dying));
    if (alpha <= 0) return;
    const hit = e.flash > 0;
    c.globalAlpha = alpha;
    c.fillStyle = 'rgba(0,0,0,.5)';
    c.beginPath(); c.ellipse(x, feet, 7 * k, 1.6 * k, 0, 0, 7); c.fill();
    c.fillStyle = hit ? '#ffcf9a' : '#6b4a33';
    c.fillRect(x - 1.6 * k, top, 3.2 * k, feet - top);
    c.fillRect(x - 11 * k, p.y - 12 * k, 22 * k, 2.6 * k);
    c.fillStyle = hit ? '#fff0c8' : '#b8955a';
    c.beginPath(); c.ellipse(x, p.y - 2 * k, 6 * k, 11 * k, 0, 0, 7); c.fill();
    c.strokeStyle = '#5a3d25';
    c.lineWidth = k;
    for (const dy of [-8, 4]) { c.beginPath(); c.moveTo(x - 6 * k, p.y + dy * k); c.lineTo(x + 6 * k, p.y + dy * k); c.stroke(); }
    c.fillStyle = hit ? '#fff0c8' : '#d8c39a';
    c.beginPath(); c.arc(x, top, 5 * k, 0, 7); c.fill();
    c.strokeStyle = '#c0392b';
    c.beginPath(); c.arc(x, top, 3.4 * k, 0, 7); c.stroke();
    c.fillStyle = '#c0392b';
    c.beginPath(); c.arc(x, top, 1.2 * k, 0, 7); c.fill();
    c.globalAlpha = 1;
  }

  private drawProjectiles(g: Game, far: boolean): void {
    const c = this.ctx;
    c.globalCompositeOperation = 'lighter';
    for (const p of g.projs) {
      if ((p.z > 1) !== far) continue;
      const q = this.project(p.x, p.y, p.z), r = p.r * q.s * this.u, set = p.kind === 'player' ? SPR.fire : SPR.spirit;
      c.drawImage(set[1], q.x - r * 2.4, q.y - r * 2.4, r * 4.8, r * 4.8);
      c.drawImage(set[0], q.x - r * 1.2, q.y - r * 1.2, r * 2.4, r * 2.4);
    }
    c.globalCompositeOperation = 'source-over';
  }

  /** Where each incoming attack will land if you stay still; red = it would hit you. */
  private drawLandingMarkers(g: Game): void {
    const c = this.ctx;
    for (const p of g.projs) {
      if (p.kind !== 'enemy' || p.z > 5 || p.z < 0.2) continue;
      const a = arrival(p), q = this.project(a.x, a.y, 0), near = 1 - p.z / 5, danger = g.isThreat(p);
      c.strokeStyle = danger ? `rgba(255,80,70,${0.25 + near * 0.65})` : `rgba(130,220,255,${near * 0.6})`;
      c.lineWidth = danger ? 3 : 2;
      const r = p.r * this.u * (1.3 + (1 - near) * 2.5);
      c.beginPath(); c.arc(q.x, q.y, r, 0, 7); c.stroke();
      if (danger) { c.beginPath(); c.arc(q.x, q.y, r * 0.35, 0, 7); c.stroke(); }
    }
  }

  /** Warm light around each hand (small for embers in a fist, big for open flame) and the shield's flame sheet. */
  private drawHandLight(g: Game): void {
    const c = this.ctx, u = this.u, flicker = 0.9 + 0.1 * Math.sin(this.t * 20);
    c.globalCompositeOperation = 'lighter';
    for (const h of [g.hands.l, g.hands.r]) {
      if (!h?.inView) continue;
      const C = this.viewToScreen(h.pos), r = (h.open ? 32 : 12) * u;
      const gr = c.createRadialGradient(C.x, C.y, 0, C.x, C.y, r);
      gr.addColorStop(0, `rgba(255,140,60,${(h.open ? 0.26 : 0.14) * flicker})`); gr.addColorStop(1, 'rgba(255,120,40,0)');
      c.fillStyle = gr;
      c.fillRect(C.x - r, C.y - r, r * 2, r * 2);
    }
    const { l, r } = g.hands;
    if (g.shield.on && l && r) {
      const a = this.viewToScreen(l.pos), b = this.viewToScreen(r.pos), e = g.shield.energy, hgt = (14 + e * 10) * u;
      const sg = c.createLinearGradient(0, a.y, 0, a.y - hgt);
      sg.addColorStop(0, `rgba(255,150,60,${0.35 * e + 0.1})`); sg.addColorStop(1, 'rgba(255,90,30,0)');
      c.fillStyle = sg;
      c.beginPath();
      c.moveTo(a.x, a.y + 2 * u);
      c.lineTo(b.x, b.y + 2 * u);
      for (let i = 0; i <= 10; i++) {
        const k = 1 - i / 10;
        c.lineTo(lerp(a.x, b.x, k), lerp(a.y, b.y, k) - hgt * (0.75 + 0.25 * Math.sin(k * 14 + this.t * 12)));
      }
      c.closePath();
      c.fill();
    }
    c.globalCompositeOperation = 'source-over';
  }

  /**
   * First-person arm plus an open hand or a fist, in screen space. side: −1 left, +1 right.
   * With a tracked elbow the forearm bends where yours does; the upper arm always runs off the
   * bottom of the screen, since your shoulders are behind the camera.
   */
  private handShape(c: CanvasRenderingContext2D, h: Vec2, side: number, grow: number, open: boolean, elbowAt: Vec2 | null = null): void {
    const k = 1.5 * this.u, g = grow;
    c.lineCap = 'round';
    c.lineJoin = 'round';
    const line = (x1: number, y1: number, x2: number, y2: number, w: number) => {
      c.lineWidth = w + 2 * g;
      c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
    };
    const shoulder = { x: (elbowAt?.x ?? h.x) + side * (elbowAt ? 10 : 16) * k, y: this.H + 14 * k };
    const elbow = elbowAt ?? { x: lerp(shoulder.x, h.x, 0.55), y: lerp(shoulder.y, h.y + 4 * k, 0.55) };
    line(shoulder.x, shoulder.y, elbow.x, elbow.y, 9 * k);
    line(elbow.x, elbow.y, h.x, h.y + 3.5 * k, 6.4 * k);
    if (!open) {
      // fist: curled fingers with knuckle bumps, thumb wrapped across the front
      c.beginPath(); c.ellipse(h.x, h.y, 3.5 * k + g, 3.2 * k + g, 0, 0, 7); c.fill();
      for (let i = 0; i < 4; i++) {
        c.beginPath(); c.arc(h.x + (i - 1.5) * 1.55 * k, h.y - 2.5 * k, 1 * k + g, 0, 7); c.fill();
      }
      line(h.x - side * 2.8 * k, h.y + 0.8 * k, h.x + side * 0.6 * k, h.y - 0.2 * k, 1.8 * k);
      return;
    }
    const pw = 5.4 * k;
    c.beginPath(); c.ellipse(h.x, h.y, pw / 2 + g, 3.6 * k + g, 0, 0, 7); c.fill();
    for (let i = 0; i < 4; i++) {
      const a = -Math.PI / 2 + (i - 1.5) * 0.2, len = (i === 1 || i === 2 ? 5.2 : 4.4) * k;
      const bx = h.x + (i - 1.5) * pw * 0.26, by = h.y - 2.6 * k;
      line(bx, by, bx + Math.cos(a) * len, by + Math.sin(a) * len, 1.6 * k);
    }
    const ta = -Math.PI / 2 - side * 1.05, tx = h.x - side * pw * 0.42, ty = h.y + 0.4 * k;
    line(tx, ty, tx + Math.cos(ta) * 3.6 * k, ty + Math.sin(ta) * 3.6 * k, 1.9 * k);
  }

  private drawHands(g: Game): void {
    const hands = ([['l', -1], ['r', 1]] as const)
      .map(([side, sign]) => ({ h: g.hands[side], sign }))
      .filter((x): x is { h: NonNullable<typeof x.h>; sign: -1 | 1 } => x.h !== null && x.h.inView);
    if (!hands.length) return;
    const c = this.ctx, u = this.u, flicker = 0.9 + 0.1 * Math.sin(this.t * 20);
    // rim glow first, then the dark hands, lit on top by their own fire
    for (const { h, sign } of hands) {
      c.strokeStyle = c.fillStyle = `rgba(255,130,60,${(h.open ? 0.4 : 0.18) * flicker})`;
      this.handShape(c, this.viewToScreen(h.pos), sign, 0.9 * u, h.open, h.elbow && this.viewToScreen(h.elbow));
    }
    const hl = this.handLayer.getContext('2d')!;
    hl.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    hl.globalCompositeOperation = 'source-over';
    hl.clearRect(0, 0, this.W, this.H);
    hl.strokeStyle = hl.fillStyle = g.inv > 0 && Math.sin(this.t * 40) > 0 ? '#3a1216' : '#150f19';
    for (const { h, sign } of hands) this.handShape(hl, this.viewToScreen(h.pos), sign, 0, h.open, h.elbow && this.viewToScreen(h.elbow));
    hl.globalCompositeOperation = 'source-atop';
    for (const { h } of hands) {
      const C = this.viewToScreen(h.pos), gr = hl.createRadialGradient(C.x, C.y - 3 * u, 0, C.x, C.y, 30 * u);
      gr.addColorStop(0, `rgba(255,160,80,${(h.open ? 0.8 : 0.4) * flicker})`); gr.addColorStop(1, 'rgba(255,90,30,0)');
      hl.fillStyle = gr;
      hl.fillRect(0, 0, this.W, this.H);
    }
    c.drawImage(this.handLayer, 0, 0, this.W, this.H);
  }

  /** A hand outside the camera picture: a marker at the nearest screen edge, pointing toward it. */
  private drawOffscreenHands(g: Game): void {
    const c = this.ctx, u = this.u, margin = 6 * u;
    for (const side of ['l', 'r'] as const) {
      const h = g.hands[side];
      if (!h || h.inView) continue;
      const p = this.viewToScreen(h.pos);
      const q = { x: clamp(p.x, margin, this.W - margin), y: clamp(p.y, margin, this.H - margin) };
      const dir = Math.atan2(p.y - this.H / 2, p.x - this.W / 2), pulse = 0.6 + 0.4 * Math.sin(this.t * 6);
      c.globalCompositeOperation = 'lighter';
      const gr = c.createRadialGradient(q.x, q.y, 0, q.x, q.y, 6 * u);
      gr.addColorStop(0, `rgba(255,140,60,${0.35 * pulse})`); gr.addColorStop(1, 'rgba(255,120,40,0)');
      c.fillStyle = gr;
      c.fillRect(q.x - 6 * u, q.y - 6 * u, 12 * u, 12 * u);
      c.globalCompositeOperation = 'source-over';
      c.save();
      c.translate(q.x, q.y);
      c.rotate(dir);
      c.fillStyle = `rgba(255,190,120,${0.5 + 0.4 * pulse})`;
      c.beginPath(); c.moveTo(3.2 * u, 0); c.lineTo(1 * u, -1.6 * u); c.lineTo(1 * u, 1.6 * u); c.closePath(); c.fill();
      c.restore();
      c.strokeStyle = 'rgba(255,190,120,.8)';
      c.lineWidth = 2;
      c.beginPath(); c.arc(q.x, q.y, 1.8 * u, 0, 7); c.stroke();
      c.fillStyle = '#ffe2b8';
      c.font = `600 ${Math.round(1.8 * u)}px Inter, sans-serif`;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText(side.toUpperCase(), q.x, q.y);
    }
  }

  // ---------- particles ----------

  private emit(x: number, y: number, z: number, vx: number, vy: number, vz: number, life: number, size: number, pal: Pal = 'fire', rise = 1): void {
    if (this.parts.length < MAX_PARTICLES) this.parts.push({ x, y, z, vx, vy, vz, life, max: life, size, pal, rise });
  }

  private burst(x: number, y: number, z: number, pal: Pal, n: number, spd: number): void {
    for (let i = 0; i < n; i++) {
      const a = Math.random() * 6.283, sp = rnd(0.3, 1) * spd;
      this.emit(x, y, z, Math.cos(a) * sp, Math.sin(a) * sp, rnd(-1, 1), rnd(0.25, 0.65), rnd(2, 4.5), pal, 0.5);
    }
  }

  private emitFromState(g: Game, dt: number): void {
    // embers smoulder in closed fists; open hands pour out flame
    for (const h of [g.hands.l, g.hands.r]) {
      if (!h?.inView || (h.open && g.shield.on)) continue;
      const w = g.handWorld(h.pos), vx0 = h.vel.x * 0.3, vy0 = h.vel.y * 0.3;
      if (h.open) {
        for (let i = nOf(110, dt); i > 0; i--) {
          const a = Math.random() * 6.283, d = Math.sqrt(Math.random()) * 1.8;
          this.emit(w.x + Math.cos(a) * d, w.y - 2 + Math.sin(a) * d, 0, rnd(-5, 5) + vx0, rnd(-18, -6) + vy0, 0, rnd(0.3, 0.55), rnd(2.2, 3.6));
        }
      } else {
        for (let i = nOf(24, dt); i > 0; i--) {
          this.emit(w.x + rnd(-2.5, 2.5), w.y - 3 + rnd(-1, 1), 0, rnd(-3, 3) + vx0, rnd(-14, -6) + vy0, 0, rnd(0.3, 0.7), rnd(0.6, 1.4));
        }
      }
    }
    const { l, r } = g.hands;
    if (g.shield.on && l && r) {
      const a = g.handWorld(l.pos), b = g.handWorld(r.pos), e = g.shield.energy;
      for (let i = nOf(260, dt); i > 0; i--) {
        const k = Math.random();
        this.emit(lerp(a.x, b.x, k) + rnd(-1, 1), lerp(a.y, b.y, k) + rnd(-2, 3), 0,
          rnd(-3, 3), rnd(-34, -14) * (0.6 + e * 0.6), 0, rnd(0.25, 0.5), (3 + e * 2) * rnd(0.7, 1.1));
      }
    }
    for (const p of g.projs) {
      const pal: Pal = p.kind === 'player' ? 'fire' : 'spirit';
      for (let j = nOf(p.kind === 'player' ? 120 : 90, dt); j > 0; j--) {
        this.emit(p.x + rnd(-0.4, 0.4) * p.r, p.y + rnd(-0.4, 0.4) * p.r, p.z, rnd(-3, 3), rnd(-6, 2), p.vz * 0.25, rnd(0.2, 0.45), p.r * rnd(0.6, 1), pal, 0.6);
      }
    }
    for (const e of g.enemies) {
      if (e.hp > 0) continue;
      const s = FOCAL / (FOCAL + e.z);
      for (let j = nOf(120, dt); j > 0; j--) {
        this.emit(e.x + rnd(-8, 8), e.y + rnd(-20, 20), e.z, rnd(-10, 10) / s, rnd(-30, -5) / s, 0, rnd(0.4, 0.8), rnd(3, 6) / s, e.dummy ? 'fire' : 'spirit', 0.6);
      }
    }
  }

  private updateParticles(dt: number): void {
    const drag = Math.exp(-2.2 * dt), parts = this.parts;
    for (let i = parts.length - 1; i >= 0; i--) {
      const p = parts[i];
      p.life -= dt;
      if (p.life <= 0) { parts[i] = parts[parts.length - 1]; parts.pop(); continue; }
      p.vy -= p.rise * 38 * dt;
      p.vx *= drag;
      p.vy *= drag;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z += p.vz * dt;
    }
  }

  private drawParticles(far: boolean): void {
    const c = this.ctx;
    c.globalCompositeOperation = 'lighter';
    for (const p of this.parts) {
      if ((p.z > 1) !== far) continue;
      const q = this.project(p.x, p.y, p.z), k = 1 - p.life / p.max, idx = k < 0.28 ? 0 : k < 0.62 ? 1 : 2;
      const sz = p.size * (1 - k * 0.55) * q.s * this.u;
      c.globalAlpha = k < 0.12 ? k / 0.12 : 1 - (k - 0.12) / 0.88;
      c.drawImage(SPR[p.pal][idx], q.x - sz, q.y - sz, sz * 2, sz * 2);
    }
    c.globalAlpha = 1;
    c.globalCompositeOperation = 'source-over';
  }
}
