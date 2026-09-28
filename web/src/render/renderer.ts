import { arrival, FLOOR_Y, FOCAL, TUNE, type Blade, type Enemy, type Game, type GameEvent, type Hazard, type Wall } from '../game/game';
import type { Side } from '../input/types';
import { TUNING } from '../intent/interpret';
import { clamp, lerp, mulberry32, type Vec2 } from '../math';
import { ghostPose } from './ghost';
import { FLOORS, paintMid, paintSky, type Frame, type Glows, type Scene } from './scenes';

type Pal = 'fire' | 'spirit' | 'earth' | 'blue';
interface Particle {
  x: number; y: number; z: number; vx: number; vy: number; vz: number;
  life: number; max: number; size: number; pal: Pal; rise: number;
}

/** How far each background layer shifts when your head moves (1 = as much as the floor at your feet). */
const PAR_SKY = 0.03, PAR_MID = 0.16;
/** Leaning tilts the view this much (radians per world unit of lean), so leaning feels big. */
const LEAN_TILT = 0.0014;
const MAX_PARTICLES = 3200;
/** How long a hand keeps burning after it attacks. */
const FLARE_S = 0.35;
/** A fist resting in guard (above REST_LOW_Y) is drawn this much lower, near you (view units). */
const REST_DROP = 16, REST_LOW_Y = 50;
/** Blue inferno flames nearer than this are drawn in front of the enemies. */
const INFERNO_NEAR_Z = 5;
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
  // charged fire burns blue
  blue: [
    sprite([[0, 'rgba(240,250,255,1)'], [0.3, 'rgba(140,190,255,.9)'], [1, 'rgba(60,110,255,0)']]),
    sprite([[0, 'rgba(120,170,255,.95)'], [0.4, 'rgba(70,110,255,.6)'], [1, 'rgba(40,40,220,0)']]),
    sprite([[0, 'rgba(90,90,240,.6)'], [0.5, 'rgba(50,40,180,.25)'], [1, 'rgba(20,0,90,0)']]),
  ],
  earth: [
    sprite([[0, 'rgba(230,200,150,1)'], [0.35, 'rgba(170,120,70,.8)'], [1, 'rgba(90,60,30,0)']]),
    sprite([[0, 'rgba(160,110,60,.9)'], [0.45, 'rgba(110,75,40,.5)'], [1, 'rgba(60,40,20,0)']]),
    sprite([[0, 'rgba(90,70,50,.6)'], [0.5, 'rgba(60,45,30,.25)'], [1, 'rgba(30,20,10,0)']]),
  ],
};

/** Draws the first-person world. World units: eyes at the origin, x right, y down, z into the screen. */
export class Renderer {
  W = 0;
  H = 0;
  /** Pixels per world unit at the player's plane. */
  u = 8;
  VP: Vec2 = { x: 0, y: 0 };
  /** Ghost hands to show (campaign practice), and how visible. */
  ghost: { lessonId: string; alpha: number } | null = null;
  /** Which backdrop the fight is in (see scenes.ts). */
  get scene(): Scene { return this._scene; }
  set scene(v: Scene) {
    if (v === this._scene) return;
    this._scene = v;
    this.drawSky();
    this.drawMid();
  }
  private _scene: Scene = 'night';

  private ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private M = 0;
  private t = 0;
  private shake = 0;
  private flash = 0;
  /** Seconds of fire left in each hand after it attacks (hands only burn while doing something). */
  private flare: Record<Side, number> = { l: 0, r: 0 };

  private cam: Vec2 = { x: 0, y: 0 };
  private sky = document.createElement('canvas');
  private mid = document.createElement('canvas');
  private vig = document.createElement('canvas');
  private handLayer = document.createElement('canvas');
  private parts: Particle[] = [];
  private glows: Glows = { pts: [], color: [255, 160, 80] };
  /** The blue inferno's flames: where each stands (x across from you, z into the field) and how it flickers. */
  /** Screen-space trail of each charged shot (its dragon's body). */
  private dragonTrails = new Map<number, { x: number; y: number; r: number }[]>();
  private infernoFlames: { x: number; z: number; w: number; h: number; phase: number; speed: number }[] = [];

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
      case 'punch':
        this.burst(e.x, e.y, e.z, 'fire', 26, 30);
        this.shake = Math.max(this.shake, 0.12);
        this.flare[e.side] = FLARE_S;
        break;
      case 'wall':
        for (let i = 0; i < 80; i++) this.emit(e.x + rnd(-30, 30), e.y - rnd(0, 4), e.z, rnd(-8, 8), rnd(-70, -30), 0, rnd(0.4, 0.8), rnd(4, 8));
        this.shake = Math.max(this.shake, 0.3);
        this.flare = { l: FLARE_S * 1.5, r: FLARE_S * 1.5 };
        break;
      case 'inferno':
        // the ground bursts into blue flame from where the fists came down: lay out a field of flames
        this.infernoFlames = [];
        for (let i = 0; i < 150; i++) {
          const z = 0.4 + Math.random() ** 1.25 * 14;
          this.infernoFlames.push({ x: rnd(-1, 1) * (70 + z * 26), z, w: rnd(3, 5.5), h: rnd(12, 22), phase: rnd(0, 20), speed: rnd(2.5, 4.5) });
        }
        this.infernoFlames.sort((a, b) => b.z - a.z);
        this.burst(e.x, FLOOR_Y - 2, 0.4, 'blue', 90, 60);
        this.shake = Math.max(this.shake, 0.7);
        this.flare = { l: FLARE_S * 2, r: FLARE_S * 2 };
        break;
      case 'ultimate':
        this.burst(e.x, e.y, 0.3, 'fire', 60, 50);
        this.shake = 0.8;
        this.flare = { l: FLARE_S * 3, r: FLARE_S * 3 };
        break;
      case 'pillar':
        this.shake = Math.max(this.shake, 0.35);
        this.flare[e.side] = FLARE_S * 2;
        break;
      case 'combo':
        // a flourish at the hand or where it happened
        this.burst(e.x, e.y, e.z, e.name === 'charged' ? 'blue' : 'fire', e.name === 'charged' ? 40 : 30, 40);
        this.shake = Math.max(this.shake, e.name === 'finisher' ? 0.8 : 0.3);
        if (e.side) this.flare[e.side] = FLARE_S * 2;
        else this.flare = { l: FLARE_S * 2, r: FLARE_S * 2 };
        break;
      case 'wallPush':
        for (let i = 0; i < 60; i++) this.emit(e.x + rnd(-30, 30), e.y - rnd(0, 4), e.z, rnd(-8, 8), rnd(-70, -30), rnd(2, 6), rnd(0.4, 0.8), rnd(4, 8));
        this.shake = Math.max(this.shake, 0.4);
        this.flare = { l: FLARE_S * 2, r: FLARE_S * 2 };
        break;
      case 'stonePillar':
        this.burst(e.x, e.y - 4, e.z, 'earth', 30, 30);
        this.shake = Math.max(this.shake, 0.08);
        break;
      case 'slab': this.burst(e.x, e.y, e.z, 'spirit', 20, 30); break;
      // out of breath: the attack comes out as a puff of smoke
      case 'fizzle': this.burst(e.x, e.y, e.z, 'earth', 10, 12); break;
      case 'cut': this.burst(e.x, e.y, e.z, 'fire', 16, 30); break;
      case 'hitEnemy':
      case 'killEnemy': this.burst(e.x, e.y, e.z, 'fire', 30, 40); break;
      case 'clash': this.burst(e.x, e.y, e.z, 'spirit', 24, 34); break;
      case 'blocked':
        this.burst(e.x, e.y, 0, 'spirit', 20, 30);
        this.burst(e.x, e.y, 0, 'fire', 14, 24);
        this.shake = Math.max(this.shake, 0.25);
        break;
      // felt, not startling: a small shake and a soft red edge
      case 'playerHit': this.burst(e.x, e.y, 0, 'spirit', 26, 36); this.shake = Math.max(this.shake, 0.3); this.flash = 0.45; break;
    }
  }

  render(g: Game | null, dt: number): void {
    this.t += dt;
    this.cam = g ? g.cam : { x: 0, y: 0 };
    if (g) this.emitFromState(g, dt);
    this.updateParticles(dt);
    this.shake = Math.max(0, this.shake - dt * 2.5);
    this.flash = Math.max(0, this.flash - dt * 2);
    this.flare = { l: Math.max(0, this.flare.l - dt), r: Math.max(0, this.flare.r - dt) };


    const c = this.ctx, { W, H, M, u } = this;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    if (this.shake > 0) c.translate(rnd(-1, 1) * this.shake * 1.2 * u, rnd(-1, 1) * this.shake * 1.2 * u);
    // lean → the world tilts the other way (zoomed a touch so no edge shows)
    const tilt = -this.cam.x * LEAN_TILT;
    if (tilt) {
      c.translate(W / 2, H / 2);
      c.rotate(tilt);
      c.scale(1 + Math.abs(tilt) * 0.9, 1 + Math.abs(tilt) * 0.9);
      c.translate(-W / 2, -H / 2);
    }

    c.drawImage(this.sky, -M - this.cam.x * u * PAR_SKY, -M - this.cam.y * u * PAR_SKY, W + 2 * M, H + 2 * M);
    this.drawFloor();
    const mx = -M - this.cam.x * u * PAR_MID, my = -M - this.cam.y * u * PAR_MID;
    c.drawImage(this.mid, mx, my, W + 2 * M, H + 2 * M);
    this.drawLanterns(mx + M, my + M);
    if (g && g.groundFire > 0) this.drawGroundFire(g, true);

    if (g) [...g.enemies].sort((a, b) => b.z - a.z).forEach(e => this.drawEnemy(e));
    // near flames of the blue inferno burn in front of the enemies standing in them
    if (g && g.groundFire > 0) this.drawGroundFire(g, false);
    if (g) {
      g.walls.forEach(w => this.drawWall(w));
      // far pillars first, so nearer ones glow over them
      [...g.pillars].sort((a, b) => b.z - a.z).forEach(p => this.drawColumn(p.x, p.z, p.halfW, Math.min(1, p.age / 0.1)));
      [...g.hazards].sort((a, b) => b.z - a.z).forEach(h => this.drawHazard(g, h));
    }
    this.drawParticles(true);
    if (g) {
      this.drawProjectiles(g, true);
      this.drawLandingMarkers(g);
      g.blades.forEach(b => this.drawBlade(b));
      this.drawHandLight(g);
      this.drawProjectiles(g, false);
      this.drawAimReticles(g);
      if (g.xBlock) this.drawXBlock(g);
      this.drawHands(g);
      this.drawGhost();
      this.drawOffscreenHands(g);
    }
    this.drawParticles(false);
    if (g) {
      // a bright core in a hand that is releasing fire
      c.globalCompositeOperation = 'lighter';
      for (const side of ['l', 'r'] as const) {
        const h = g.hands[side];
        if (!h?.inView || this.flare[side] <= 0) continue;
        const p = this.viewToScreen(this.shownAt(h)), r = 3 * u * (this.flare[side] / FLARE_S) ** 0.5;
        c.drawImage(SPR.fire[0], p.x - r, p.y - r, r * 2, r * 2);
      }
      c.globalCompositeOperation = 'source-over';
    }

    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.drawImage(this.vig, 0, 0, W, H);
    if (g) this.drawPillarWarnings(g);
    if (this.flash > 0) {
      const gr = c.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.2, W / 2, H / 2, Math.max(W, H) * 0.7);
      gr.addColorStop(0, 'rgba(255,0,0,0)');
      gr.addColorStop(1, `rgba(255,30,30,${0.5 * this.flash})`);
      c.fillStyle = gr;
      c.fillRect(0, 0, W, H);
    }
  }

  // ---------- static layers ----------

  private frame(g: CanvasRenderingContext2D): Frame {
    g.setTransform(this.dpr, 0, 0, this.dpr, this.dpr * this.M, this.dpr * this.M);
    g.clearRect(-this.M, -this.M, this.W + 2 * this.M, this.H + 2 * this.M);
    return { g, W: this.W, H: this.H, M: this.M, u: this.u, hz: this.VP.y };
  }

  private drawSky(): void { paintSky(this.frame(this.sky.getContext('2d')!), this._scene); }

  private drawMid(): void { this.glows = paintMid(this.frame(this.mid.getContext('2d')!), this._scene); }

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
    const c = this.ctx, hz = this.VP.y, st = FLOORS[this._scene];
    const gr = c.createLinearGradient(0, hz, 0, this.H);
    gr.addColorStop(0, st.top); gr.addColorStop(1, st.bottom);
    c.fillStyle = gr;
    c.fillRect(0, hz, this.W, this.H - hz);
    c.strokeStyle = st.line;
    c.lineWidth = 1;
    const across = (zs: number[]) => {
      for (const z of zs) {
        const y = this.project(0, FLOOR_Y, z).y;
        c.beginPath(); c.moveTo(0, y); c.lineTo(this.W, y); c.stroke();
      }
    };
    const along = (step: number, from = -18, to = 18) => {
      for (let i = from; i <= to; i++) {
        const a = this.project(i * step, FLOOR_Y, 16), b = this.project(i * step, FLOOR_Y, -0.4);
        c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
      }
    };
    const depths = (n: number, far: number) => Array.from({ length: n }, (_, i) => far * (i / n) ** 1.6 - 0.3);
    switch (st.kind) {
      case 'grid':
        along((this.W * 0.09) / this.u);
        across([0, 0.5, 1.1, 1.9, 2.9, 4.2, 5.9, 8.1, 11, 15]);
        break;
      case 'tiles':
        along(14);
        across(depths(16, 16));
        break;
      case 'steps':
        c.lineWidth = 2;
        across(depths(12, 16));
        break;
      case 'planks':
        // boards run across the bridge; its edges are the rope rails' posts
        c.lineWidth = 1.5;
        across(depths(34, 16));
        c.strokeStyle = 'rgba(0,0,0,.35)';
        along(60, -1, 1);
        break;
      case 'sand': {
        // raked lines in long arcs round the arena
        for (let i = 1; i < 26; i++) {
          const z = 16 * (i / 26) ** 1.5 - 0.3, a = this.project(-200, FLOOR_Y, z), b = this.project(200, FLOOR_Y, z);
          const mid = this.project(0, FLOOR_Y, z * 0.92 - 0.1);
          c.beginPath(); c.moveTo(a.x, a.y); c.quadraticCurveTo(mid.x, mid.y, b.x, b.y); c.stroke();
        }
        break;
      }
      case 'dirt': {
        const R = mulberry32(5);
        c.fillStyle = st.line;
        for (let i = 0; i < 90; i++) {
          const p = this.project((R() - 0.5) * 260, FLOOR_Y, R() ** 1.7 * 15), r = (1 + R() * 3) * this.u * p.s;
          c.beginPath(); c.ellipse(p.x, p.y, r, r * 0.35, 0, 0, 7); c.fill();
        }
        // the trampled road up to the gate
        c.strokeStyle = 'rgba(0,0,0,.25)'; c.lineWidth = 2;
        along(40, -1, 1);
        break;
      }
    }
  }

  private drawLanterns(ox: number, oy: number): void {
    const c = this.ctx, u = this.u, [r0, g0, b0] = this.glows.color;
    c.globalCompositeOperation = 'lighter';
    this.glows.pts.forEach((l, i) => {
      const x = l.x + ox, y = l.y + oy, r = 7 * u * (0.75 + 0.25 * Math.sin(this.t * 9 + i * 2) * Math.sin(this.t * 5.3 + i));
      const gr = c.createRadialGradient(x, y, 0, x, y, r);
      gr.addColorStop(0, `rgba(${r0},${g0},${b0},.5)`); gr.addColorStop(1, `rgba(${r0},${g0},${b0},0)`);
      c.fillStyle = gr;
      c.fillRect(x - r, y - r, r * 2, r * 2);
    });
    c.globalCompositeOperation = 'source-over';
  }

  private drawEnemy(e: Enemy): void {
    if (e.dummy) {
      this.drawDummy(e);
      return;
    }
    if (e.earth) {
      if (e.boss?.wall) this.drawStoneWall(e, true);
      this.drawEarthbender(e);
      if (e.boss?.wall) this.drawStoneWall(e, false);
      return;
    }
    this.drawSpirit(e);
  }

  /**
   * A water spirit: a hooded, flowing shape with a white mask and glowing eyes, trailing off into a
   * wisp instead of legs. Its sleeve reaches out to the orb it's gathering, or both lift the wave.
   */
  private drawSpirit(e: Enemy): void {
    const c = this.ctx, u = this.u, p = this.project(e.x, e.y, e.z), s = p.s, x = p.x, k = u * s;
    const alpha = e.appear * (1 - Math.min(1, e.dying));
    if (alpha <= 0) return;
    const bob = Math.sin(e.t * 2 + e.phase) * 1.5 * k, cy = p.y + bob, feet = this.project(e.x, FLOOR_Y, e.z).y;
    const hit = e.flash > 0, sway = Math.sin(e.t * 1.7 + e.phase) * 2 * k;
    c.globalAlpha = alpha * 0.35;
    c.fillStyle = '#000';
    c.beginPath(); c.ellipse(x, feet, 7 * k, 1.6 * k, 0, 0, 7); c.fill();
    // aura
    c.globalCompositeOperation = 'lighter';
    c.globalAlpha = alpha;
    let gr = c.createRadialGradient(x, cy - 6 * k, 0, x, cy - 6 * k, 34 * k);
    gr.addColorStop(0, `rgba(80,190,255,${0.24 + e.flash * 0.4})`); gr.addColorStop(1, 'rgba(80,190,255,0)');
    c.fillStyle = gr;
    c.fillRect(x - 34 * k, cy - 40 * k, 68 * k, 68 * k);
    c.globalCompositeOperation = 'source-over';
    // the cloak: wide at the shoulders, falling to a wisp that trails and sways
    const sh = cy - 15 * k, tail = cy + 24 * k;
    gr = c.createLinearGradient(0, sh - 6 * k, 0, tail);
    gr.addColorStop(0, hit ? 'rgba(255,236,210,.95)' : 'rgba(40,110,170,.95)');
    gr.addColorStop(0.45, hit ? 'rgba(255,200,160,.7)' : 'rgba(60,150,220,.7)');
    gr.addColorStop(1, 'rgba(90,190,255,0)');
    c.fillStyle = gr;
    c.beginPath();
    c.moveTo(x - 4 * k, sh - 7 * k);
    c.quadraticCurveTo(x - 9 * k, sh - 3 * k, x - 9.5 * k, sh + 2 * k);
    c.bezierCurveTo(x - 10 * k, sh + 16 * k, x - 6 * k + sway, tail - 10 * k, x + sway * 2.2, tail);
    c.bezierCurveTo(x + 5 * k + sway, tail - 10 * k, x + 10 * k, sh + 16 * k, x + 9.5 * k, sh + 2 * k);
    c.quadraticCurveTo(x + 9 * k, sh - 3 * k, x + 4 * k, sh - 7 * k);
    c.closePath();
    c.fill();
    // flowing light down the cloak
    c.globalCompositeOperation = 'lighter';
    c.strokeStyle = 'rgba(150,230,255,.35)';
    c.lineWidth = Math.max(1, 0.5 * k);
    for (const dx of [-4, 0, 4]) {
      c.beginPath();
      c.moveTo(x + dx * k, sh + 2 * k);
      c.bezierCurveTo(x + dx * 1.2 * k, sh + 12 * k, x + dx * 0.4 * k + sway, tail - 12 * k, x + sway * 1.6 + dx * 0.2 * k, tail - 4 * k);
      c.stroke();
    }
    c.globalCompositeOperation = 'source-over';
    // sleeves: drifting at rest; one reaches to the orb it gathers; both rise for a wave
    for (const side of [-1, 1] as const) {
      const root = { x: x + side * 8 * k, y: sh + 1 * k };
      let hand = { x: x + side * 10 * k, y: sh + 11 * k + Math.sin(e.t * 2.4 + side) * 1.2 * k };
      if (e.winding && e.attack === 'slab') hand = { x: x + side * 9 * k, y: this.project(e.x, e.y - 28, e.z).y + bob };
      else if (e.winding && side === e.side) {
        const o = this.project(e.x + e.side * 11, e.y - 14, e.z);
        hand = { x: lerp(hand.x, o.x - side * 1.5 * k, e.wind), y: lerp(hand.y, o.y + bob, e.wind) };
      }
      c.fillStyle = hit ? 'rgba(255,230,200,.9)' : 'rgba(70,160,225,.85)';
      c.beginPath();
      c.moveTo(root.x - side * 1 * k, root.y - 2 * k);
      c.quadraticCurveTo((root.x + hand.x) / 2 + side * 3 * k, (root.y + hand.y) / 2 - 1 * k, hand.x + side * 2 * k, hand.y - 1.5 * k);
      c.lineTo(hand.x - side * 1 * k, hand.y + 2 * k);
      c.quadraticCurveTo((root.x + hand.x) / 2, (root.y + hand.y) / 2 + 2 * k, root.x - side * 2 * k, root.y + 4 * k);
      c.closePath();
      c.fill();
      c.fillStyle = hit ? '#fff4e4' : '#cfeeff';
      c.beginPath(); c.arc(hand.x, hand.y, 1.3 * k, 0, 7); c.fill();
    }
    // the hood
    const hy = sh - 6 * k;
    c.fillStyle = hit ? '#f0c8a0' : '#15365a';
    c.beginPath();
    c.moveTo(x, hy - 7.5 * k);
    c.bezierCurveTo(x + 6.5 * k, hy - 7 * k, x + 7.5 * k, hy + 1 * k, x + 6 * k, hy + 6 * k);
    c.lineTo(x - 6 * k, hy + 6 * k);
    c.bezierCurveTo(x - 7.5 * k, hy + 1 * k, x - 6.5 * k, hy - 7 * k, x, hy - 7.5 * k);
    c.fill();
    // the mask: pale, with glowing slit eyes and a water mark
    c.fillStyle = '#eef6f8';
    c.beginPath(); c.ellipse(x, hy + 0.4 * k, 3.9 * k, 4.8 * k, 0, 0, 7); c.fill();
    c.fillStyle = 'rgba(0,30,60,.25)';
    c.beginPath(); c.ellipse(x + 1.2 * k, hy + 1 * k, 2.6 * k, 4 * k, 0, -1.2, 1.2); c.fill();
    c.globalCompositeOperation = 'lighter';
    const glow = 0.7 + 0.3 * Math.sin(e.t * 4 + e.phase) + (e.winding ? 0.4 : 0);
    for (const side of [-1, 1]) {
      const ex = x + side * 1.5 * k, ey = hy - 0.2 * k;
      c.drawImage(SPR.spirit[0], ex - 2 * k * glow, ey - 2 * k * glow, 4 * k * glow, 4 * k * glow);
      c.fillStyle = '#e8fbff';
      c.beginPath(); c.ellipse(ex, ey, 1 * k, 0.35 * k, side * 0.3, 0, 7); c.fill();
    }
    c.globalCompositeOperation = 'source-over';
    c.strokeStyle = '#2a7ab8';
    c.lineWidth = Math.max(1, 0.45 * k);
    c.beginPath(); c.arc(x, hy - 3.2 * k, 0.9 * k, Math.PI * 0.1, Math.PI * 1.6); c.stroke();
    c.beginPath(); c.arc(x, hy + 2.8 * k, 1.1 * k, 0.2, Math.PI - 0.2); c.stroke();
    // wind-up telegraph: an orb in the hand, or a disc spinning up (sweep)
    if (e.winding && e.attack === 'slab') {
      const o = this.project(e.x, e.y - 30, e.z), rx = (4 + e.wind * 12) * k;
      c.globalCompositeOperation = 'lighter';
      c.strokeStyle = `rgba(140,230,255,${0.4 + 0.5 * e.wind})`;
      c.lineWidth = 3;
      c.beginPath(); c.ellipse(o.x, o.y + bob, rx, rx * 0.22, 0, e.t * 9, e.t * 9 + 5); c.stroke();
      c.globalCompositeOperation = 'source-over';
    } else if (e.winding) {
      const o = this.project(e.x + e.side * 11, e.y - 14, e.z), r = (1.2 + e.wind * 3.5) * k;
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

  /** Daro's stone wall: dressed blocks raised in front of him (the back half, then the front edge). */
  private drawStoneWall(e: Enemy, back: boolean): void {
    const c = this.ctx, z = e.z - 0.8, a = this.project(e.x - 16, FLOOR_Y, z), b = this.project(e.x + 16, FLOOR_Y - 45, z);
    if (back) {
      c.fillStyle = 'rgba(0,0,0,.4)';
      c.beginPath(); c.ellipse((a.x + b.x) / 2, a.y, (b.x - a.x) * 0.6, (b.x - a.x) * 0.06, 0, 0, 7); c.fill();
      return;
    }
    const w = b.x - a.x, h = a.y - b.y, rows = 5, R = mulberry32(e.id);
    c.fillStyle = '#6f5a44';
    c.fillRect(a.x, b.y, w, h);
    for (let r = 0; r < rows; r++) {
      const y0 = b.y + (h * r) / rows, rh = h / rows, off = r % 2 ? 0.5 : 0;
      for (let i = -1; i < 3; i++) {
        const x0 = a.x + w * ((i + off) / 3), x1 = x0 + w / 3, l = Math.max(a.x, x0), rr = Math.min(a.x + w, x1);
        if (rr <= l) continue;
        c.fillStyle = `hsl(28, ${18 + R() * 8}%, ${30 + R() * 10}%)`;
        c.fillRect(l + 1.5, y0 + 1.5, rr - l - 3, rh - 3);
        c.fillStyle = 'rgba(255,230,190,.12)';
        c.fillRect(l + 1.5, y0 + 1.5, rr - l - 3, rh * 0.18);
      }
    }
    c.strokeStyle = '#2d2014'; c.lineWidth = 2;
    c.strokeRect(a.x, b.y, w, h);
    c.strokeStyle = 'rgba(30,20,10,.7)'; c.lineWidth = 1.5;
    c.beginPath(); c.moveTo(a.x + w * 0.62, b.y); c.lineTo(a.x + w * 0.55, b.y + h * 0.3); c.lineTo(a.x + w * 0.66, b.y + h * 0.5); c.stroke();
  }

  /**
   * An earthbender: green tunic with a gold sash, brown trousers, boots, bare forearms wrapped in
   * cloth, topknot. Raising a pillar he stomps and lifts both arms (first 60% of the wind-up), then
   * drives both palms forward. Daro is bigger, with stone pauldrons, a beard and a cape.
   */
  private drawEarthbender(e: Enemy): void {
    const c = this.ctx, u = this.u, p = this.project(e.x, e.y, e.z), boss = !!e.boss, k = u * p.s * (boss ? 1.6 : 1), x = p.x;
    const feet = this.project(e.x, FLOOR_Y, e.z).y;
    const alpha = e.appear * (1 - Math.min(1, e.dying));
    if (alpha <= 0) return;
    const hit = e.flash > 0, w = e.winding ? e.wind : 0;
    const lift = e.attack === 'pillar' ? Math.min(1, w / 0.6) : 0, shove = e.attack === 'pillar' ? clamp((w - 0.6) / 0.4, 0, 1) : 0;
    const winded = boss && e.boss!.winded > 0, sag = winded ? 2.5 * k : 0;
    const crouch = (lift - shove) * 3 * k + sag;
    const col = (normal: string) => (hit ? '#fff0c8' : normal);
    const OUT = '#1a120a', skin = col('#c98f62'), tunic = col(boss ? '#3f5a2e' : '#4e6b3a'), tunicDark = col(boss ? '#2e4422' : '#3a5230');
    if (winded) {
      c.globalAlpha = alpha;
      c.globalCompositeOperation = 'lighter';
      const gr = c.createRadialGradient(x, p.y, 0, x, p.y, 24 * k);
      gr.addColorStop(0, 'rgba(255,220,120,.35)');
      gr.addColorStop(1, 'rgba(255,220,120,0)');
      c.fillStyle = gr;
      c.fillRect(x - 24 * k, p.y - 24 * k, 48 * k, 48 * k);
      c.globalCompositeOperation = 'source-over';
    }
    c.globalAlpha = alpha;
    c.lineJoin = 'round'; c.lineCap = 'round';
    c.fillStyle = 'rgba(0,0,0,.5)';
    c.beginPath(); c.ellipse(x, feet, 11 * k, 2.2 * k, 0, 0, 7); c.fill();
    const hipY = p.y + 8 * k + crouch, top = p.y - 14 * k + crouch;
    // Daro's cape, behind everything
    if (boss) {
      c.fillStyle = col('#2a3a1e');
      c.beginPath();
      c.moveTo(x - 7 * k, top + 1 * k); c.lineTo(x + 7 * k, top + 1 * k);
      c.quadraticCurveTo(x + 12 * k, hipY + 6 * k, x + 10 * k + Math.sin(e.t * 2) * k, hipY + 12 * k);
      c.lineTo(x - 10 * k + Math.sin(e.t * 2 + 1) * k, hipY + 12 * k);
      c.quadraticCurveTo(x - 12 * k, hipY + 6 * k, x - 7 * k, top + 1 * k);
      c.fill();
    }
    // legs: trousers bent at the knee in a wide, rooted stance, then boots
    c.strokeStyle = OUT;
    for (const side of [-1, 1]) {
      const hip = { x: x + side * 3 * k, y: hipY }, foot = { x: x + side * 8 * k, y: feet - 1.2 * k };
      const knee = { x: x + side * (7.5 * k + crouch * 0.4), y: (hip.y + foot.y) / 2 };
      c.fillStyle = col('#6b5234');
      c.beginPath();
      c.moveTo(hip.x - 2.6 * k, hip.y); c.lineTo(hip.x + 2.6 * k, hip.y);
      c.lineTo(knee.x + 2 * k, knee.y); c.lineTo(foot.x + 1.6 * k, foot.y - 2.5 * k);
      c.lineTo(foot.x - 1.6 * k, foot.y - 2.5 * k); c.lineTo(knee.x - 2 * k, knee.y);
      c.closePath(); c.fill();
      c.lineWidth = Math.max(1, 0.35 * k); c.stroke();
      c.fillStyle = col('#2a1e14');
      c.beginPath(); c.ellipse(foot.x + side * 0.8 * k, foot.y - 0.6 * k, 2.6 * k, 1.5 * k, 0, 0, 7); c.fill();
      c.fillRect(foot.x - 1.7 * k, foot.y - 3.6 * k, 3.4 * k, 3 * k);
    }
    // tunic: broad shoulders, cinched at the sash, flaring below it
    c.fillStyle = tunic;
    c.beginPath();
    c.moveTo(x - 7.5 * k, top + 1 * k);
    c.quadraticCurveTo(x, top - 1 * k, x + 7.5 * k, top + 1 * k);
    c.lineTo(x + 5.2 * k, hipY - 3 * k);
    c.lineTo(x + 7.5 * k, hipY + 4 * k);
    c.lineTo(x - 7.5 * k, hipY + 4 * k);
    c.lineTo(x - 5.2 * k, hipY - 3 * k);
    c.closePath(); c.fill();
    c.lineWidth = Math.max(1, 0.35 * k); c.strokeStyle = OUT; c.stroke();
    c.fillStyle = tunicDark;
    c.beginPath(); c.moveTo(x + 7.5 * k, top + 1 * k); c.lineTo(x + 5.2 * k, hipY - 3 * k); c.lineTo(x + 7.5 * k, hipY + 4 * k); c.lineTo(x + 3.5 * k, hipY + 4 * k); c.lineTo(x + 3 * k, top + 0.5 * k); c.closePath(); c.fill();
    // crossed collar and the gold sash with its knot
    c.strokeStyle = col('#c8a860'); c.lineWidth = Math.max(1, 0.7 * k);
    c.beginPath(); c.moveTo(x - 3 * k, top); c.lineTo(x + 1.5 * k, top + 6 * k); c.moveTo(x + 3 * k, top); c.lineTo(x - 0.5 * k, top + 4 * k); c.stroke();
    c.fillStyle = col(boss ? '#8a6a3a' : '#b58a3c');
    c.fillRect(x - 5.6 * k, hipY - 4 * k, 11.2 * k, 2.6 * k);
    c.beginPath(); c.moveTo(x + 2 * k, hipY - 1.4 * k); c.lineTo(x + 3.4 * k, hipY + 4 * k); c.lineTo(x + 1.2 * k, hipY + 3.6 * k); c.closePath(); c.fill();
    if (boss) {
      c.fillStyle = col('#7a746a');
      c.beginPath(); c.arc(x, hipY - 2.7 * k, 1.5 * k, 0, 7); c.fill();
    }
    // arms: rest → raised (lifting the stone) → driven forward (shoving it)
    for (const side of [-1, 1]) {
      const sh = { x: x + side * 6.8 * k, y: top + 2 * k };
      let hand = { x: sh.x + side * 3.5 * k, y: sh.y + 12 * k };
      if (e.attack === 'pillar' && e.winding) {
        const up = { x: sh.x + side * 5 * k, y: sh.y - 10 * k }, fwd = { x: sh.x + side * 1.5 * k, y: sh.y + 3 * k };
        hand = { x: lerp(lerp(hand.x, up.x, lift), fwd.x, shove), y: lerp(lerp(hand.y, up.y, lift), fwd.y, shove) };
      }
      const elbow = { x: (sh.x + hand.x) / 2 + side * 2.4 * k, y: (sh.y + hand.y) / 2 + 1 * k };
      const limb = (a: { x: number; y: number }, b: { x: number; y: number }, width: number, fill: string) => {
        c.strokeStyle = OUT; c.lineWidth = width + Math.max(1.5, 0.7 * k);
        c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
        c.strokeStyle = fill; c.lineWidth = width;
        c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
      };
      limb(sh, elbow, 3.4 * k, tunic);
      limb(elbow, hand, 2.5 * k, skin);
      c.strokeStyle = col('#d8c8a0'); c.lineWidth = Math.max(1, 0.5 * k);
      for (const t of [0.35, 0.6]) {
        const q = { x: lerp(elbow.x, hand.x, t), y: lerp(elbow.y, hand.y, t) };
        c.beginPath(); c.moveTo(q.x - 1.2 * k, q.y - 0.4 * k); c.lineTo(q.x + 1.2 * k, q.y + 0.4 * k); c.stroke();
      }
      c.fillStyle = skin; c.strokeStyle = OUT; c.lineWidth = Math.max(1, 0.35 * k);
      c.beginPath(); c.arc(hand.x, hand.y, 1.8 * k, 0, 7); c.fill(); c.stroke();
      if (boss) {
        // stone pauldrons
        c.fillStyle = col('#7d6d5a');
        c.beginPath(); c.ellipse(sh.x + side * 0.6 * k, sh.y - 0.4 * k, 3.6 * k, 2.6 * k, side * 0.3, 0, 7); c.fill(); c.stroke();
        c.strokeStyle = 'rgba(0,0,0,.35)';
        c.beginPath(); c.moveTo(sh.x - 2 * k, sh.y); c.lineTo(sh.x + 2 * k, sh.y + 0.5 * k); c.stroke();
      }
    }
    // head: neck, face, stern brows, topknot (Daro: a beard)
    const hy = top - 5 * k;
    c.fillStyle = skin;
    c.fillRect(x - 1.6 * k, hy + 2 * k, 3.2 * k, 3.5 * k);
    c.strokeStyle = OUT; c.lineWidth = Math.max(1, 0.35 * k);
    c.beginPath(); c.ellipse(x, hy, 3.6 * k, 4.2 * k, 0, 0, 7); c.fill(); c.stroke();
    c.fillStyle = '#1e1a16';
    c.beginPath(); c.ellipse(x, hy - 1.8 * k, 3.8 * k, 2.8 * k, 0, Math.PI, 0); c.fill();
    c.beginPath(); c.arc(x, hy - 5.2 * k, 1.5 * k, 0, 7); c.fill();
    c.fillStyle = col('#4e6b3a');
    c.fillRect(x - 3.8 * k, hy - 2.3 * k, 7.6 * k, 0.9 * k);
    c.fillStyle = '#1a120a';
    for (const side of [-1, 1]) {
      c.beginPath(); c.arc(x + side * 1.4 * k, hy + 0.2 * k, 0.45 * k, 0, 7); c.fill();
      c.strokeStyle = '#1a120a'; c.lineWidth = Math.max(1, 0.45 * k);
      c.beginPath(); c.moveTo(x + side * 2.3 * k, hy - 1 * k); c.lineTo(x + side * 0.6 * k, hy - 0.4 * k); c.stroke();
    }
    if (boss) {
      c.fillStyle = '#2a1e16';
      c.beginPath(); c.moveTo(x - 3.4 * k, hy + 0.8 * k); c.quadraticCurveTo(x, hy + 7 * k, x + 3.4 * k, hy + 0.8 * k); c.lineTo(x + 2 * k, hy + 2 * k); c.lineTo(x - 2 * k, hy + 2 * k); c.closePath(); c.fill();
    } else {
      c.strokeStyle = '#6a3a24'; c.lineWidth = Math.max(1, 0.35 * k);
      c.beginPath(); c.moveTo(x - 1 * k, hy + 2.2 * k); c.lineTo(x + 1 * k, hy + 2.2 * k); c.stroke();
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
      const q = this.project(p.x, p.y, p.z), r = p.r * q.s * this.u;
      if (p.shot === 'charged') { this.drawDragon(p.id, q.x, q.y, r); continue; }
      const set = p.kind === 'enemy' ? SPR.spirit : SPR.fire;
      c.drawImage(set[1], q.x - r * 2.4, q.y - r * 2.4, r * 4.8, r * 4.8);
      c.drawImage(set[0], q.x - r * 1.2, q.y - r * 1.2, r * 2.4, r * 2.4);
      if (p.shot === 'counter' || p.shot === 'flurry') {
        // a white-hot ring marks a combo shot
        c.strokeStyle = p.shot === 'counter' ? 'rgba(255,255,255,.8)' : 'rgba(255,220,150,.7)';
        c.lineWidth = Math.max(1.5, r * 0.18);
        c.beginPath(); c.arc(q.x, q.y, r * 1.5, this.t * 8, this.t * 8 + 4.5); c.stroke();
      }
    }
    c.globalCompositeOperation = 'source-over';
    // forget the trails of shots that are gone
    for (const id of this.dragonTrails.keys()) if (!g.projs.some(p => p.id === id)) this.dragonTrails.delete(id);
  }

  /**
   * A charged shot: a dragon of blue fire — a horned head with open jaws leading, its body a
   * sinuous ribbon of flame writhing along the path it has flown (screen space, drawn additively).
   */
  private drawDragon(id: number, x: number, y: number, r0: number): void {
    // drawn well bigger than the shot's hit size, so it reads as a dragon even far away
    const c = this.ctx, trail = this.dragonTrails.get(id) ?? [], r = Math.max(r0 * 2, 2.2 * this.u);
    const last = trail[trail.length - 1];
    if (!last || Math.hypot(last.x - x, last.y - y) > r * 0.25) trail.push({ x, y, r });
    if (trail.length > 26) trail.shift();
    this.dragonTrails.set(id, trail);
    // heading: from a little way back along the trail
    const back = trail[Math.max(0, trail.length - 4)];
    let ang = Math.atan2(y - back.y, x - back.x);
    if (trail.length < 2) ang = -Math.PI / 2;
    // the body: a ribbon through the trail, waving side to side, tapering to the tail
    if (trail.length >= 2) {
      const L: { x: number; y: number }[] = [], R: { x: number; y: number }[] = [];
      for (let i = 0; i < trail.length; i++) {
        const a = trail[Math.max(0, i - 1)], b = trail[Math.min(trail.length - 1, i + 1)];
        const dir = Math.atan2(b.y - a.y, b.x - a.x) + Math.PI / 2, k = i / (trail.length - 1);
        const wave = Math.sin(this.t * 12 - i * 0.7) * trail[i].r * 0.7 * (0.3 + 0.7 * k);
        const w = trail[i].r * (0.15 + 0.6 * k) * (1 + 0.15 * Math.sin(this.t * 20 + i * 2));
        const cx = trail[i].x + Math.cos(dir) * wave, cy = trail[i].y + Math.sin(dir) * wave;
        L.push({ x: cx + Math.cos(dir) * w, y: cy + Math.sin(dir) * w });
        R.push({ x: cx - Math.cos(dir) * w, y: cy - Math.sin(dir) * w });
      }
      const tail = trail[0];
      const gr = c.createLinearGradient(tail.x, tail.y, x, y);
      gr.addColorStop(0, 'rgba(40,70,230,0)');
      gr.addColorStop(0.45, 'rgba(50,95,240,.75)');
      gr.addColorStop(1, 'rgba(90,150,255,.95)');
      // the body in solid blue (so it stays blue on bright ground), a hot glow down its middle
      c.globalCompositeOperation = 'source-over';
      c.fillStyle = gr;
      c.beginPath();
      L.forEach((q, i) => (i ? c.lineTo(q.x, q.y) : c.moveTo(q.x, q.y)));
      for (let i = R.length - 1; i >= 0; i--) c.lineTo(R[i].x, R[i].y);
      c.closePath();
      c.fill();
      c.globalCompositeOperation = 'lighter';
      c.strokeStyle = 'rgba(160,210,255,.55)';
      c.lineCap = 'round';
      for (let i = 1; i < trail.length; i++) {
        const a = L[i - 1], b = L[i], ra = R[i - 1], rb = R[i];
        c.lineWidth = trail[i].r * 0.35 * (i / trail.length);
        c.beginPath(); c.moveTo((a.x + ra.x) / 2, (a.y + ra.y) / 2); c.lineTo((b.x + rb.x) / 2, (b.y + rb.y) / 2); c.stroke();
      }
      // spines of flame along its back
      c.fillStyle = 'rgba(120,180,255,.6)';
      for (let i = 2; i < L.length - 1; i += 2) {
        const q = L[i], n = trail[i], d = Math.atan2(q.y - n.y, q.x - n.x), h = n.r * 0.7 * (i / L.length);
        c.beginPath();
        c.moveTo(q.x + Math.cos(d + 1.2) * h * 0.4, q.y + Math.sin(d + 1.2) * h * 0.4);
        c.lineTo(q.x + Math.cos(d) * h, q.y + Math.sin(d) * h);
        c.lineTo(q.x + Math.cos(d - 1.2) * h * 0.4, q.y + Math.sin(d - 1.2) * h * 0.4);
        c.fill();
      }
    }
    // a glow around the head
    c.drawImage(SPR.blue[1], x - r * 2, y - r * 2, r * 4, r * 4);
    // the head, solid so its shape reads: drawn pointing along +x then turned to its heading
    c.save();
    c.globalCompositeOperation = 'source-over';
    // in profile, facing the way it's heading (left or right), tilted up or down at most ~40°
    const dx = Math.cos(ang), dy = Math.sin(ang), face = dx < -0.05 ? -1 : 1;
    const tilt = clamp(Math.atan2(dy, Math.abs(dx)), -0.7, 0.7);
    c.translate(x, y);
    c.scale(face, 1);
    c.rotate(tilt);
    c.strokeStyle = 'rgba(16,30,110,.85)';
    c.lineWidth = Math.max(1, r * 0.08);
    c.lineJoin = 'round';
    const s = r * 1.4, bite = 0.12 + 0.12 * Math.sin(this.t * 10 + id);
    // a mane of flame streaming back off the skull
    c.globalCompositeOperation = 'lighter';
    for (let i = 0; i < 6; i++) {
      const bx = -s * (0.2 + i * 0.18), by = -s * (0.45 - i * 0.05), len = s * (0.8 + 0.3 * Math.sin(this.t * 11 + i * 1.7));
      const g2 = c.createLinearGradient(bx, by, bx - len, by - len * 0.3);
      g2.addColorStop(0, 'rgba(120,180,255,.9)'); g2.addColorStop(1, 'rgba(60,90,255,0)');
      c.fillStyle = g2;
      c.beginPath();
      c.moveTo(bx + s * 0.1, by);
      c.quadraticCurveTo(bx - len * 0.4, by - len * 0.45, bx - len, by - len * 0.25);
      c.quadraticCurveTo(bx - len * 0.5, by + len * 0.05, bx - s * 0.1, by + s * 0.2);
      c.fill();
    }
    c.globalCompositeOperation = 'source-over';
    // antler horns swept back from the crown
    c.lineCap = 'round';
    for (const [ox, oy, k] of [[-0.25, -0.5, 1], [-0.05, -0.52, 0.8]] as const) {
      c.strokeStyle = '#10206a';
      c.lineWidth = s * 0.16 * k + 2;
      c.beginPath(); c.moveTo(s * ox, s * oy); c.quadraticCurveTo(s * (ox - 0.4), s * (oy - 0.45), s * (ox - 1.0 * k), s * (oy - 0.55 * k)); c.stroke();
      c.strokeStyle = '#cfe4ff';
      c.lineWidth = s * 0.16 * k;
      c.beginPath(); c.moveTo(s * ox, s * oy); c.quadraticCurveTo(s * (ox - 0.4), s * (oy - 0.45), s * (ox - 1.0 * k), s * (oy - 0.55 * k)); c.stroke();
      c.lineWidth = s * 0.09 * k;
      c.beginPath(); c.moveTo(s * (ox - 0.45), s * (oy - 0.33)); c.lineTo(s * (ox - 0.5), s * (oy - 0.7 * k)); c.stroke();
    }
    const head = c.createLinearGradient(-s * 0.7, -s * 0.5, s * 1.1, 0);
    head.addColorStop(0, '#2449d8');
    head.addColorStop(0.55, '#5f98ff');
    head.addColorStop(1, '#cfe6ff');
    c.fillStyle = head;
    c.strokeStyle = '#10206a';
    c.lineWidth = Math.max(1, s * 0.06);
    // skull and upper jaw: crown, brow, long snout, bulbous nose, lip back to the corner of the mouth
    c.beginPath();
    c.moveTo(-s * 0.7, s * 0.12);
    c.quadraticCurveTo(-s * 0.65, -s * 0.5, -s * 0.2, -s * 0.55);
    c.quadraticCurveTo(s * 0.1, -s * 0.62, s * 0.25, -s * 0.46);
    c.quadraticCurveTo(s * 0.55, -s * 0.36, s * 0.88, -s * 0.36);
    c.quadraticCurveTo(s * 1.15, -s * 0.34, s * 1.1, -s * 0.12);
    c.quadraticCurveTo(s * 1.08, -s * 0.02, s * 0.95, -s * 0.02);
    c.lineTo(s * 0.15, s * 0.02);
    c.closePath();
    c.fill(); c.stroke();
    // lower jaw, opening and closing
    c.beginPath();
    c.moveTo(s * 0.12, s * 0.08);
    c.lineTo(s * 0.9, s * (0.1 + bite));
    c.quadraticCurveTo(s * 0.92, s * (0.24 + bite), s * 0.6, s * (0.26 + bite * 0.6));
    c.quadraticCurveTo(s * 0.0, s * 0.42, -s * 0.55, s * 0.3);
    c.lineTo(-s * 0.7, s * 0.12);
    c.closePath();
    c.fill(); c.stroke();
    // teeth
    c.fillStyle = '#f2f8ff';
    for (let i = 0; i < 4; i++) {
      const tx = s * (0.3 + i * 0.16);
      c.beginPath(); c.moveTo(tx, s * 0.02); c.lineTo(tx + s * 0.05, s * 0.13); c.lineTo(tx + s * 0.1, s * 0.02); c.fill();
    }
    // fire in its mouth
    c.globalCompositeOperation = 'lighter';
    c.drawImage(SPR.blue[0], s * 0.75 - s * 0.35, s * (0.06 + bite / 2) - s * 0.35, s * 0.7, s * 0.7);
    c.globalCompositeOperation = 'source-over';
    // nostril, brow and a white-hot eye
    c.fillStyle = '#10206a';
    c.beginPath(); c.ellipse(s * 0.98, -s * 0.24, s * 0.05, s * 0.03, 0.3, 0, 7); c.fill();
    c.beginPath(); c.moveTo(-s * 0.05, -s * 0.4); c.quadraticCurveTo(s * 0.2, -s * 0.52, s * 0.42, -s * 0.34); c.lineTo(s * 0.34, -s * 0.3); c.quadraticCurveTo(s * 0.18, -s * 0.42, -s * 0.03, -s * 0.33); c.fill();
    c.fillStyle = '#ffffff';
    c.beginPath(); c.ellipse(s * 0.2, -s * 0.28, s * 0.11, s * 0.07, -0.15, 0, 7); c.fill();
    c.fillStyle = '#10206a';
    c.beginPath(); c.ellipse(s * 0.23, -s * 0.28, s * 0.03, s * 0.065, -0.15, 0, 7); c.fill();
    // long whiskers from the nose, streaming back
    c.strokeStyle = 'rgba(170,210,255,.85)';
    c.lineWidth = Math.max(1, s * 0.05);
    for (const sgn of [0, 1]) {
      c.beginPath();
      c.moveTo(s * 1.0, -s * 0.12 + sgn * s * 0.1);
      c.bezierCurveTo(s * 0.6, s * (0.5 + sgn * 0.2), -s * 0.3, s * (0.1 + 0.25 * Math.sin(this.t * 8 + sgn * 2)), -s * 1.3, s * (0.45 + sgn * 0.25));
      c.stroke();
    }
    c.restore();
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

  /** How strongly a hand burns right now: attacking, shielding or charged, else not at all. */
  private burning(g: Game, side: Side): number {
    return Math.max(g.shield.on ? 1 : 0, Math.min(1, this.flare[side] / FLARE_S));
  }

  /** Warm light around a hand that is releasing fire, and the shield's flame sheet. */
  private drawHandLight(g: Game): void {
    const c = this.ctx, u = this.u, flicker = 0.9 + 0.1 * Math.sin(this.t * 20);
    c.globalCompositeOperation = 'lighter';
    for (const side of ['l', 'r'] as const) {
      const h = g.hands[side], burn = this.burning(g, side);
      if (h?.inView && h.charge > 0) {
        // a charging fist glows blue, brighter as it fills; charged, it pulses
        const C = this.viewToScreen(this.shownAt(h)), full = h.charge >= 1, rr = (14 + 16 * h.charge) * u;
        const pulse = full ? 0.8 + 0.2 * Math.sin(this.t * 10) : 1;
        const gr = c.createRadialGradient(C.x, C.y, 0, C.x, C.y, rr);
        gr.addColorStop(0, `rgba(140,190,255,${(0.2 + 0.35 * h.charge) * pulse})`); gr.addColorStop(1, 'rgba(60,110,255,0)');
        c.fillStyle = gr;
        c.fillRect(C.x - rr, C.y - rr, rr * 2, rr * 2);
      }
      if (!h?.inView || burn <= 0) continue;
      const C = this.viewToScreen(this.shownAt(h)), r = 32 * u;
      const gr = c.createRadialGradient(C.x, C.y, 0, C.x, C.y, r);
      gr.addColorStop(0, `rgba(255,140,60,${0.26 * burn * flicker})`); gr.addColorStop(1, 'rgba(255,120,40,0)');
      c.fillStyle = gr;
      c.fillRect(C.x - r, C.y - r, r * 2, r * 2);
    }
    const { l, r } = g.hands;
    // finisher gather: a ball of fire growing between the hands; gathered (and charged), it blazes
    if (g.gather > 0 && l?.inView && r?.inView) {
      const a = this.viewToScreen(l.pos), b = this.viewToScreen(r.pos), m = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const ready = g.ultimateIn <= 0, full = g.gather >= 1, pulse = full ? 0.85 + 0.15 * Math.sin(this.t * 12) : 1;
      const rr = (6 + 12 * g.gather) * u * pulse;
      const glow = c.createRadialGradient(m.x, m.y, 0, m.x, m.y, rr * 2.2);
      glow.addColorStop(0, ready ? `rgba(255,200,110,${0.25 + 0.45 * g.gather})` : 'rgba(170,150,140,.25)');
      glow.addColorStop(1, 'rgba(255,120,40,0)');
      c.fillStyle = glow;
      c.fillRect(m.x - rr * 2.2, m.y - rr * 2.2, rr * 4.4, rr * 4.4);
      if (ready) c.drawImage(SPR.fire[0], m.x - rr * 0.6, m.y - rr * 0.6, rr * 1.2, rr * 1.2);
      if (full && ready) {
        // a ring closing in: spread now
        c.strokeStyle = `rgba(255,220,150,${0.5 + 0.3 * Math.sin(this.t * 8)})`;
        c.lineWidth = 3;
        c.beginPath(); c.arc(m.x, m.y, rr * 1.6, 0, 7); c.stroke();
      }
    }
    if (g.shield.on && l && r) {
      const a = this.viewToScreen(l.pos), b = this.viewToScreen(r.pos), e = 1, hgt = (14 + e * 10) * u;
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
  private handShape(
    c: CanvasRenderingContext2D, h: Vec2, side: number, grow: number, open: boolean, elbowAt: Vec2 | null = null,
    /** > 1 draws the hand bigger, e.g. a fist punching toward you. */
    scale = 1,
  ): void {
    const k = 1.5 * this.u * scale, g = grow;
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

  /** 0 → 1 as a fist drives out from guard to a full punch. */
  private punchOut(h: NonNullable<Game['hands']['l']>): number {
    return h.reach === null || h.reachBase === null || h.open ? 0 : clamp((h.reach - h.reachBase) / 0.3, 0, 1);
  }

  /**
   * How much a fist is resting close to your body: 1 in guard, 0 punched all the way out (open
   * hands and fists already low, e.g. at the hip, don't count).
   */
  private resting(h: NonNullable<Game['hands']['l']>): number {
    return h.open ? 0 : (1 - this.punchOut(h)) * clamp((REST_LOW_Y - h.pos.y) / 24, 0, 1);
  }

  /**
   * Where to draw a hand (view units): a fist resting in guard sits low and close to you, as your
   * own fists look from your eyes; it rises to where it's aimed only as it punches out.
   */
  private shownAt(h: NonNullable<Game['hands']['l']>, p: Vec2 = h.pos): Vec2 {
    return { x: p.x, y: p.y + REST_DROP * this.resting(h) };
  }

  /** Drawn size: a resting fist is close to the camera (bigger); punched out it's further away. */
  private shownScale(h: NonNullable<Game['hands']['l']>): number {
    return h.open ? 1 : 1.05 + 0.25 * this.resting(h);
  }

  private drawHands(g: Game): void {
    const hands = ([['l', -1], ['r', 1]] as const)
      .map(([side, sign]) => ({ h: g.hands[side], sign, burn: this.burning(g, side) }))
      .filter((x): x is { h: NonNullable<typeof x.h>; sign: -1 | 1; burn: number } => x.h !== null && x.h.inView);
    if (!hands.length) return;
    const c = this.ctx, u = this.u, flicker = 0.9 + 0.1 * Math.sin(this.t * 20);
    // rim glow first, then the dark hands, lit on top by their own fire
    // idle hands keep only a faint warm rim so you can see them; burning hands glow
    for (const { h, sign, burn } of hands) {
      c.strokeStyle = c.fillStyle = `rgba(255,130,60,${(0.12 + 0.3 * burn) * flicker})`;
      this.handShape(c, this.viewToScreen(this.shownAt(h)), sign, 0.9 * u, h.open, h.elbow && this.viewToScreen(this.shownAt(h, h.elbow)), this.shownScale(h));
    }
    const hl = this.handLayer.getContext('2d')!;
    hl.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    hl.globalCompositeOperation = 'source-over';
    hl.clearRect(0, 0, this.W, this.H);
    hl.strokeStyle = hl.fillStyle = g.inv > 0 && Math.sin(this.t * 40) > 0 ? '#3a1216' : '#150f19';
    for (const { h, sign } of hands) {
      this.handShape(hl, this.viewToScreen(this.shownAt(h)), sign, 0, h.open, h.elbow && this.viewToScreen(this.shownAt(h, h.elbow)), this.shownScale(h));
    }
    hl.globalCompositeOperation = 'source-atop';
    for (const { h, burn } of hands) {
      const C = this.viewToScreen(this.shownAt(h)), gr = hl.createRadialGradient(C.x, C.y - 3 * u, 0, C.x, C.y, 30 * u);
      gr.addColorStop(0, `rgba(255,160,80,${(0.18 + 0.62 * burn) * flicker})`); gr.addColorStop(1, 'rgba(255,90,30,0)');
      hl.fillStyle = gr;
      hl.fillRect(0, 0, this.W, this.H);
    }
    c.drawImage(this.handLayer, 0, 0, this.W, this.H);
  }

  /** Translucent hands demonstrating the move being learned, over your own. */
  private drawGhost(): void {
    if (!this.ghost) return;
    const pose = ghostPose(this.ghost.lessonId, this.t);
    if (!pose) return;
    const c = this.ctx, a = this.ghost.alpha * (0.55 + 0.1 * Math.sin(this.t * 3));
    c.save();
    c.globalAlpha = a;
    c.globalCompositeOperation = 'lighter';
    c.strokeStyle = c.fillStyle = 'rgba(160,210,255,0.55)';
    for (const [side, sign] of [['l', -1], ['r', 1]] as const) {
      const h = pose[side];
      this.handShape(c, this.viewToScreen(h.pos), sign, 0.6 * this.u, h.open, null, h.scale);
    }
    c.restore();
  }

  /**
   * Fist punches: a small ring on whatever a punch from each ready fist would hit (or where it
   * would fly), so you can line up before you throw. Uses the game's own aim.
   */
  private drawAimReticles(g: Game): void {
    if (TUNING.punchTrigger !== 'extend' || g.shield.on || g.xBlock) return;
    const c = this.ctx, u = this.u;
    for (const side of ['l', 'r'] as const) {
      const h = g.hands[side];
      if (!h?.inView || h.open || !h.punchReady) continue;
      const aim = g.previewAim(side, h.pos, h.aimDir);
      const p = aim.target ? this.project(aim.target.x, aim.target.y, aim.target.z) : this.project(aim.point.x, aim.point.y, aim.depth);
      const r = (aim.target ? 9 : 3) * u * p.s + 1.2 * u;
      c.strokeStyle = aim.target ? 'rgba(255,190,110,.75)' : 'rgba(255,190,110,.35)';
      c.lineWidth = 2;
      c.beginPath(); c.arc(p.x, p.y, r, 0, 7); c.stroke();
      // a tick on the side of the fist that's aiming
      const dx = side === 'l' ? -1 : 1;
      c.beginPath(); c.moveTo(p.x + dx * r, p.y); c.lineTo(p.x + dx * (r + 1.2 * u), p.y); c.stroke();
    }
  }

  /** Crossed forearms: a big X of fire across the hands. */
  private drawXBlock(g: Game): void {
    const { l, r } = g.hands;
    if (!l || !r) return;
    const c = this.ctx, u = this.u, a = this.viewToScreen(l.pos), b = this.viewToScreen(r.pos);
    const C = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 - 4 * u }, L = 24 * u, flick = 0.85 + 0.15 * Math.sin(this.t * 22);
    c.globalCompositeOperation = 'lighter';
    const gr = c.createRadialGradient(C.x, C.y, 0, C.x, C.y, L * 1.4);
    gr.addColorStop(0, `rgba(255,150,60,${0.35 * flick})`); gr.addColorStop(1, 'rgba(255,100,30,0)');
    c.fillStyle = gr;
    c.fillRect(C.x - L * 1.4, C.y - L * 1.4, L * 2.8, L * 2.8);
    c.lineCap = 'round';
    for (const [w, col] of [[5 * u, `rgba(255,110,40,${0.45 * flick})`], [2.2 * u, `rgba(255,190,110,${0.8 * flick})`], [0.8 * u, 'rgba(255,245,215,.95)']] as const) {
      c.strokeStyle = col;
      c.lineWidth = w;
      c.beginPath();
      c.moveTo(C.x - L, C.y - L); c.lineTo(C.x + L, C.y + L);
      c.moveTo(C.x + L, C.y - L); c.lineTo(C.x - L, C.y + L);
      c.stroke();
    }
    c.globalCompositeOperation = 'source-over';
  }

  /** A glowing curtain behind a fire wall's flames, fading as it burns out. */
  private drawWall(w: Wall): void {
    const c = this.ctx, fade = Math.min(1, w.life / 0.6), top = FLOOR_Y - 75;
    const a = this.project(w.x - w.halfW, FLOOR_Y, w.z), b = this.project(w.x + w.halfW, FLOOR_Y, w.z);
    const at = this.project(w.x - w.halfW, top, w.z), bt = this.project(w.x + w.halfW, top, w.z);
    const gr = c.createLinearGradient(0, a.y, 0, at.y);
    gr.addColorStop(0, `rgba(255,140,50,${0.45 * fade})`);
    gr.addColorStop(0.6, `rgba(255,90,30,${0.18 * fade})`);
    gr.addColorStop(1, 'rgba(255,60,20,0)');
    c.globalCompositeOperation = 'lighter';
    c.fillStyle = gr;
    c.beginPath();
    c.moveTo(a.x, a.y);
    c.lineTo(b.x, b.y);
    for (let i = 0; i <= 12; i++) {
      const k = 1 - i / 12, x = lerp(at.x, bt.x, k);
      c.lineTo(x, lerp(at.y, bt.y, k) + (a.y - at.y) * 0.25 * (1 + Math.sin(k * 17 + this.t * 9)));
    }
    c.closePath();
    c.fill();
    c.globalCompositeOperation = 'source-over';
  }

  /**
   * The blue inferno's burning ground: a blue glow over the floor and a field of flames standing on
   * it, each flickering and swaying on its own, bigger up close. `far`: the glow and the flames
   * behind the enemies; otherwise the near ones in front of them.
   */
  private drawGroundFire(g: Game, far: boolean): void {
    const c = this.ctx, hz = this.VP.y, u = this.u;
    const fade = Math.min(1, g.groundFire / 0.8), rise = Math.min(1, (TUNE.infernoS - g.groundFire) / 0.35);
    const flick = 0.85 + 0.15 * Math.sin(this.t * 17) * Math.sin(this.t * 7.3);
    c.globalCompositeOperation = 'lighter';
    if (far) {
      const gr = c.createLinearGradient(0, hz, 0, this.H);
      gr.addColorStop(0, `rgba(60,110,255,${0.2 * fade})`);
      gr.addColorStop(0.4, `rgba(70,130,255,${0.3 * fade * flick})`);
      gr.addColorStop(1, `rgba(120,190,255,${0.4 * fade * flick})`);
      c.fillStyle = gr;
      c.fillRect(0, hz, this.W, this.H - hz);
    }
    for (const f of this.infernoFlames) {
      if ((f.z > INFERNO_NEAR_Z) !== far) continue;
      const p = this.project(g.cam.x + f.x, FLOOR_Y, f.z);
      if (p.x < -60 || p.x > this.W + 60) continue;
      const k = u * p.s, t = this.t * f.speed + f.phase;
      const h = f.h * k * fade * rise * (0.75 + 0.25 * Math.sin(t * 2.3) + 0.12 * Math.sin(t * 5.1));
      if (h < 2) continue;
      const w = f.w * k, sway = Math.sin(t * 1.7) * w * 0.9 + Math.sin(t * 4.3) * w * 0.25;
      this.flameTongue(c, p.x, p.y, w, h, sway, fade);
    }
    c.globalCompositeOperation = 'source-over';
  }

  /** One flame standing at (x, y): a curling tongue, blue at the edges, white-blue at its base. */
  private flameTongue(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, sway: number, a: number): void {
    const tip = { x: x + sway, y: y - h };
    const body = c.createLinearGradient(0, y, 0, tip.y);
    body.addColorStop(0, `rgba(150,200,255,${0.75 * a})`);
    body.addColorStop(0.45, `rgba(60,120,255,${0.55 * a})`);
    body.addColorStop(1, 'rgba(40,60,220,0)');
    c.fillStyle = body;
    c.beginPath();
    c.moveTo(x - w, y);
    c.bezierCurveTo(x - w * 1.1, y - h * 0.35, x + sway * 0.4 - w * 0.5, y - h * 0.7, tip.x, tip.y);
    c.bezierCurveTo(x + sway * 0.4 + w * 0.5, y - h * 0.7, x + w * 1.1, y - h * 0.35, x + w, y);
    c.quadraticCurveTo(x, y + w * 0.25, x - w, y);
    c.fill();
    // the hot core
    const ch = h * 0.5, cw = w * 0.5, ct = { x: x + sway * 0.5, y: y - ch };
    const core = c.createLinearGradient(0, y, 0, ct.y);
    core.addColorStop(0, `rgba(235,245,255,${0.85 * a})`);
    core.addColorStop(1, 'rgba(150,200,255,0)');
    c.fillStyle = core;
    c.beginPath();
    c.moveTo(x - cw, y);
    c.bezierCurveTo(x - cw, y - ch * 0.5, ct.x - cw * 0.3, y - ch * 0.8, ct.x, ct.y);
    c.bezierCurveTo(ct.x + cw * 0.3, y - ch * 0.8, x + cw, y - ch * 0.5, x + cw, y);
    c.closePath();
    c.fill();
  }

  /** A pillar of fire standing on the floor at (x, z): a glowing column with a ragged top. */
  private drawColumn(x: number, z: number, halfW: number, glow: number): void {
    if (glow <= 0) return;
    const c = this.ctx, top = FLOOR_Y - TUNE.pillarHeight;
    const a = this.project(x - halfW, FLOOR_Y, z), b = this.project(x + halfW, FLOOR_Y, z);
    const at = this.project(x - halfW, top, z), bt = this.project(x + halfW, top, z);
    const gr = c.createLinearGradient(0, a.y, 0, at.y);
    gr.addColorStop(0, `rgba(255,200,90,${0.7 * glow})`);
    gr.addColorStop(0.5, `rgba(255,120,40,${0.45 * glow})`);
    gr.addColorStop(1, 'rgba(255,60,20,0)');
    c.globalCompositeOperation = 'lighter';
    c.fillStyle = gr;
    c.beginPath();
    c.moveTo(a.x, a.y);
    c.lineTo(b.x, b.y);
    for (let i = 0; i <= 8; i++) {
      const k = 1 - i / 8, sway = Math.sin(k * 9 + this.t * 14) * (b.x - a.x) * 0.08;
      c.lineTo(lerp(at.x, bt.x, k) + sway, lerp(at.y, bt.y, k) + (a.y - at.y) * 0.12 * (1 + Math.sin(k * 13 + this.t * 11)));
    }
    c.closePath();
    c.fill();
    c.globalCompositeOperation = 'source-over';
  }

  /**
   * A stone pillar: a tall column of rock rising out of the ground in its lane, ploughing a furrow
   * down it to you (the screen edge on its side glows softly while you're in its path). A high
   * sweep: a wave of water rolling at you with its crest at standing eye height; its
   * underside sits just above where your eyes must duck to.
   */
  private drawHazard(g: Game, h: Hazard): void {
    const c = this.ctx, u = this.u, near = clamp(1 - h.z / 6, 0, 1);
    if (h.kind === 'stonePillar') {
      const hw = h.halfW ?? TUNE.stonePillarHalfW;
      // the furrow it will plough: its lane on the ground, from the pillar all the way to you
      if (h.z > 0.2) {
        const zs = [h.z, h.z * 0.66, h.z * 0.33, 0.2];
        const edge = (dx: number) => zs.map(z => this.project(h.laneX + dx, FLOOR_Y, z));
        const L = edge(-hw), Rt = edge(hw), fade = 0.35 + 0.45 * h.rise;
        c.fillStyle = `rgba(70,45,22,${0.55 * fade})`;
        c.beginPath();
        L.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)));
        [...Rt].reverse().forEach(p => c.lineTo(p.x, p.y));
        c.closePath(); c.fill();
        c.strokeStyle = `rgba(215,165,105,${0.8 * fade})`;
        c.lineWidth = 3;
        for (const side of [L, Rt]) {
          c.beginPath();
          side.forEach((p, i) => (i ? c.lineTo(p.x + Math.sin(i * 5 + h.id) * 2, p.y) : c.moveTo(p.x, p.y)));
          c.stroke();
        }
      }
      if (h.z < -0.5) return;
      const hgt = TUNE.pillarHeightStone * h.rise, base = this.project(h.x, FLOOR_Y, h.z), topP = this.project(h.x, FLOOR_Y - hgt, h.z);
      const w = hw * base.s * u;
      c.fillStyle = '#7d6246';
      c.strokeStyle = '#3b2a1a';
      c.lineWidth = Math.max(1, w * 0.06);
      c.beginPath();
      c.moveTo(base.x - w, base.y);
      c.lineTo(topP.x - w * 0.85, topP.y + w * 0.1);
      c.lineTo(topP.x - w * 0.3, topP.y - w * 0.12);
      c.lineTo(topP.x + w * 0.4, topP.y);
      c.lineTo(topP.x + w * 0.9, topP.y + w * 0.15);
      c.lineTo(base.x + w, base.y);
      c.closePath(); c.fill(); c.stroke();
      // strata and cracks
      c.strokeStyle = 'rgba(40,28,16,.6)';
      for (let i = 1; i < 5; i++) {
        const y = lerp(base.y, topP.y, i / 5);
        c.beginPath(); c.moveTo(base.x - w * 0.95, y + Math.sin(i * 3.1) * w * 0.05); c.lineTo(base.x + w * 0.95, y - Math.sin(i * 1.7) * w * 0.05); c.stroke();
      }
      c.fillStyle = 'rgba(255,230,190,.12)';
      c.fillRect(base.x - w, topP.y, w * 0.5, base.y - topP.y);
      return;
    }
    if (h.z <= -0.3) return;
    if (h.look === 'boulder') {
      const p = this.project(g.cam.x, h.y, Math.max(0.05, h.z)), r = 18 * u * p.s, N = 7;
      c.fillStyle = '#8a6a48';
      c.strokeStyle = '#3b2a1a';
      c.lineWidth = Math.max(1, r * 0.1);
      c.beginPath();
      for (let i = 0; i <= N; i++) {
        const a = (i / N) * Math.PI * 2 + this.t * 3, rr = r * (0.85 + 0.15 * Math.sin(i * 2.7));
        const px = p.x + Math.cos(a) * rr, py = p.y + Math.sin(a) * rr * 0.9;
        i ? c.lineTo(px, py) : c.moveTo(px, py);
      }
      c.closePath(); c.fill(); c.stroke();
      return;
    }
    // the sweep: a wave of water rolling at you, crest at your eye height, underside just above a duck
    const z = Math.max(0.05, h.z), crest = h.y - 4, under = h.y + TUNE.slabDuck - 2, N = 24;
    const pts = (y: number, wav: number) => Array.from({ length: N + 1 }, (_, i) => {
      const x = g.cam.x - 130 + (260 * i) / N;
      return this.project(x, y + wav * Math.sin(i * 0.9 + this.t * 6 + h.id), z);
    });
    const top = pts(crest, 1.6), bot = pts(under, 0.6);
    const body = c.createLinearGradient(0, top[0].y, 0, bot[0].y);
    body.addColorStop(0, `rgba(170,235,255,${0.75 + 0.2 * near})`);
    body.addColorStop(0.35, 'rgba(70,160,230,.7)');
    body.addColorStop(1, 'rgba(30,70,150,.35)');
    c.fillStyle = body;
    c.beginPath();
    top.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)));
    [...bot].reverse().forEach(p => c.lineTo(p.x, p.y));
    c.closePath(); c.fill();
    // foam along the crest
    c.globalCompositeOperation = 'lighter';
    c.strokeStyle = 'rgba(235,250,255,.85)';
    c.lineWidth = Math.max(2, 1.4 * u * top[0].s);
    c.beginPath();
    top.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)));
    c.stroke();
    c.globalCompositeOperation = 'source-over';
  }

  /**
   * While a stone pillar is coming down a lane you're standing in, the edge of the screen on its
   * side glows a soft red, slowly breathing and deepening as it closes in; it goes once you're out
   * of the way. Gentle on purpose: a cue, not a jump scare.
   */
  private drawPillarWarnings(g: Game): void {
    const c = this.ctx, { W, H } = this;
    for (const side of [-1, 1] as const) {
      let level = 0;
      for (const h of g.hazards) {
        if (h.kind !== 'stonePillar' || h.resolved || (h.laneX > g.cam.x ? 1 : -1) !== side) continue;
        if (Math.abs(g.cam.x - h.laneX) >= TUNE.stonePillarHalfW + TUNE.bodyHalfW) continue; // already out of its way
        level = Math.max(level, h.owner !== null ? 0.35 * h.rise : 0.45 + 0.55 * clamp(1 - h.z / h.startZ, 0, 1));
      }
      if (level <= 0) continue;
      const breathe = 0.85 + 0.15 * Math.sin(this.t * 2.5);
      const a = level * 0.28 * breathe, w = W * 0.3;
      const x0 = side < 0 ? 0 : W, x1 = side < 0 ? w : W - w;
      const gr = c.createLinearGradient(x0, 0, x1, 0);
      gr.addColorStop(0, `rgba(255,40,30,${a})`);
      gr.addColorStop(1, 'rgba(255,40,30,0)');
      c.fillStyle = gr;
      c.fillRect(Math.min(x0, x1), 0, w, H);
    }
  }

  /** A point on a blade's rim: angle 0 = your right, π/2 = straight ahead, π = your left. */
  private bladePoint(b: Blade, angle: number, r = b.r): { x: number; y: number } {
    return this.project(b.x + Math.cos(angle) * r * TUNE.bladeWidthPerDepth, b.y, Math.max(0.05, Math.sin(angle) * r));
  }

  /**
   * The ultimate: a flat, spinning disc of fire at hand height, spreading out like a saw blade.
   * Drawn as a thin sheet with a bright rim and tooth marks racing around the edge.
   */
  private drawBlade(b: Blade): void {
    const c = this.ctx, fade = 1 - (b.r / TUNE.bladeMaxR) ** 3, N = 64;
    const rim = Array.from({ length: N + 1 }, (_, i) => this.bladePoint(b, (Math.PI * i) / N));
    const path = (pts: { x: number; y: number }[]) => {
      c.beginPath();
      pts.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)));
    };
    c.globalCompositeOperation = 'lighter';
    // the sheet: brighter toward the rim
    for (const [k, a] of [[1, 0.1], [0.8, 0.08], [0.55, 0.06]] as const) {
      path(Array.from({ length: N + 1 }, (_, i) => this.bladePoint(b, (Math.PI * i) / N, b.r * k)));
      c.closePath();
      c.fillStyle = `rgba(255,130,50,${a * fade})`;
      c.fill();
    }
    // the rim: a wide glow and a hot core
    path(rim);
    c.lineJoin = 'round';
    c.strokeStyle = `rgba(255,110,40,${0.35 * fade})`;
    c.lineWidth = 3.2 * this.u;
    c.stroke();
    c.strokeStyle = `rgba(255,225,160,${0.9 * fade})`;
    c.lineWidth = 0.9 * this.u;
    c.stroke();
    // saw teeth sweeping around the rim
    const teeth = 36, spin = (this.t * 5) % ((2 * Math.PI) / teeth);
    c.strokeStyle = `rgba(255,240,200,${0.8 * fade})`;
    c.lineWidth = 0.5 * this.u;
    for (let i = 0; i < teeth; i++) {
      const a = (Math.PI * i) / teeth + spin;
      if (a > Math.PI) continue;
      const p = this.bladePoint(b, a), q = this.bladePoint(b, a - 0.06, b.r * 0.9);
      c.beginPath(); c.moveTo(p.x, p.y); c.lineTo(q.x, q.y); c.stroke();
    }
    c.globalCompositeOperation = 'source-over';
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
    // hands only pour out flame just after they attack (the shield has its own flames below)
    for (const side of ['l', 'r'] as const) {
      const h = g.hands[side], flare = this.flare[side] / FLARE_S;
      if (!h?.inView || flare <= 0 || g.shield.on) continue;
      const w = g.handWorld(this.shownAt(h)), vx0 = h.vel.x * 0.3, vy0 = h.vel.y * 0.3;
      for (let i = nOf(140 * Math.min(1, flare), dt); i > 0; i--) {
        const a = Math.random() * 6.283, d = Math.sqrt(Math.random()) * 1.8;
        this.emit(w.x + Math.cos(a) * d, w.y - 2 + Math.sin(a) * d, 0, rnd(-5, 5) + vx0, rnd(-18, -6) + vy0, 0, rnd(0.3, 0.55), rnd(2.2, 3.6));
      }
    }
    // finisher gather: fire swirls into the space between your hands, then both hands blaze
    const { l: gl, r: gr } = g.hands;
    if (g.gather > 0 && gl?.inView && gr?.inView) {
      const ready = g.ultimateIn <= 0, m = g.handWorld({ x: (gl.pos.x + gr.pos.x) / 2, y: (gl.pos.y + gr.pos.y) / 2 });
      const rate = (ready ? 90 : 25) + (g.gather >= 1 && ready ? 220 : 0);
      for (let i = nOf(rate * g.gather, dt); i > 0; i--) {
        const a = Math.random() * 6.283, d = rnd(3, 7);
        // sucked in toward the middle
        this.emit(m.x + Math.cos(a) * d, m.y + Math.sin(a) * d, 0, -Math.cos(a) * d * 3, -Math.sin(a) * d * 3 - 6, 0, rnd(0.2, 0.4), rnd(2, 3.4), ready ? 'fire' : 'earth', 0.3);
      }
      if (g.gather >= 1 && ready) this.flare = { l: Math.max(this.flare.l, FLARE_S * 0.6), r: Math.max(this.flare.r, FLARE_S * 0.6) };
    }
    // the blue inferno: flames licking up all over the ground while it burns
    if (g.groundFire > 0) {
      const k = Math.min(1, g.groundFire / 0.8) * Math.min(1, (TUNE.infernoS - g.groundFire) / 0.3 + 0.3);
      // a few sparks rising off the flames
      for (let i = nOf(120 * k, dt); i > 0; i--) {
        const z = Math.random() ** 1.3 * 14, x = g.cam.x + rnd(-1, 1) * (60 + z * 28);
        this.emit(x, FLOOR_Y - rnd(8, 16), z, rnd(-3, 3), rnd(-40, -20), 0, rnd(0.3, 0.6), rnd(1, 2), 'blue', 1);
      }
    }
    // sparks flung off the ultimate's spinning rim
    for (const b of g.blades) {
      const fade = 1 - b.r / TUNE.bladeMaxR, w = TUNE.bladeWidthPerDepth;
      for (let i = nOf(600 * fade, dt); i > 0; i--) {
        const a = Math.random() * Math.PI, z = Math.max(0.05, Math.sin(a) * b.r);
        const s = FOCAL / (FOCAL + z);
        // tangent to the rim (spin) plus outward, in world units scaled to stay visible far away
        this.emit(b.x + Math.cos(a) * b.r * w, b.y + rnd(-1, 1), z,
          (-Math.sin(a) * 90 + Math.cos(a) * 40) / s, rnd(-8, 2), Math.cos(a) * 4 + Math.sin(a) * TUNE.bladeSpeed * 0.5,
          rnd(0.18, 0.4), rnd(2, 3.5) / Math.sqrt(s), 'fire', 0.15);
      }
    }
    // flames licking off the X block
    if (g.xBlock && g.hands.l && g.hands.r) {
      const m = { x: (g.hands.l.pos.x + g.hands.r.pos.x) / 2, y: (g.hands.l.pos.y + g.hands.r.pos.y) / 2 - 4 };
      const w = g.handWorld(m);
      for (let i = nOf(260, dt); i > 0; i--) {
        const k = rnd(-24, 24), diag = Math.random() < 0.5 ? 1 : -1;
        this.emit(w.x + k, w.y + k * diag, 0.2, rnd(-4, 4), rnd(-22, -8), 0, rnd(0.2, 0.4), rnd(2, 3.5));
      }
    }
    // rolling pillars of fire
    for (const col of g.pillars) {
      for (let i = nOf(420, dt); i > 0; i--) {
        this.emit(col.x + rnd(-col.halfW, col.halfW), FLOOR_Y - rnd(0, TUNE.pillarHeight * 0.8), col.z + rnd(-0.3, 0.3),
          rnd(-6, 6), rnd(-90, -40), rnd(-2, 0), rnd(0.25, 0.55), rnd(5, 9));
      }
    }
    // fire walls: standing ones burn in place, rolling ones carry their flames forward
    for (const wall of g.walls) {
      const fade = Math.min(1, wall.life / 0.6);
      for (let i = nOf(520 * fade, dt); i > 0; i--) {
        this.emit(wall.x + rnd(-wall.halfW, wall.halfW), FLOOR_Y - rnd(0, 6), wall.z, rnd(-4, 4), rnd(-110, -55), wall.vz, rnd(0.55, 1.05), rnd(5, 9));
      }
    }
    // dust off stone pillars as they rise and grind forward
    for (const h of g.hazards) {
      if (h.kind !== 'stonePillar' || h.z < 0 || (h.owner !== null && h.rise >= 1)) continue;
      for (let i = nOf(140, dt); i > 0; i--) {
        this.emit(h.x + rnd(-1, 1) * TUNE.stonePillarHalfW, FLOOR_Y - rnd(0, 4), h.z, rnd(-10, 10), rnd(-25, -8), h.vz, rnd(0.3, 0.7), rnd(4, 8), 'earth', 0.3);
      }
    }
    const { l, r } = g.hands;
    if (g.shield.on && l && r) {
      const a = g.handWorld(l.pos), b = g.handWorld(r.pos), e = 1;
      for (let i = nOf(260, dt); i > 0; i--) {
        const k = Math.random();
        this.emit(lerp(a.x, b.x, k) + rnd(-1, 1), lerp(a.y, b.y, k) + rnd(-2, 3), 0,
          rnd(-3, 3), rnd(-34, -14) * (0.6 + e * 0.6), 0, rnd(0.25, 0.5), (3 + e * 2) * rnd(0.7, 1.1));
      }
    }
    // a charged fist smoulders with blue flame
    for (const side of ['l', 'r'] as const) {
      const h = g.hands[side];
      if (!h?.inView || h.charge <= 0) continue;
      const w = g.handWorld(this.shownAt(h));
      for (let i = nOf(200 * h.charge * h.charge, dt); i > 0; i--) {
        const a = Math.random() * 6.283, d = Math.sqrt(Math.random()) * 2;
        this.emit(w.x + Math.cos(a) * d, w.y - 2 + Math.sin(a) * d, 0, rnd(-4, 4), rnd(-16, -5), 0, rnd(0.25, 0.5), rnd(2, 3.4), 'blue');
      }
    }
    for (const p of g.projs) {
      const pal: Pal = p.kind === 'enemy' ? 'spirit' : p.shot === 'charged' ? 'blue' : 'fire';
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
