/**
 * Small looping animations of each lesson's move, drawn in the lesson card: stylised hands (or a
 * figure, for the dodges) doing the gesture, with arrows. Seen from your side, like your own hands.
 */

/** Hands are drawn this much bigger than their 16-unit base, to read at a glance. */
const HAND = 1.6;
const FIRE = '#ffb45e', SKIN = '#c98f68', SKIN_DARK = '#8a5a3c', LINE = 'rgba(255,226,184,.85)', STONE = '#8b6d4c', WATER = '#7fd6ff';

/** 0 → 1 → 0 over a loop, eased, with a pause at each end. */
const swing = (k: number) => { const x = Math.min(1, Math.max(0, (Math.sin(k * Math.PI * 2 - Math.PI / 2) + 1) / 2 * 1.3 - 0.15)); return x * x * (3 - 2 * x); };
/** 0 → 1 quickly, hold, snap back: a strike. */
const strike = (k: number) => (k < 0.25 ? k / 0.25 : k < 0.55 ? 1 : k < 0.75 ? 1 - (k - 0.55) / 0.2 : 0);
/** A strike lasting `dur` seconds from `start`, within a loop at time `u`. */
const hit = (u: number, start: number, dur: number) => (u >= start && u < start + dur ? strike((u - start) / dur) : 0);
/** 0 → 1 over `dur` seconds from `start` (held at 1 after), eased. */
const ramp = (u: number, start: number, dur: number) => { const x = Math.min(1, Math.max(0, (u - start) / dur)); return x * x * (3 - 2 * x); };

export class LessonDemo {
  private ctx: CanvasRenderingContext2D;
  private w = 0;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
  }

  /** Draw lesson `id` at time t (s). */
  draw(id: string, t: number): void {
    const dpr = Math.min(devicePixelRatio || 1, 2), cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    if (this.canvas.width !== Math.round(cw * dpr) || this.canvas.height !== Math.round(ch * dpr)) {
      this.canvas.width = Math.round(cw * dpr);
      this.canvas.height = Math.round(ch * dpr);
    }
    // drawn on a 150 × 112 stage, scaled to fit the canvas
    const c = this.ctx, z = Math.min(cw / 150, ch / 112);
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, cw, ch);
    c.setTransform(dpr * z, 0, 0, dpr * z, (dpr * (cw - 150 * z)) / 2, (dpr * (ch - 112 * z)) / 2);
    const W = 150, H = 112, cx = W / 2;
    this.w = W;
    const k = (t % 2) / 2; // a 2 s loop
    switch (id) {
      case 'move': {
        // lean left, lean right, duck — one after another
        const ph = Math.floor((t % 6) / 2), s = swing(k);
        const dx = ph === 0 ? -s * 26 : ph === 1 ? s * 26 : 0, dy = ph === 2 ? s * 18 : 0;
        this.person(cx + dx, H * 0.34 + dy, dx * 0.012);
        if (ph === 0) this.arrow(cx - 16, H * 0.3, cx - 44, H * 0.3);
        if (ph === 1) this.arrow(cx + 16, H * 0.3, cx + 44, H * 0.3);
        if (ph === 2) this.arrow(cx + 34, H * 0.2, cx + 34, H * 0.55);
        break;
      }
      case 'punch': {
        // right fist snaps out (grows toward you), fire leaves it
        const s = strike(k);
        this.fist(cx - 34, H * 0.62, 1);
        this.fist(cx + 34 - s * 20, H * 0.62 - s * 22, 1 + s * 0.45);
        if (s > 0.9) this.flame(cx + 8, H * 0.3, 8);
        this.arrow(cx + 34, H * 0.75, cx + 12, H * 0.42, 0.5);
        break;
      }
      case 'pillar': {
        // a pillar slides down the right; the figure leans left out of its way
        const s = swing(k);
        this.stone(cx + 30, H * 0.9, 22, 46 * (0.7 + 0.3 * s));
        this.person(cx - s * 26, H * 0.36, -s * 0.3);
        this.arrow(cx - 8, H * 0.18, cx - 40, H * 0.18);
        break;
      }
      case 'wave': {
        // a wave at head height; the figure ducks under it
        const s = swing(k);
        this.person(cx, H * 0.3 + s * 22, 0);
        this.waveBand(H * 0.24);
        this.arrow(cx + 40, H * 0.35, cx + 40, H * 0.72);
        break;
      }
      case 'shield': {
        // both hands open and still, fire between them
        const f = 0.8 + 0.2 * Math.sin(t * 9);
        this.palm(cx - 36, H * 0.6, 1);
        this.palm(cx + 36, H * 0.6, 1, true);
        this.sheet(cx - 24, cx + 24, H * 0.62, 30 * f);
        break;
      }
      case 'xblock': {
        // forearms cross into an X
        const s = swing(k);
        this.forearm(cx - 34 + s * 30, H * 0.95, cx + 22 - s * 2 - (1 - s) * 40, H * 0.3 + (1 - s) * 16);
        this.forearm(cx + 34 - s * 30, H * 0.95, cx - 22 + s * 2 + (1 - s) * 40, H * 0.3 + (1 - s) * 16);
        if (s > 0.9) this.flame(cx, H * 0.55, 10);
        break;
      }
      case 'palm': {
        // one fist stays, the other hand is an open palm shoved forward: a pillar of fire rolls out
        const s = strike(k);
        this.fist(cx - 36, H * 0.62, 1);
        this.palm(cx + 34 - s * 14, H * 0.62 - s * 12, 1 + s * 0.4, true);
        if (s > 0.9) this.column(cx + 10, H * 0.42, 8, 26);
        this.arrow(cx + 40, H * 0.9, cx + 20, H * 0.5, 0.5);
        break;
      }
      case 'wall': {
        // both palms sweep up from low; the wall of fire rises with them
        const s = strike(k);
        const y = H * 0.85 - s * H * 0.5;
        this.palm(cx - 32, y, 0.9);
        this.palm(cx + 32, y, 0.9, true);
        if (s > 0.3) this.sheet(cx - 36, cx + 36, H * 0.95, (H * 0.95 - y) * 0.9);
        this.arrow(cx + 50, H * 0.85, cx + 50, H * 0.3);
        break;
      }
      case 'charge': {
        // fist to the hip (or cocked by the ear, every other loop) and held: it glows, then burns
        // blue; then punch a blue fireball
        const ear = Math.floor(t / 3) % 2 === 1, u = t % 3;
        const into = ramp(u, 0.1, 0.3) * (1 - ramp(u, 1.5, 0.08)), glow = ramp(u, 0.4, 0.9), s = hit(u, 1.5, 0.6);
        this.fist(cx - 34, H * 0.62, 1);
        const fx = cx + 34 + into * (ear ? 4 : 6) - s * 20, fy = H * 0.62 + into * (ear ? -44 : 32) - s * 22;
        if (glow > 0 && s === 0 && u < 1.5) this.glow(fx, fy, 10 + glow * 16, `rgba(120,170,255,${0.3 + 0.5 * glow})`);
        this.fist(fx, fy, 1 + s * 0.45);
        if (s > 0.9) this.glow(cx + 8, H * 0.25, 14, 'rgba(150,200,255,1)');
        if (u < 1.4) this.arrow(cx + 54, H * 0.55, cx + 54, ear ? H * 0.12 : H * 0.95, 0.6);
        break;
      }
      case 'flurry': {
        // three quick jabs, left-right-left; the third throws a big fireball
        const u = t % 2.2;
        const a = hit(u, 0, 0.35), b = hit(u, 0.35, 0.35), c3 = hit(u, 0.7, 0.45);
        this.fist(cx - 34 + (a + c3) * 20, H * 0.62 - (a + c3) * 22, 1 + (a + c3) * 0.45);
        this.fist(cx + 34 - b * 20, H * 0.62 - b * 22, 1 + b * 0.45);
        if (a > 0.9 || b > 0.9) this.flame(cx, H * 0.3, 7);
        if (c3 > 0.9) this.flame(cx - 4, H * 0.26, 16);
        break;
      }
      case 'counter': {
        // shield up, an orb bursts on it; then a quick punch fires a white-hot counter
        const u = t % 2.6, shield = u < 1.2, s = hit(u, 1.3, 0.6);
        if (shield) {
          this.palm(cx - 36, H * 0.62, 1);
          this.palm(cx + 36, H * 0.62, 1, true);
          this.sheet(cx - 24, cx + 24, H * 0.64, 26);
          const orb = Math.min(1, u / 0.9);
          if (orb < 1) this.glow(cx, 10 + orb * H * 0.35, 5 + orb * 6, 'rgba(120,220,255,.9)');
          else this.glow(cx, H * 0.45, 16, 'rgba(255,220,160,.8)');
        } else {
          this.fist(cx - 34, H * 0.62, 1);
          this.fist(cx + 34 - s * 20, H * 0.62 - s * 22, 1 + s * 0.45);
          if (s > 0.9) { this.flame(cx + 6, H * 0.28, 9); this.ring(cx + 6, H * 0.28, 13); }
        }
        break;
      }
      case 'onetwo': {
        // jab, jab, then an open palm shoved — a wide pillar
        const u = t % 2.6, a = hit(u, 0, 0.35), b = hit(u, 0.35, 0.35), p = hit(u, 0.8, 0.8);
        this.fist(cx - 34 + a * 20, H * 0.62 - a * 22, 1 + a * 0.45);
        if (u < 0.75) this.fist(cx + 34 - b * 20, H * 0.62 - b * 22, 1 + b * 0.45);
        else this.palm(cx + 34 - p * 14, H * 0.62 - p * 12, 1 + p * 0.4, true);
        if (p > 0.9) this.column(cx + 4, H * 0.42, 20, 30);
        break;
      }
      case 'volley': {
        // a palm push with each hand, one right after the other: one wide wave
        const u = t % 2.2, r = hit(u, 0, 0.7), l = hit(u, 0.3, 0.7);
        this.palm(cx - 34 + l * 14, H * 0.62 - l * 12, 1 + l * 0.4);
        this.palm(cx + 34 - r * 14, H * 0.62 - r * 12, 1 + r * 0.4, true);
        if (r > 0.9 && l < 0.5) this.column(cx + 12, H * 0.42, 8, 26);
        if (l > 0.9) this.column(cx, H * 0.42, 34, 30);
        break;
      }
      case 'wallbreaker': {
        // palms sweep up (a wall rises), then shove forward: the wall rolls away
        const u = t % 3.2, up = ramp(u, 0.1, 0.5), push = ramp(u, 1.4, 0.3), away = ramp(u, 1.6, 1.2);
        const y = H * 0.85 - up * H * 0.45 - push * 8;
        this.palm(cx - 32 + push * 6, y, 0.9 + push * 0.3);
        this.palm(cx + 32 - push * 6, y, 0.9 + push * 0.3, true);
        if (up > 0.3) this.sheet(cx - 40 * (1 - away * 0.6), cx + 40 * (1 - away * 0.6), H * (0.95 - away * 0.35), (H * 0.5) * (0.4 + 0.6 * up) * (1 - away * 0.6));
        if (u > 1.3 && u < 2.2) this.arrow(cx, H * 0.98, cx, H * 0.72, 0.6);
        break;
      }
      case 'ultimate': {
        // jab, jab, then gather both open hands and fling them apart: the blade of fire
        const u = t % 3.4, a = hit(u, 0, 0.35), b = hit(u, 0.35, 0.35);
        if (u < 0.8) {
          this.fist(cx - 34 + a * 20, H * 0.62 - a * 22, 1 + a * 0.45);
          this.fist(cx + 34 - b * 20, H * 0.62 - b * 22, 1 + b * 0.45);
        } else {
          const d = 14 + 22 * (1 - ramp(u, 0.9, 0.5)) + 44 * ramp(u, 1.9, 0.25);
          this.palm(cx - d, H * 0.58, 0.9);
          this.palm(cx + d, H * 0.58, 0.9, true);
          if (u > 2.0) this.disc(cx, H * 0.72, 30 + 40 * ramp(u, 2.0, 0.6));
          if (u > 1.9) { this.arrow(cx - 14, H * 0.3, cx - 50, H * 0.3, 0.8); this.arrow(cx + 14, H * 0.3, cx + 50, H * 0.3, 0.8); }
        }
        break;
      }
    }
  }

  // ---------- pieces ----------

  private person(x: number, y: number, tilt: number): void {
    const c = this.ctx;
    c.save();
    c.translate(x, y);
    c.rotate(tilt);
    c.fillStyle = LINE;
    c.beginPath(); c.arc(0, 0, 9, 0, 7); c.fill();
    c.beginPath(); c.moveTo(-22, 26); c.quadraticCurveTo(0, 8, 22, 26); c.lineTo(22, 40); c.lineTo(-22, 40); c.closePath(); c.fill();
    c.restore();
  }

  private fist(x: number, y: number, s: number): void {
    s *= HAND;
    const c = this.ctx, w = 16 * s, h = 14 * s;
    c.fillStyle = SKIN;
    c.strokeStyle = SKIN_DARK;
    c.lineWidth = 1.2;
    c.beginPath(); c.roundRect(x - w / 2, y - h / 2, w, h, 4 * s); c.fill(); c.stroke();
    for (let i = 1; i < 4; i++) { c.beginPath(); c.moveTo(x - w / 2 + (w * i) / 4, y - h / 2 + 1); c.lineTo(x - w / 2 + (w * i) / 4, y - h / 2 + h * 0.4); c.stroke(); }
  }

  /** An open palm, fingers up; `mirror` puts the thumb on the other side. */
  private palm(x: number, y: number, s: number, mirror = false): void {
    const c = this.ctx;
    c.save();
    c.translate(x, y);
    s *= HAND;
    c.scale(mirror ? -s : s, s);
    c.fillStyle = SKIN;
    c.strokeStyle = SKIN_DARK;
    c.lineWidth = 1;
    c.beginPath(); c.roundRect(-8, -6, 16, 14, 4); c.fill(); c.stroke();
    [-6, -2, 2, 6].forEach((fx, i) => { c.beginPath(); c.roundRect(fx - 1.7, -18 + Math.abs(i - 1.5) * 2, 3.4, 13, 1.7); c.fill(); c.stroke(); });
    c.beginPath(); c.roundRect(7, -3, 8, 3.4, 1.7); c.fill(); c.stroke();
    c.restore();
  }

  private forearm(x0: number, y0: number, x1: number, y1: number): void {
    const c = this.ctx;
    c.strokeStyle = SKIN;
    c.lineWidth = 12;
    c.lineCap = 'round';
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
    this.fist(x1, y1, 0.8);
  }

  /** A soft coloured glow. */
  private glow(x: number, y: number, r: number, color: string): void {
    const c = this.ctx, g = c.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, color); g.addColorStop(1, 'rgba(0,0,0,0)');
    c.fillStyle = g;
    c.beginPath(); c.arc(x, y, r, 0, 7); c.fill();
  }

  /** A bright ring (a counter shot). */
  private ring(x: number, y: number, r: number): void {
    const c = this.ctx;
    c.strokeStyle = 'rgba(255,240,210,.85)';
    c.lineWidth = 2.5;
    c.beginPath(); c.arc(x, y, r, 0, 7); c.stroke();
  }

  private flame(x: number, y: number, r: number): void {
    const c = this.ctx, g = c.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, 'rgba(255,245,210,1)'); g.addColorStop(0.4, 'rgba(255,170,70,.9)'); g.addColorStop(1, 'rgba(255,90,20,0)');
    c.fillStyle = g;
    c.beginPath(); c.arc(x, y, r, 0, 7); c.fill();
  }

  /** A sheet of fire standing from y (bottom) up h. */
  private sheet(x0: number, x1: number, y: number, h: number): void {
    const c = this.ctx, g = c.createLinearGradient(0, y, 0, y - h);
    g.addColorStop(0, 'rgba(255,190,90,.9)'); g.addColorStop(1, 'rgba(255,90,20,0)');
    c.fillStyle = g;
    c.beginPath();
    c.moveTo(x0, y);
    for (let i = 0; i <= 8; i++) c.lineTo(x0 + ((x1 - x0) * i) / 8, y - h * (0.7 + 0.3 * Math.sin(i * 2.1 + performance.now() / 90)));
    c.lineTo(x1, y);
    c.closePath(); c.fill();
  }

  private column(x: number, y: number, hw: number, h: number): void {
    this.sheet(x - hw, x + hw, y + h / 2, h);
  }

  private disc(x: number, y: number, r: number): void {
    const c = this.ctx;
    c.strokeStyle = FIRE;
    c.lineWidth = 3;
    c.beginPath(); c.ellipse(x, y, r, r * 0.18, 0, 0, 7); c.stroke();
  }

  private stone(x: number, bottom: number, w: number, h: number): void {
    const c = this.ctx;
    c.fillStyle = STONE;
    c.strokeStyle = '#3b2a1a';
    c.lineWidth = 1.2;
    c.beginPath(); c.moveTo(x - w / 2, bottom); c.lineTo(x - w / 2 + 2, bottom - h); c.lineTo(x + w / 2 - 3, bottom - h - 3); c.lineTo(x + w / 2, bottom); c.closePath(); c.fill(); c.stroke();
  }

  private waveBand(y: number): void {
    const c = this.ctx;
    c.fillStyle = 'rgba(80,170,230,.75)';
    c.beginPath();
    c.moveTo(0, y + 8);
    for (let i = 0; i <= 12; i++) c.lineTo((this.w * i) / 12, y + Math.sin(i + performance.now() / 150) * 2);
    c.lineTo(this.w, y + 8);
    c.closePath(); c.fill();
    c.strokeStyle = WATER;
    c.lineWidth = 2;
    c.beginPath();
    for (let i = 0; i <= 12; i++) c[i ? 'lineTo' : 'moveTo']((this.w * i) / 12, y + Math.sin(i + performance.now() / 150) * 2);
    c.stroke();
  }

  private arrow(x0: number, y0: number, x1: number, y1: number, alpha = 1): void {
    const c = this.ctx, a = Math.atan2(y1 - y0, x1 - x0);
    c.strokeStyle = `rgba(255,195,107,${alpha})`;
    c.fillStyle = `rgba(255,195,107,${alpha})`;
    c.lineWidth = 2.5;
    c.lineCap = 'round';
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1 - Math.cos(a) * 5, y1 - Math.sin(a) * 5); c.stroke();
    c.beginPath();
    c.moveTo(x1, y1);
    c.lineTo(x1 - Math.cos(a - 0.5) * 9, y1 - Math.sin(a - 0.5) * 9);
    c.lineTo(x1 - Math.cos(a + 0.5) * 9, y1 - Math.sin(a + 0.5) * 9);
    c.closePath(); c.fill();
  }
}
