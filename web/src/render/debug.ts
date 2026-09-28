import type { TrackingFrame } from '../input/types';
import { palmOf } from '../input/landmarks';
import { fistThresholds, TUNING, type Intent } from '../intent/interpret';
import type { Vec2 } from '../math';

/** Corner panel: what the camera sees plus the tracked points; backtick adds live numbers. */
export class DebugView {
  private ctx: CanvasRenderingContext2D;
  private detailed = false;

  constructor(private canvas: HTMLCanvasElement, private text: HTMLElement) {
    this.ctx = canvas.getContext('2d')!;
    this.resize();
  }

  resize(): void {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.canvas.clientWidth * dpr);
    this.canvas.height = Math.round(this.canvas.clientHeight * dpr);
  }

  toggle(): void {
    this.detailed = !this.detailed;
    this.text.classList.toggle('hidden', !this.detailed);
  }

  draw(f: TrackingFrame | null, intent: Intent | null, video: HTMLVideoElement | null): void {
    const c = this.ctx, w = this.canvas.width, h = this.canvas.height;
    c.fillStyle = '#05070a';
    c.fillRect(0, 0, w, h);
    if (video && video.readyState >= 2) {
      c.save();
      c.globalAlpha = 0.6;
      c.translate(w, 0);
      c.scale(-1, 1); // mirror, to match the tracking frame
      c.drawImage(video, 0, 0, w, h);
      c.restore();
    }
    if (f) {
      const P = (p: Vec2) => ({ x: p.x * w, y: p.y * h });
      c.lineWidth = 2;
      if (f.shoulderL && f.shoulderR) {
        const a = P(f.shoulderL), b = P(f.shoulderR);
        c.strokeStyle = '#9dffcf';
        c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
      }
      if (f.head) {
        const p = P(f.head);
        c.fillStyle = '#fff';
        c.beginPath(); c.arc(p.x, p.y, 5, 0, 7); c.fill();
      }
      // arms: shoulder → elbow → wrist, faded where the model is unsure; a wrist outside the picture
      // is pinned to the panel edge with a triangle pointing where it is
      for (const arm of [f.arms.l, f.arms.r]) {
        if (!arm) continue;
        const pts = [arm.shoulder, arm.elbow, arm.wrist];
        for (let i = 0; i < 2; i++) {
          const a = P(pts[i]), b = P(pts[i + 1]);
          c.strokeStyle = `rgba(157,255,207,${Math.max(0.2, Math.min(pts[i].vis, pts[i + 1].vis))})`;
          c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
        }
        const wp = P(arm.wrist), m = 6;
        const q = { x: Math.max(m, Math.min(w - m, wp.x)), y: Math.max(m, Math.min(h - m, wp.y)) };
        c.fillStyle = arm.wrist.vis >= 0.5 ? '#9dffcf' : '#ff9d6b';
        if (q.x !== wp.x || q.y !== wp.y) {
          const dir = Math.atan2(wp.y - q.y, wp.x - q.x);
          c.beginPath();
          c.moveTo(q.x + Math.cos(dir) * m, q.y + Math.sin(dir) * m);
          c.lineTo(q.x + Math.cos(dir + 2.4) * m, q.y + Math.sin(dir + 2.4) * m);
          c.lineTo(q.x + Math.cos(dir - 2.4) * m, q.y + Math.sin(dir - 2.4) * m);
          c.closePath(); c.fill();
        } else {
          c.beginPath(); c.arc(q.x, q.y, 3, 0, 7); c.fill();
        }
      }
      // each hand: filled orange = open, red ring = fist; label shows openness and palm facing
      c.font = `${Math.round(h / 14)}px ui-monospace, Menlo, monospace`;
      const sorted = [...f.hands].sort((a, b) => a.center.x - b.center.x);
      sorted.forEach((hand, i) => {
        const p = P(hand.center), open = hand.open >= 0.5, r = Math.max(4, hand.size * w * 0.5);
        c.strokeStyle = c.fillStyle = open ? '#ffb347' : '#ff5a4a';
        c.beginPath(); c.arc(p.x, p.y, r, 0, 7);
        if (open) c.fill(); else c.stroke();
        // left hand's label hangs off to the left, right hand's to the right, so they never overlap
        const leftmost = sorted.length > 1 && i === 0;
        c.textAlign = leftmost ? 'right' : 'left';
        const x = leftmost ? p.x - r - 3 : p.x + r + 3;
        c.fillStyle = '#fff';
        c.fillText(`${open ? 'OPEN' : 'FIST'} ${hand.open.toFixed(2)}`, x, p.y);
        const n = hand.normal && hand.side ? palmOf(hand.normal, hand.side) : null;
        c.fillText(n ? `palm ${n.x.toFixed(1)} ${n.y.toFixed(1)} ${n.z.toFixed(1)}` : `face ${hand.facing.toFixed(2)}`, x, p.y + h / 12);
      });
    }
    if (intent) this.drawArmMeters(intent);
    if (this.detailed && intent) {
      const line = (name: string, h: Intent['hands']['l']) => h
        ? `${name} ${h.open ? 'open' : 'fist'} ${h.openness.toFixed(2)}  palm ${h.palm ? `${h.palm.x.toFixed(1)} ${h.palm.y.toFixed(1)} ${h.palm.z.toFixed(1)} (x right, y down, z forward)` : '—'}  speed ${Math.hypot(h.vel.x, h.vel.y).toFixed(0)}`
          + `\n  ${h.source}${h.inView ? '' : ' (out of view)'}  arm ${h.extension === null ? '—' : h.extension.toFixed(2)}`
        : `${name} —`;
      const deg = (r: number) => `${Math.round((r * 180) / Math.PI)}°`;
      this.text.textContent = [
        `present ${intent.present}  shield ${intent.shield}`,
        `head x ${intent.head.x.toFixed(1)}  y ${intent.head.y.toFixed(1)}`
          + (intent.face ? `  turn ${intent.face.yaw.toFixed(2)}  tilt ${deg(intent.face.roll)}` : ''),
        `shoulders tilt ${deg(intent.bodyTilt)}`,
        line('L', intent.hands.l),
        line('R', intent.hands.r),
        f?.hands.length ? `size ${f.hands.map(x => x.size.toFixed(3)).join('  ')}` : '',
      ].join('\n');
    }
  }

  /**
   * Per-fist bars along the bottom: how far the fist jolted forward just now (0–0.3 m), with the
   * orange tick where a fist punch fires. The dot is lit while that fist is ready to fire again.
   * Without 3D data: arm straightness instead.
   */
  private drawArmMeters(intent: Intent): void {
    const c = this.ctx, w = this.canvas.width, h = this.canvas.height;
    const barH = Math.max(4, h * 0.045), y = h - barH - h * 0.03, font = Math.round(h / 16);
    c.font = `${font}px ui-monospace, Menlo, monospace`;
    c.textBaseline = 'bottom';
    c.textAlign = 'left';
    c.fillStyle = '#cfe';
    c.fillText(`punch: ${TUNING.punchTrigger === 'extend' ? `fist ×${TUNING.punchSensitivity.toFixed(1)} ([ ])` : 'open hand'}  (P)`, w * 0.04, y - font * 0.4);
    const noiseY = y - font * 0.4;
    const noises: string[] = [];
    (['l', 'r'] as const).forEach((side, i) => {
      const hand = intent.hands[side], x0 = w * (0.04 + i * 0.5), bw = w * 0.42;
      c.fillStyle = 'rgba(255,255,255,.1)';
      c.fillRect(x0, y, bw, barH);
      // the fist's forward jolt over the last moment, against what a punch needs
      const byReach = hand && hand.punchRise !== null;
      const RANGE = 0.3, noise = hand?.reachNoise ?? 0;
      const value = byReach ? Math.max(0, hand!.punchRise!) / RANGE : hand?.extension ?? null;
      const marks = byReach
        ? [[fistThresholds(noise).rise / RANGE, '#ff7a3d']] as const
        : [[TUNING.extendRearmBelow, '#9dffcf'], [TUNING.extendFireAbove, '#ff7a3d']] as const;
      if (value !== null) {
        c.fillStyle = hand!.punchReady ? '#ffb347' : '#8a6a4a';
        c.fillRect(x0, y, bw * Math.max(0, Math.min(1, value)), barH);
      }
      for (const [at, col] of marks) {
        c.fillStyle = col;
        c.fillRect(x0 + bw * Math.min(1, at) - 1, y - 2, 2, barH + 4);
      }
      if (byReach) noises.push(`${side.toUpperCase()} ±${Math.round(noise * 100)}cm`);
      c.fillStyle = hand?.punchReady ? '#ffb347' : '#444';
      c.beginPath(); c.arc(x0 - barH * 0.1 + bw + barH * 0.9, y + barH / 2, barH * 0.45, 0, 7); c.fill();
    });
    if (noises.length) {
      c.textAlign = 'right';
      c.fillStyle = '#cfe';
      c.fillText(noises.join('  '), w * 0.96, noiseY);
    }
  }
}
