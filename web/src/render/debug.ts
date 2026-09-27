import type { TrackingFrame } from '../input/types';
import type { Intent } from '../intent/interpret';
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
        c.fillText(`face ${hand.facing.toFixed(2)}`, x, p.y + h / 12);
      });
    }
    if (this.detailed && intent) {
      const line = (name: string, h: Intent['hands']['l']) => h
        ? `${name} ${h.open ? 'open' : 'fist'} ${h.openness.toFixed(2)}  face ${h.facing.toFixed(2)}  speed ${Math.hypot(h.vel.x, h.vel.y).toFixed(0)}`
        : `${name} —`;
      this.text.textContent = [
        `present ${intent.present}  shield ${intent.shield}`,
        `head x ${intent.head.x.toFixed(1)}  y ${intent.head.y.toFixed(1)}`,
        line('L', intent.hands.l),
        line('R', intent.hands.r),
        f?.hands.length ? `size ${f.hands.map(x => x.size.toFixed(3)).join('  ')}` : '',
      ].join('\n');
    }
  }
}
