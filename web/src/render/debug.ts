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
      c.strokeStyle = '#ffc36b';
      for (const hand of f.hands) {
        const p = P(hand.center);
        c.beginPath(); c.arc(p.x, p.y, Math.max(3, hand.size * w * 0.5), 0, 7); c.stroke();
      }
    }
    if (this.detailed && intent) {
      const hd = intent.hands;
      this.text.textContent = [
        `present ${intent.present}  raised ${intent.raised}`,
        `head    x ${intent.head.x.toFixed(1)}  y ${intent.head.y.toFixed(1)}`,
        hd ? `hands   ${hd.center.x.toFixed(1)}, ${hd.center.y.toFixed(1)}  spread ${hd.spread.toFixed(1)}` : 'hands   —',
        hd ? `vel     ${hd.vel.x.toFixed(0)}, ${hd.vel.y.toFixed(0)}` : '',
        f?.hands.length ? `size    ${f.hands.map(x => x.size.toFixed(3)).join('  ')}` : '',
      ].join('\n');
    }
  }
}
