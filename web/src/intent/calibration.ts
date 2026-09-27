import type { TrackingFrame } from '../input/types';
import { dist, type Vec2 } from '../math';

export interface Calibration {
  /** Neutral head position, normalized video coords. */
  head: Vec2;
  /** Neutral shoulder width, normalized. */
  sw: number;
}

export const CALIBRATION_SECONDS = 1.5;
/** Head drift, in shoulder widths, that restarts calibration. */
const MAX_DRIFT_SW = 0.15;

/** Averages the pose of someone standing still for CALIBRATION_SECONDS. */
export class Calibrator {
  private samples: { head: Vec2; sw: number; t: number }[] = [];

  /** Feed a frame; returns progress in [0, 1]. */
  add(f: TrackingFrame): number {
    if (!f.head || !f.shoulderL || !f.shoulderR) {
      this.samples = [];
      return 0;
    }
    const sw = dist(f.shoulderL, f.shoulderR);
    const first = this.samples[0];
    if (first && dist(first.head, f.head) / sw > MAX_DRIFT_SW) this.samples = [];
    this.samples.push({ head: { ...f.head }, sw, t: f.t });
    return this.progress();
  }

  progress(): number {
    const n = this.samples.length;
    if (n < 2) return 0;
    return Math.min(1, (this.samples[n - 1].t - this.samples[0].t) / CALIBRATION_SECONDS);
  }

  result(): Calibration | null {
    if (this.progress() < 1) return null;
    const n = this.samples.length, head = { x: 0, y: 0 };
    let sw = 0;
    for (const s of this.samples) {
      head.x += s.head.x / n;
      head.y += s.head.y / n;
      sw += s.sw / n;
    }
    return { head, sw };
  }
}
