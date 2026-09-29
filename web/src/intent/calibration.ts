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

/** Are you ready to start: seen, both hands brought to your chest, at a good distance? */
export interface SetupChecks { seen: boolean; handsAtChest: boolean; distance: 'ok' | 'close' | 'far' | 'unknown' }

/** Distance to the camera (m) that plays well. */
export const GOOD_DISTANCE_M = { min: 0.9, max: 2.2 };

/** Checks one frame for the start pose: head and shoulders seen, both hands at the chest, not too close or far. */
export function setupChecks(f: TrackingFrame): SetupChecks {
  if (!f.head || !f.shoulderL || !f.shoulderR) return { seen: false, handsAtChest: false, distance: 'unknown' };
  const sw = dist(f.shoulderL, f.shoulderR), mid = { x: (f.shoulderL.x + f.shoulderR.x) / 2, y: (f.shoulderL.y + f.shoulderR.y) / 2 };
  // each hand in front of the chest: within a shoulder width of the middle, from the shoulders to the belly
  const atChest = (p: Vec2) => Math.abs(p.x - mid.x) <= sw && p.y >= mid.y - 0.3 * sw && p.y <= mid.y + 1.4 * sw;
  const handsAtChest = f.hands.length >= 2 && f.hands.slice(0, 2).every(h => atChest(h.center));
  const d = f.body ? (1.05 * f.body.span3) / f.body.span2 : null;
  const distance = d === null ? 'unknown' : d < GOOD_DISTANCE_M.min ? 'close' : d > GOOD_DISTANCE_M.max ? 'far' : 'ok';
  return { seen: true, handsAtChest, distance };
}

/** All set: seen, hands at the chest, and not too close or far (an unknown distance doesn't hold you up). */
export const setupReady = (c: SetupChecks) => c.seen && c.handsAtChest && (c.distance === 'ok' || c.distance === 'unknown');

/** A check failing for longer than this starts calibration over; shorter slips only pause it. */
const SLIP_S = 0.3;

/**
 * Averages the pose of someone holding the start pose (both hands at the chest, see setupChecks)
 * for CALIBRATION_SECONDS.
 */
export class Calibrator {
  private samples: { head: Vec2; sw: number; t: number }[] = [];
  private badSince: number | null = null;

  /** Feed a frame; returns progress in [0, 1]. */
  add(f: TrackingFrame): number {
    if (!f.head || !f.shoulderL || !f.shoulderR || !setupReady(setupChecks(f))) {
      this.badSince ??= f.t;
      if (f.t - this.badSince > SLIP_S) this.samples = [];
      return this.progress();
    }
    this.badSince = null;
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
