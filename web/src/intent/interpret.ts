import type { HandObs, TrackingFrame } from '../input/types';
import type { Calibration } from './calibration';
import { clamp, dist, type Vec2 } from '../math';

/**
 * View space: world units relative to the eyes, x right, y down.
 * The screen shows roughly x ∈ ±80, y ∈ −45…55.
 */
export interface HandsIntent { l: Vec2; r: Vec2; center: Vec2; spread: number; vel: Vec2 }

export interface Intent {
  /** A head and shoulders are visible. */
  present: boolean;
  /** Camera offset in world units: lean → x, duck → y (down is +). */
  head: Vec2;
  hands: HandsIntent | null;
  /** Hands are up in front of the body (not resting at the hips). */
  raised: boolean;
  /** A push toward the camera happened this frame. */
  throwNow: boolean;
}

export const TUNING = {
  leanUnitsPerSw: 40, maxLean: 30,
  duckUnitsPerSw: 40, minDuck: -10, maxDuck: 25,
  /** Hand offset from the shoulder centre (in shoulder widths) × scale = view units. */
  handScaleX: 40, handScaleY: 32, handOffsetY: 20,
  raisedAboveY: 37,
  /** Exponential smoothing rate, 1/s. Higher = snappier but jittery. */
  smoothing: 18,
  lostGraceS: 0.5,
  /** A throw = average hand size grows by pushGrowth× within pushWindowS. */
  pushWindowS: 0.15, pushGrowth: 1.18, maxPushSpread: 20, throwCooldownS: 0.4,
};

export interface InterpretState {
  head: Vec2 | null;
  l: Vec2 | null;
  r: Vec2 | null;
  center: Vec2 | null;
  vel: Vec2;
  lastT: number | null;
  lastSeen: number;
  sizes: { t: number; size: number }[];
  lastThrow: number;
}

export const initialState = (): InterpretState => ({
  head: null, l: null, r: null, center: null, vel: { x: 0, y: 0 },
  lastT: null, lastSeen: -Infinity, sizes: [], lastThrow: -Infinity,
});

const smooth = (prev: Vec2 | null, next: Vec2, k: number): Vec2 =>
  prev ? { x: prev.x + (next.x - prev.x) * k, y: prev.y + (next.y - prev.y) * k } : { ...next };

export function interpret(f: TrackingFrame, cal: Calibration, s: InterpretState): Intent {
  const dt = s.lastT === null ? 0 : Math.max(1e-3, f.t - s.lastT);
  s.lastT = f.t;
  const k = dt === 0 ? 1 : 1 - Math.exp(-TUNING.smoothing * dt);

  if (!f.head || !f.shoulderL || !f.shoulderR) {
    return { present: false, head: s.head ? { ...s.head } : { x: 0, y: 0 }, hands: null, raised: false, throwNow: false };
  }
  const sw = dist(f.shoulderL, f.shoulderR) || cal.sw;
  const mid = { x: (f.shoulderL.x + f.shoulderR.x) / 2, y: (f.shoulderL.y + f.shoulderR.y) / 2 };

  s.head = smooth(s.head, {
    x: clamp(((f.head.x - cal.head.x) / sw) * TUNING.leanUnitsPerSw, -TUNING.maxLean, TUNING.maxLean),
    y: clamp(((f.head.y - cal.head.y) / sw) * TUNING.duckUnitsPerSw, TUNING.minDuck, TUNING.maxDuck),
  }, k);

  if (f.hands.length >= 2) {
    const [a, b] = [...f.hands].sort((p, q) => p.center.x - q.center.x);
    const toView = (h: HandObs): Vec2 => ({
      x: ((h.center.x - mid.x) / sw) * TUNING.handScaleX,
      y: TUNING.handOffsetY + ((h.center.y - mid.y) / sw) * TUNING.handScaleY,
    });
    s.l = smooth(s.l, toView(a), k);
    s.r = smooth(s.r, toView(b), k);
    s.lastSeen = f.t;
    s.sizes.push({ t: f.t, size: (a.size + b.size) / 2 / sw });
    while (s.sizes.length && f.t - s.sizes[0].t > TUNING.pushWindowS) s.sizes.shift();
  } else if (f.t - s.lastSeen > TUNING.lostGraceS) {
    s.l = s.r = s.center = null;
    s.sizes = [];
    s.vel = { x: 0, y: 0 };
  }

  let hands: HandsIntent | null = null, throwNow = false;
  if (s.l && s.r) {
    const center = { x: (s.l.x + s.r.x) / 2, y: (s.l.y + s.r.y) / 2 };
    if (s.center && dt > 0) {
      const kv = 1 - Math.exp(-12 * dt);
      s.vel = {
        x: s.vel.x + ((center.x - s.center.x) / dt - s.vel.x) * kv,
        y: s.vel.y + ((center.y - s.center.y) / dt - s.vel.y) * kv,
      };
    }
    s.center = center;
    hands = { l: { ...s.l }, r: { ...s.r }, center: { ...center }, spread: dist(s.l, s.r), vel: { ...s.vel } };

    const oldest = s.sizes[0], newest = s.sizes[s.sizes.length - 1];
    if (oldest && newest && newest.size / oldest.size >= TUNING.pushGrowth
      && hands.spread <= TUNING.maxPushSpread && center.y < TUNING.raisedAboveY
      && f.t - s.lastThrow >= TUNING.throwCooldownS) {
      throwNow = true;
      s.lastThrow = f.t;
      s.sizes = [];
    }
  }
  return { present: true, head: { ...s.head }, hands, raised: !!hands && hands.center.y < TUNING.raisedAboveY, throwNow };
}
