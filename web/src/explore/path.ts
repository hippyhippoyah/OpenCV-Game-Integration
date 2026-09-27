import { PATH_LENGTH, STOPS } from '../campaign/chapter1';

export interface V3 { x: number; y: number; z: number }
/** A place on the path where the walk stops for you: a scroll to pick up, or an arena. */
export interface Pause { at: number; kind: 'scroll' | 'arena'; stop: number }

/** Walking pace along the path, metres per second. */
export const WALK_SPEED = 3.2;

/**
 * The path's centre line, from the temple courtyard (top) down the mountain to the village gate.
 * x/z across the ground, y up (metres). The places (see world3d.ts) sit along it.
 */
export const WAYPOINTS: V3[] = [
  { x: 0, y: 40, z: 0 }, { x: 0, y: 40, z: -30 },            // courtyard
  { x: 12, y: 32, z: -50 }, { x: 24, y: 22, z: -70 },        // the long stairs
  { x: 26, y: 20, z: -85 }, { x: 20, y: 18, z: -135 },       // bamboo bridge
  { x: 5, y: 12, z: -160 }, { x: -10, y: 8, z: -190 },       // stone garden
  { x: -12, y: 4, z: -225 }, { x: -8, y: 2, z: -262 },       // the village gate
  { x: -6, y: 1, z: -300 },
];

/** Cumulative distance at each waypoint, scaled so the last is PATH_LENGTH. */
const CUM = (() => {
  const raw = [0];
  for (let i = 1; i < WAYPOINTS.length; i++) {
    const a = WAYPOINTS[i - 1], b = WAYPOINTS[i];
    raw.push(raw[i - 1] + Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z));
  }
  const k = PATH_LENGTH / raw[raw.length - 1];
  return raw.map(d => d * k);
})();

export function pauses(): Pause[] {
  return STOPS.flatMap((s, i): Pause[] => [
    ...(s.scroll && s.scrollAt !== undefined ? [{ at: s.scrollAt, kind: 'scroll' as const, stop: i }] : []),
    { at: s.pathAt, kind: 'arena', stop: i },
  ]).sort((a, b) => a.at - b.at);
}

/** Point on the path `d` metres along it. */
export function pointAt(d: number): V3 {
  const t = Math.max(0, Math.min(PATH_LENGTH, d));
  let i = 1;
  while (i < CUM.length - 1 && CUM[i] < t) i++;
  const a = WAYPOINTS[i - 1], b = WAYPOINTS[i], k = (t - CUM[i - 1]) / (CUM[i] - CUM[i - 1] || 1);
  return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k, z: a.z + (b.z - a.z) * k };
}

/** Auto-walk along the path, stopping at each pause. */
export class Rail {
  constructor(public d = 0) {}

  advance(dt: number, from: Pause[]): Pause | null {
    const next = from.find(p => p.at > this.d + 1e-6);
    const onOne = from.find(p => Math.abs(p.at - this.d) < 1e-6);
    if (onOne) return null; // waiting here until moved on
    const target = next ? next.at : PATH_LENGTH;
    this.d = Math.min(target, this.d + WALK_SPEED * dt);
    return next && this.d >= next.at ? next : null;
  }

  /** Move on past the pause you're standing at (after it's done). */
  leave(): void { this.d += 0.02; }

  skip(from: Pause[]): void {
    const next = from.find(p => p.at > this.d + 1e-6);
    if (next) this.d = Math.max(this.d, next.at - 2);
  }

  pose(): { pos: V3; heading: number } {
    const a = pointAt(this.d), b = pointAt(this.d + 1.5);
    return { pos: a, heading: Math.atan2(-(b.x - a.x), -(b.z - a.z)) };
  }
}

/** Looking around with the mouse (radians, limited), drifting back to straight ahead. */
export class Look {
  yaw = 0;
  pitch = 0;

  move(dx: number, dy: number): void {
    this.yaw = Math.max(-1.2, Math.min(1.2, this.yaw - dx * 0.0025));
    this.pitch = Math.max(-0.55, Math.min(0.55, this.pitch - dy * 0.0025));
  }

  relax(dt: number): void {
    const k = Math.exp(-dt * 0.8);
    this.yaw *= k;
    this.pitch *= k;
  }
}
