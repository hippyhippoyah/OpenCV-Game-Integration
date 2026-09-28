import { PATH_LENGTH, STOPS } from '../campaign/chapter1';

export interface MapPt { x: number; y: number }
/** A place on the path where the walk stops for you: a scroll to pick up, or an arena. */
export interface Pause { at: number; kind: 'scroll' | 'arena'; stop: number }

/** Walking pace along the path, metres per second. */
export const WALK_SPEED = 3.2;

/** The map is drawn in a MAP_W × MAP_H space (see map2d.ts). */
export const MAP_W = 1600, MAP_H = 1000;

/**
 * The path on the map, as points at distances along it: from the temple (top left) down past the
 * stairs, over the river by the bamboo bridge, through the stone garden to the village gate.
 * The stops and scrolls in chapter1.ts sit on these distances.
 */
export const ROUTE: { d: number; x: number; y: number }[] = [
  { d: 0, x: 170, y: 190 }, { d: 15, x: 225, y: 228 }, { d: 30, x: 300, y: 250 },
  { d: 50, x: 415, y: 238 }, { d: 68, x: 510, y: 290 }, { d: 80, x: 548, y: 360 },
  { d: 100, x: 520, y: 452 }, { d: 122, x: 600, y: 522 }, { d: 135, x: 690, y: 548 },
  { d: 155, x: 810, y: 526 }, { d: 176, x: 920, y: 560 }, { d: 190, x: 985, y: 615 },
  { d: 210, x: 955, y: 698 }, { d: 232, x: 1025, y: 768 }, { d: 245, x: 1105, y: 795 },
  { d: 262, x: 1195, y: 822 }, { d: PATH_LENGTH, x: 1390, y: 880 },
];

export function pauses(): Pause[] {
  return STOPS.flatMap((s, i): Pause[] => [
    ...(s.scroll && s.scrollAt !== undefined ? [{ at: s.scrollAt, kind: 'scroll' as const, stop: i }] : []),
    { at: s.pathAt, kind: 'arena', stop: i },
  ]).sort((a, b) => a.at - b.at);
}

/** Where on the map you are `d` metres along the path (a smooth curve through ROUTE). */
export function pointAt(d: number): MapPt {
  const t = Math.max(0, Math.min(PATH_LENGTH, d));
  let i = 1;
  while (i < ROUTE.length - 1 && ROUTE[i].d < t) i++;
  const p0 = ROUTE[Math.max(0, i - 2)], p1 = ROUTE[i - 1], p2 = ROUTE[i], p3 = ROUTE[Math.min(ROUTE.length - 1, i + 1)];
  const k = (t - p1.d) / (p2.d - p1.d || 1), k2 = k * k, k3 = k2 * k;
  const cr = (a: number, b: number, c: number, e: number) =>
    0.5 * (2 * b + (-a + c) * k + (2 * a - 5 * b + 4 * c - e) * k2 + (-a + 3 * b - 3 * c + e) * k3);
  return { x: cr(p0.x, p1.x, p2.x, p3.x), y: cr(p0.y, p1.y, p2.y, p3.y) };
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

  /** Jump ahead to just short of `at` (the next step's walk arrives there). */
  skipTo(at: number): void {
    this.d = Math.max(this.d, at - 1e-3);
  }
}
