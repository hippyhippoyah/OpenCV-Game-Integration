import type { HandObs, TrackingFrame } from '../input/types';
import type { Vec2 } from '../math';

/** A standing body: shoulders centred on `mid`, head 0.75 shoulder-widths above them. */
export function bodyFrame(t: number, o: { mid?: Vec2; sw?: number; head?: Vec2; hands?: HandObs[] } = {}): TrackingFrame {
  const sw = o.sw ?? 0.2, mid = o.mid ?? { x: 0.5, y: 0.5 };
  return {
    t,
    head: o.head ?? { x: mid.x, y: mid.y - 0.75 * sw },
    shoulderL: { x: mid.x - sw / 2, y: mid.y },
    shoulderR: { x: mid.x + sw / 2, y: mid.y },
    hands: o.hands ?? [],
  };
}

/** Two hands placed relative to the shoulders, in shoulder widths (x right, y down). */
export function handsRel(mid: Vec2, sw: number, left: Vec2, right: Vec2, size = 0.4): HandObs[] {
  return [left, right].map(p => ({ center: { x: mid.x + p.x * sw, y: mid.y + p.y * sw }, size: size * sw }));
}
