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

/** A hand relative to the shoulder centre, in shoulder widths (x right, y down). A fist facing the camera by default. */
export interface HandSpec { x: number; y: number; open?: number; size?: number; facing?: number }

export function hand(mid: Vec2, sw: number, p: HandSpec): HandObs {
  return {
    center: { x: mid.x + p.x * sw, y: mid.y + p.y * sw },
    size: (p.size ?? 0.4) * sw,
    open: p.open ?? 0,
    facing: p.facing ?? 1,
  };
}
