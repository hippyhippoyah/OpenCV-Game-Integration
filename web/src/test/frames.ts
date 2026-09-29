import type { ArmObs, HandObs, Side, TrackingFrame } from '../input/types';
import type { Vec2 } from '../math';

/** A standing body: shoulders centred on `mid`, head 0.75 shoulder-widths above them. */
export function bodyFrame(
  t: number,
  o: { mid?: Vec2; sw?: number; head?: Vec2; hands?: HandObs[]; arms?: Partial<Record<Side, ArmObs | null>> } = {},
): TrackingFrame {
  const sw = o.sw ?? 0.2, mid = o.mid ?? { x: 0.5, y: 0.5 };
  return {
    t,
    head: o.head ?? { x: mid.x, y: mid.y - 0.75 * sw },
    shoulderL: { x: mid.x - sw / 2, y: mid.y },
    shoulderR: { x: mid.x + sw / 2, y: mid.y },
    hands: o.hands ?? [],
    arms: { l: o.arms?.l ?? null, r: o.arms?.r ?? null },
    face: null,
  };
}

/** An arm hanging from its shoulder with the wrist at `wrist` (shoulder widths from the shoulder centre). */
export function arm(mid: Vec2, sw: number, side: Side, wrist: { x: number; y: number; vis?: number }, extension: number | null = 0.3): ArmObs {
  const sx = side === 'l' ? -0.5 : 0.5;
  const at = (x: number, y: number, vis = 1) => ({ x: mid.x + x * sw, y: mid.y + y * sw, vis });
  return {
    shoulder: at(sx, 0),
    elbow: at((sx + wrist.x) / 2, (0 + wrist.y) / 2 + 0.3),
    wrist: at(wrist.x, wrist.y, wrist.vis ?? 1),
    extension,
    reach: null,
  };
}

/** A hand relative to the shoulder centre, in shoulder widths (x right, y down). A fist facing the camera by default. */
export interface HandSpec { x: number; y: number; open?: number; size?: number; facing?: number; side?: Side; fingers?: number[] }

export function hand(mid: Vec2, sw: number, p: HandSpec): HandObs {
  return {
    center: { x: mid.x + p.x * sw, y: mid.y + p.y * sw },
    size: (p.size ?? 0.4) * sw,
    open: p.open ?? 0,
    facing: p.facing ?? 1,
    ...(p.side ? { side: p.side } : {}),
    ...(p.fingers ? { fingers: p.fingers } : {}),
  };
}
