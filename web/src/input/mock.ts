import type { HandObs, Tracker, TrackingFrame } from './types';
import type { Calibration } from '../intent/calibration';
import { TUNING, type Side } from '../intent/interpret';
import { clamp, lerp, type Vec2 } from '../math';

/** The body the mock pretends to see, which is also its calibration. */
export const MOCK_CALIBRATION: Calibration = { head: { x: 0.5, y: 0.35 }, sw: 0.2 };
const MID = { x: 0.5, y: 0.5 }, SW = 0.2, HAND_SIZE = 0.08;
/** Fists up at chest height, in view units. */
const GUARD: Record<Side, Vec2> = { l: { x: -12, y: 22 }, r: { x: 12, y: 22 } };
const EXTEND_S = 0.12, OPEN_HOLD_S = 0.25, SHIELD_HALF_WIDTH = 16;

export interface ViewMapper { screenToView(x: number, y: number): Vec2 }

/**
 * Pretends to be the camera. The mouse is where you aim; a punch drives that fist to the mouse
 * and opens it; holding Space opens both hands around the mouse (shield); A/D/S lean and duck.
 */
export class MockTracker implements Tracker {
  private mouse = { x: 0, y: 0 };
  private keys = new Set<string>();
  private lean = 0;
  private duck = 0;
  private lastT: number | null = null;
  private punchStart: Record<Side, number | null> = { l: null, r: null };
  private requested: Side[] = [];

  constructor(private view: ViewMapper) {}

  setMouse(x: number, y: number): void { this.mouse = { x, y }; }
  setKey(key: string, down: boolean): void { if (down) this.keys.add(key); else this.keys.delete(key); }
  punch(side: Side): void { this.requested.push(side); }

  poll(now: number): TrackingFrame {
    const t = now / 1000, dt = this.lastT === null ? 0 : clamp(t - this.lastT, 0, 0.05);
    this.lastT = t;
    const approach = (v: number, target: number, rate: number) => v + (target - v) * Math.min(1, dt * rate);
    this.lean = approach(this.lean, (this.keys.has('a') ? -1 : 0) + (this.keys.has('d') ? 1 : 0), 7);
    this.duck = approach(this.duck, this.keys.has('s') ? 1 : 0, 8);
    for (const side of this.requested) if (this.punchStart[side] === null) this.punchStart[side] = t;
    this.requested = [];

    // Leaning/ducking moves the whole upper body, like it does in front of a real camera.
    const mid = { x: MID.x + this.lean * 0.55 * SW, y: MID.y + this.duck * 0.6 * SW };
    const head = { x: MOCK_CALIBRATION.head.x + mid.x - MID.x, y: MOCK_CALIBRATION.head.y + mid.y - MID.y };
    const aim = this.view.screenToView(this.mouse.x, this.mouse.y);
    const shield = this.keys.has(' ');

    const hand = (side: Side): HandObs => {
      const sign = side === 'l' ? -1 : 1;
      let pos = GUARD[side], open = 0, grow = 0, facing = 1;
      const start = this.punchStart[side];
      const since = start === null ? null : t - start;
      if (since !== null && since > EXTEND_S + OPEN_HOLD_S) this.punchStart[side] = null;
      if (shield) {
        pos = { x: aim.x + sign * SHIELD_HALF_WIDTH, y: aim.y };
        open = 1;
        facing = 0.2;
      } else if (since !== null && since <= EXTEND_S + OPEN_HOLD_S) {
        const e = clamp(since / EXTEND_S, 0, 1);
        pos = { x: lerp(GUARD[side].x, aim.x, e), y: lerp(GUARD[side].y, aim.y, e) };
        grow = 0.3 * e;
        open = since >= EXTEND_S ? 1 : 0;
      }
      return {
        center: { x: mid.x + (pos.x / TUNING.handScaleX) * SW, y: mid.y + ((pos.y - TUNING.handOffsetY) / TUNING.handScaleY) * SW },
        size: HAND_SIZE * (1 + grow),
        open,
        facing,
      };
    };
    return {
      t, head,
      shoulderL: { x: mid.x - SW / 2, y: mid.y },
      shoulderR: { x: mid.x + SW / 2, y: mid.y },
      hands: [hand('l'), hand('r')],
    };
  }

  dispose(): void {}
}

/** Wire browser mouse and keyboard to a MockTracker. Returns an unbind function. */
export function bindMockControls(m: MockTracker, canvas: HTMLElement): () => void {
  const onMove = (e: MouseEvent) => m.setMouse(e.clientX, e.clientY);
  const onDown = (e: MouseEvent) => {
    if (e.button === 0) m.punch('r');
    if (e.button === 2) m.punch('l');
  };
  const onMenu = (e: Event) => e.preventDefault();
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === ' ') e.preventDefault();
    m.setKey(e.key.toLowerCase(), true);
  };
  const onKeyUp = (e: KeyboardEvent) => m.setKey(e.key.toLowerCase(), false);
  addEventListener('mousemove', onMove);
  canvas.addEventListener('mousedown', onDown);
  canvas.addEventListener('contextmenu', onMenu);
  addEventListener('keydown', onKeyDown);
  addEventListener('keyup', onKeyUp);
  return () => {
    removeEventListener('mousemove', onMove);
    canvas.removeEventListener('mousedown', onDown);
    canvas.removeEventListener('contextmenu', onMenu);
    removeEventListener('keydown', onKeyDown);
    removeEventListener('keyup', onKeyUp);
  };
}
