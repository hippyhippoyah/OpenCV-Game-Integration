import type { HandObs, Tracker, TrackingFrame } from './types';
import type { Calibration } from '../intent/calibration';
import { TUNING } from '../intent/interpret';
import { clamp, type Vec2 } from '../math';

/** The body the mock pretends to see, which is also its calibration. */
export const MOCK_CALIBRATION: Calibration = { head: { x: 0.5, y: 0.35 }, sw: 0.2 };
const MID = { x: 0.5, y: 0.5 }, SW = 0.2, HAND_SIZE = 0.08;
const PUSH_RAMP_S = 0.12, PUSH_HOLD_S = 0.3;

export interface ViewMapper { screenToView(x: number, y: number): Vec2 }

/** Pretends to be the camera: mouse = both hands, scroll = spread, A/D/S = lean/duck, click = push. */
export class MockTracker implements Tracker {
  private mouse = { x: 0, y: 0 };
  private keys = new Set<string>();
  private spreadT = 6;
  private spread = 6;
  private lean = 0;
  private duck = 0;
  private pushAt: number | null = null;
  private pushRequested = false;
  private lastT: number | null = null;

  constructor(private view: ViewMapper) {}

  setMouse(x: number, y: number): void { this.mouse = { x, y }; }
  setKey(key: string, down: boolean): void { if (down) this.keys.add(key); else this.keys.delete(key); }
  wheel(deltaY: number): void { this.spreadT = clamp(this.spreadT - deltaY * 0.03, 4, 40); }
  setSpread(units: number): void { this.spreadT = units; }
  push(): void { this.pushRequested = true; }

  poll(now: number): TrackingFrame {
    const t = now / 1000, dt = this.lastT === null ? 0 : clamp(t - this.lastT, 0, 0.05);
    this.lastT = t;
    const approach = (v: number, target: number, rate: number) => v + (target - v) * Math.min(1, dt * rate);
    this.lean = approach(this.lean, (this.keys.has('a') ? -1 : 0) + (this.keys.has('d') ? 1 : 0), 7);
    this.duck = approach(this.duck, this.keys.has('s') ? 1 : 0, 8);
    this.spread = approach(this.spread, this.spreadT, 12);

    if (this.pushRequested) { this.pushAt = t; this.pushRequested = false; }
    if (this.pushAt !== null && t - this.pushAt > PUSH_HOLD_S) this.pushAt = null;
    const push = this.pushAt === null ? 0 : clamp((t - this.pushAt) / PUSH_RAMP_S, 0, 1);

    // Leaning/ducking moves the whole upper body, like it does in front of a real camera.
    const mid = { x: MID.x + this.lean * 0.55 * SW, y: MID.y + this.duck * 0.6 * SW };
    const head = { x: MOCK_CALIBRATION.head.x + mid.x - MID.x, y: MOCK_CALIBRATION.head.y + mid.y - MID.y };
    const v = this.view.screenToView(this.mouse.x, this.mouse.y);
    const hand = (vx: number): HandObs => ({
      center: {
        x: mid.x + (vx / TUNING.handScaleX) * SW,
        y: mid.y + ((v.y - TUNING.handOffsetY) / TUNING.handScaleY) * SW,
      },
      size: HAND_SIZE * (1 + 0.35 * push),
    });
    return {
      t, head,
      shoulderL: { x: mid.x - SW / 2, y: mid.y },
      shoulderR: { x: mid.x + SW / 2, y: mid.y },
      hands: [hand(v.x - this.spread / 2), hand(v.x + this.spread / 2)],
    };
  }

  dispose(): void {}
}

/** Wire browser mouse and keyboard to a MockTracker. Returns an unbind function. */
export function bindMockControls(m: MockTracker, canvas: HTMLElement): () => void {
  const onMove = (e: MouseEvent) => m.setMouse(e.clientX, e.clientY);
  const onDown = (e: MouseEvent) => { if (e.button === 0) m.push(); };
  const onWheel = (e: WheelEvent) => m.wheel(e.deltaY);
  const onKeyDown = (e: KeyboardEvent) => {
    const k = e.key.toLowerCase();
    if (k === '1') m.setSpread(6);
    if (k === '2') m.setSpread(34);
    m.setKey(k, true);
  };
  const onKeyUp = (e: KeyboardEvent) => m.setKey(e.key.toLowerCase(), false);
  addEventListener('mousemove', onMove);
  canvas.addEventListener('mousedown', onDown);
  addEventListener('wheel', onWheel, { passive: true });
  addEventListener('keydown', onKeyDown);
  addEventListener('keyup', onKeyUp);
  return () => {
    removeEventListener('mousemove', onMove);
    canvas.removeEventListener('mousedown', onDown);
    removeEventListener('wheel', onWheel);
    removeEventListener('keydown', onKeyDown);
    removeEventListener('keyup', onKeyUp);
  };
}
