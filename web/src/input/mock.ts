import type { ArmObs, HandObs, Side, Tracker, TrackingFrame } from './types';
import type { Calibration } from '../intent/calibration';
import { TUNING, type CastKind, type PalmKind } from '../intent/interpret';
import { FOCAL_H } from './landmarks';
import { clamp, lerp, type Vec2 } from '../math';

/** The body the mock pretends to see, which is also its calibration. */
export const MOCK_CALIBRATION: Calibration = { head: { x: 0.5, y: 0.35 }, sw: 0.2 };
const MID = { x: 0.5, y: 0.5 }, SW = 0.2, HAND_SIZE = 0.08;
/** Fists up at chest height, in view units. */
const GUARD: Record<Side, Vec2> = { l: { x: -12, y: 22 }, r: { x: 12, y: 22 } };
const EXTEND_S = 0.12, OPEN_HOLD_S = 0.25, SHIELD_HALF_WIDTH = 16;
/** Two-hand casts: open hands move for CAST_MOVE_S, then stay open for CAST_HOLD_S. */
const CAST_MOVE_S = 0.25, CAST_HOLD_S = 0.3;
/** Palm push (right hand): opens for PALM_OPEN_S, pushes for PALM_MOVE_S, stays open for PALM_HOLD_S. */
const PALM_OPEN_S = 0.15, PALM_MOVE_S = 0.15, PALM_HOLD_S = 0.25;
/** The mock body stands this far away (m) with shoulders this wide (m); fists rest this far in front (m). */
const MOCK_DISTANCE = 1.5, MOCK_SHOULDERS_M = 0.38, GUARD_REACH_M = 0.25, PUNCH_REACH_M = 0.52;
/** Crossed forearms: each fist on the other side (view units). */
const XBLOCK: Record<Side, Vec2> = { l: { x: 9, y: 12 }, r: { x: -9, y: 12 } };
/** Where the right hand goes while O is held: out past the right edge of the picture. */
const OUT_OF_VIEW: Vec2 = { x: 140, y: 10 };

export interface ViewMapper { screenToView(x: number, y: number): Vec2 }

/**
 * Pretends to be the camera. The mouse is where you aim; a punch drives that fist to the mouse
 * (opening it at the end in the open-hand punch style); holding Space opens both hands around the mouse (shield); A/D/S lean and duck;
 * W sweeps open hands up (fire wall); U spreads open hands apart (ultimate); F pushes both open
 * palms forward (wall push); X crosses the arms;
 * E pushes an open right palm toward the mouse (pillar);
 * holding O swings the right hand out of the picture (only its arm is still tracked).
 */
export class MockTracker implements Tracker {
  private mouse = { x: 0, y: 0 };
  private keys = new Set<string>();
  private lean = 0;
  private duck = 0;
  private lastT: number | null = null;
  private punchStart: Record<Side, number | null> = { l: null, r: null };
  private requested: Side[] = [];
  private castReq: CastKind | null = null;
  private casting: { kind: CastKind; t: number } | null = null;
  private palmReq: PalmKind | null = null;
  private palming: { kind: PalmKind; t: number } | null = null;

  constructor(private view: ViewMapper) {}

  setMouse(x: number, y: number): void { this.mouse = { x, y }; }
  setKey(key: string, down: boolean): void { if (down) this.keys.add(key); else this.keys.delete(key); }
  punch(side: Side): void { this.requested.push(side); }
  cast(kind: CastKind): void { this.castReq = kind; }
  palm(kind: PalmKind): void { this.palmReq = kind; }

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
    if (this.castReq && !this.casting) this.casting = { kind: this.castReq, t };
    this.castReq = null;
    if (this.casting && t - this.casting.t > CAST_MOVE_S + CAST_HOLD_S) this.casting = null;
    if (this.palmReq && !this.palming) this.palming = { kind: this.palmReq, t };
    this.palmReq = null;
    if (this.palming && t - this.palming.t > PALM_OPEN_S + PALM_MOVE_S + PALM_HOLD_S) this.palming = null;

    const toNorm = (p: Vec2): Vec2 => ({
      x: mid.x + (p.x / TUNING.handScaleX) * SW,
      y: mid.y + ((p.y - TUNING.handOffsetY) / TUNING.handScaleY) * SW,
    });
    const hands: HandObs[] = [];
    const arms: Record<Side, ArmObs | null> = { l: null, r: null };
    for (const side of ['l', 'r'] as const) {
      const sign = side === 'l' ? -1 : 1;
      let pos = GUARD[side], open = 0, grow = 0, facing = 1, ext = 0.25, reachM = GUARD_REACH_M;
      const start = this.punchStart[side];
      const since = start === null ? null : t - start;
      if (since !== null && since > EXTEND_S + OPEN_HOLD_S) this.punchStart[side] = null;
      if (this.casting) {
        const e = clamp((t - this.casting.t) / CAST_MOVE_S, 0, 1);
        const kind = this.casting.kind;
        pos = kind === 'wall' ? { x: aim.x + sign * 14, y: lerp(40, aim.y - 10, e) } // from low, sweeping up
          : kind === 'push' ? { x: aim.x + sign * SHIELD_HALF_WIDTH, y: aim.y }       // shoved toward the camera
          : { x: aim.x + sign * lerp(4, 34, e), y: aim.y };                          // from together, flung apart
        open = 1;
        ext = 0.6;
        if (kind === 'push') reachM = 0.3 + (PUNCH_REACH_M - 0.3) * e;
      } else if (this.palming && side === 'r') {
        const e = clamp((t - this.palming.t - PALM_OPEN_S) / PALM_MOVE_S, 0, 1);
        open = 1;
        pos = { x: lerp(GUARD.r.x, aim.x, e), y: lerp(GUARD.r.y, aim.y, e) };
        ext = 0.25 + 0.65 * e;
        reachM = GUARD_REACH_M + (PUNCH_REACH_M - GUARD_REACH_M) * e;
      } else if (this.keys.has('x')) {
        pos = XBLOCK[side];
      } else if (shield) {
        pos = { x: aim.x + sign * SHIELD_HALF_WIDTH, y: aim.y };
        open = 1;
        facing = 0.2;
        ext = 0.8;
        reachM = 0.32;
      } else if (since !== null && since <= EXTEND_S + OPEN_HOLD_S) {
        const e = clamp(since / EXTEND_S, 0, 1);
        pos = { x: lerp(GUARD[side].x, aim.x, e), y: lerp(GUARD[side].y, aim.y, e) };
        grow = 0.3 * e;
        // a real punch stays a fist; only the open-hand style opens at the end
        open = TUNING.punchTrigger === 'open' && since >= EXTEND_S ? 1 : 0;
        ext = 0.25 + 0.65 * e;
        reachM = GUARD_REACH_M + (PUNCH_REACH_M - GUARD_REACH_M) * e;
      }
      const away = side === 'r' && this.keys.has('o');
      if (away) pos = OUT_OF_VIEW;
      const palm = toNorm(pos), shoulder = { x: mid.x + (sign * SW) / 2, y: mid.y };
      const wrist = { x: palm.x, y: palm.y + 0.1 * SW };
      const elbow = { x: lerp(shoulder.x, wrist.x, 0.5), y: lerp(shoulder.y, wrist.y, 0.5) + 0.35 * SW * (1 - ext) };
      const inPicture = wrist.x >= 0 && wrist.x <= 1 && wrist.y >= 0 && wrist.y <= 1;
      arms[side] = {
        shoulder: { ...shoulder, vis: 1 },
        elbow: { ...elbow, vis: 0.9 },
        wrist: { ...wrist, vis: inPicture ? 0.9 : 0.1 },
        extension: ext,
        // shoulder → wrist, pushed toward the camera as the arm straightens (metres-ish)
        reach: { x: (wrist.x - shoulder.x) * 1.5, y: (wrist.y - shoulder.y) * 1.5, z: -0.6 * ext },
      };
      // metres relative to the shoulder centre, for the reach-from-size detection
      const body3 = { x: (pos.x / TUNING.handScaleX) * MOCK_SHOULDERS_M, y: ((pos.y - TUNING.handOffsetY) / TUNING.handScaleY) * MOCK_SHOULDERS_M, z: reachM };
      if (!away) hands.push({ center: palm, size: HAND_SIZE * (1 + grow), open, facing, side, body3, depth: MOCK_DISTANCE - reachM });
    }
    return {
      t, head,
      shoulderL: arms.l!.shoulder,
      shoulderR: arms.r!.shoulder,
      hands,
      arms,
      face: { yaw: 0, roll: 0 },
      body: { span3: MOCK_SHOULDERS_M, span2: (FOCAL_H * MOCK_SHOULDERS_M) / MOCK_DISTANCE },
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
    const k = e.key.toLowerCase();
    if (!e.repeat && k === 'w') m.cast('wall');
    if (!e.repeat && k === 'u') m.cast('ultimate');
    if (!e.repeat && k === 'e') m.palm('push');
    if (!e.repeat && k === 'f') m.cast('push');
    m.setKey(k, true);
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
