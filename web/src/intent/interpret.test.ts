import { describe, expect, it } from 'vitest';
import { initialState, interpret, TUNING, type Intent } from './interpret';
import type { Calibration } from './calibration';
import { bodyFrame, hand, type HandSpec } from '../test/frames';
import type { Vec2 } from '../math';

const cal: Calibration = { head: { x: 0.5, y: 0.35 }, sw: 0.2 };
const MID = { x: 0.5, y: 0.5 }, SW = 0.2, FPS = 30;
const GUARD_L: HandSpec = { x: -0.3, y: 0.1 }, GUARD_R: HandSpec = { x: 0.3, y: 0.1 };

type Pair = HandSpec[];
const repeat = <T>(n: number, f: (i: number) => T): T[] => Array.from({ length: n }, (_, i) => f(i));
/** Shoulder-width position → view units, as interpret maps it. */
const view = (p: HandSpec): Vec2 => ({ x: p.x * TUNING.handScaleX, y: TUNING.handOffsetY + p.y * TUNING.handScaleY });

/** Feed frames of visible hands at 30 fps; returns every intent. */
function play(frames: Pair[], s = initialState()): Intent[] {
  return frames.map((hs, i) => interpret(bodyFrame(i / FPS, { hands: hs.map(h => hand(MID, SW, h)) }), cal, s));
}
const punchesIn = (out: Intent[]) => out.flatMap(o => o.punches);

/** Right fist at guard, snaps to `to` over 4 frames, then opens and holds. */
function rightPunch(to: HandSpec): Pair[] {
  return [
    ...repeat(6, () => [GUARD_L, GUARD_R]),
    ...repeat(4, i => [GUARD_L, { x: GUARD_R.x + ((to.x - GUARD_R.x) * (i + 1)) / 4, y: GUARD_R.y + ((to.y - GUARD_R.y) * (i + 1)) / 4 }]),
    ...repeat(10, () => [GUARD_L, { ...to, open: 1 }]),
  ];
}

describe('interpret', () => {
  it('maps hands relative to the shoulders, independent of distance to the camera', () => {
    const at = (mid: Vec2, sw: number) =>
      interpret(bodyFrame(0, { mid, sw, hands: [hand(mid, sw, { x: -0.5, y: -0.5 }), hand(mid, sw, { x: 0.5, y: -0.5 })] }), cal, initialState());
    for (const r of [at(MID, 0.2), at({ x: 0.5, y: 0.45 }, 0.1)]) {
      expect(r.hands.l!.pos.x).toBeCloseTo(-0.5 * TUNING.handScaleX);
      expect(r.hands.r!.pos.x).toBeCloseTo(0.5 * TUNING.handScaleX);
      expect(r.hands.l!.pos.y).toBeCloseTo(TUNING.handOffsetY - 0.5 * TUNING.handScaleY);
    }
  });

  it('turns head offset into camera lean and duck', () => {
    const r = interpret(bodyFrame(0, { head: { x: 0.6, y: 0.4 } }), cal, initialState());
    expect(r.head.x).toBeCloseTo(0.5 * TUNING.leanUnitsPerSw);
    expect(r.head.y).toBeCloseTo(0.25 * TUNING.duckUnitsPerSw);
  });

  it('reads fist vs open with hysteresis and passes palm facing through', () => {
    const out = play([
      ...repeat(5, () => [GUARD_L, GUARD_R]),
      ...repeat(5, () => [{ ...GUARD_L, open: 0.5 }, GUARD_R]), // half-open still counts as a fist
      ...repeat(8, () => [{ ...GUARD_L, open: 1, facing: 0.2 }, GUARD_R]),
    ]);
    expect(out[4].hands.l!.open).toBe(false);
    expect(out[9].hands.l!.open).toBe(false);
    expect(out[17].hands.l!.open).toBe(true);
    expect(out[17].hands.l!.facing).toBeCloseTo(0.2, 1);
  });

  it('fires one punch from the hand that opens at the end of a fast move, where it opened', () => {
    const to = { x: -0.1, y: -0.6 };
    const p = punchesIn(play(rightPunch(to)));
    expect(p).toHaveLength(1);
    expect(p[0].hand).toBe('r');
    expect(Math.abs(p[0].at.x - view(to).x)).toBeLessThan(2);
    expect(Math.abs(p[0].at.y - view(to).y)).toBeLessThan(2);
    expect(p[0].shoulder.x).toBeCloseTo(0.5 * TUNING.handScaleX);
  });

  it('does not punch when a still fist simply opens', () => {
    const out = play([...repeat(6, () => [GUARD_L, GUARD_R]), ...repeat(10, () => [GUARD_L, { ...GUARD_R, open: 1 }])]);
    expect(punchesIn(out)).toHaveLength(0);
  });

  it('does not punch with hands down at rest', () => {
    const low = { x: 0.3, y: 1.2 };
    const out = play([...repeat(6, () => [GUARD_L, GUARD_R]), ...rightPunch(low).slice(6)]);
    expect(punchesIn(out)).toHaveLength(0);
  });

  it('opening both hands raises a shield instead of punching', () => {
    const out = play([
      ...repeat(6, () => [GUARD_L, GUARD_R]),
      ...repeat(3, i => [{ x: -0.3 - 0.15 * (i + 1), y: 0.1 - 0.2 * (i + 1) }, { x: 0.3 + 0.15 * (i + 1), y: 0.1 - 0.2 * (i + 1) }]),
      [{ x: -0.75, y: -0.5 }, { x: 0.75, y: -0.5, open: 1 }],
      ...repeat(10, () => [{ x: -0.75, y: -0.5, open: 1 }, { x: 0.75, y: -0.5, open: 1 }]),
    ]);
    expect(punchesIn(out)).toHaveLength(0);
    expect(out[11].shield).toBe(false);
    expect(out[out.length - 1].shield).toBe(true);
  });

  it("keeps each hand's identity when a punch crosses the body", () => {
    const frames: Pair[] = [
      ...repeat(5, () => [GUARD_L, GUARD_R]),
      ...repeat(20, i => [GUARD_L, { x: 0.3 - (0.9 * (i + 1)) / 20, y: -0.2 }]),
    ].map((hs, i) => (i % 2 ? [...hs].reverse() : hs)); // detection order must not matter
    const last = play(frames).at(-1)!;
    expect(last.hands.r!.pos.x).toBeLessThan(last.hands.l!.pos.x);
  });

  it('keeps hands through a short tracking dropout, then lets go', () => {
    const s = initialState();
    interpret(bodyFrame(0, { hands: [hand(MID, SW, GUARD_L), hand(MID, SW, GUARD_R)] }), cal, s);
    expect(interpret(bodyFrame(0.3), cal, s).hands.l).not.toBeNull();
    expect(interpret(bodyFrame(0.6), cal, s).hands.l).toBeNull();
  });

  it('reports nobody present without a head and shoulders', () => {
    const r = interpret({ t: 0, head: null, shoulderL: null, shoulderR: null, hands: [] }, cal, initialState());
    expect(r.present).toBe(false);
    expect(r.hands.l).toBeNull();
  });
});
