import { describe, expect, it } from 'vitest';
import { initialState, interpret, TUNING } from './interpret';
import type { Calibration } from './calibration';
import { bodyFrame, handsRel } from '../test/frames';

const cal: Calibration = { head: { x: 0.5, y: 0.35 }, sw: 0.2 };
const MID = { x: 0.5, y: 0.5 };

function pushSequence(growthPerFrame: number, spreadSw = 0.1) {
  const s = initialState();
  const out = [];
  for (let i = 0; i < 12; i++) {
    const size = 0.4 * (1 + growthPerFrame * i);
    const hands = handsRel(MID, 0.2, { x: -spreadSw / 2, y: -0.3 }, { x: spreadSw / 2, y: -0.3 }, size);
    out.push(interpret(bodyFrame(i / 30, { hands }), cal, s));
  }
  return out;
}

describe('interpret', () => {
  it('maps hands relative to the shoulders, independent of distance to the camera', () => {
    const near = interpret(
      bodyFrame(0, { sw: 0.2, hands: handsRel(MID, 0.2, { x: -0.5, y: -0.5 }, { x: 0.5, y: -0.5 }) }), cal, initialState());
    const farMid = { x: 0.5, y: 0.45 };
    const far = interpret(
      bodyFrame(0, { mid: farMid, sw: 0.1, hands: handsRel(farMid, 0.1, { x: -0.5, y: -0.5 }, { x: 0.5, y: -0.5 }) }), cal, initialState());
    for (const r of [near, far]) {
      expect(r.hands!.l.x).toBeCloseTo(-0.5 * TUNING.handScaleX);
      expect(r.hands!.r.x).toBeCloseTo(0.5 * TUNING.handScaleX);
      expect(r.hands!.center.y).toBeCloseTo(TUNING.handOffsetY - 0.5 * TUNING.handScaleY);
    }
  });

  it('labels the left-most hand as l regardless of detection order', () => {
    const hands = handsRel(MID, 0.2, { x: 0.6, y: 0 }, { x: -0.6, y: 0 });
    const r = interpret(bodyFrame(0, { hands }), cal, initialState());
    expect(r.hands!.l.x).toBeLessThan(r.hands!.r.x);
  });

  it('turns head offset into camera lean and duck', () => {
    const r = interpret(bodyFrame(0, { head: { x: 0.6, y: 0.4 } }), cal, initialState());
    expect(r.head.x).toBeCloseTo(0.5 * TUNING.leanUnitsPerSw);
    expect(r.head.y).toBeCloseTo(0.25 * TUNING.duckUnitsPerSw);
  });

  it('reports raised hands at chest height but not at the hips', () => {
    const up = interpret(bodyFrame(0, { hands: handsRel(MID, 0.2, { x: -0.1, y: 0 }, { x: 0.1, y: 0 }) }), cal, initialState());
    const down = interpret(bodyFrame(0, { hands: handsRel(MID, 0.2, { x: -0.1, y: 1.2 }, { x: 0.1, y: 1.2 }) }), cal, initialState());
    expect(up.raised).toBe(true);
    expect(down.raised).toBe(false);
  });

  it('fires exactly one throw when both hands push quickly toward the camera', () => {
    expect(pushSequence(0.06).filter(r => r.throwNow)).toHaveLength(1);
  });

  it('ignores slow drift in hand size', () => {
    expect(pushSequence(0.005).some(r => r.throwNow)).toBe(false);
  });

  it('does not throw when the hands are spread apart', () => {
    expect(pushSequence(0.06, 1.2).some(r => r.throwNow)).toBe(false);
  });

  it('keeps the hands through a short tracking dropout, then lets go', () => {
    const s = initialState();
    const hands = handsRel(MID, 0.2, { x: -0.1, y: 0 }, { x: 0.1, y: 0 });
    interpret(bodyFrame(0, { hands }), cal, s);
    expect(interpret(bodyFrame(0.3), cal, s).hands).not.toBeNull();
    expect(interpret(bodyFrame(0.6), cal, s).hands).toBeNull();
  });

  it('reports nobody present without a head and shoulders', () => {
    const r = interpret({ t: 0, head: null, shoulderL: null, shoulderR: null, hands: [] }, cal, initialState());
    expect(r.present).toBe(false);
    expect(r.hands).toBeNull();
  });
});
