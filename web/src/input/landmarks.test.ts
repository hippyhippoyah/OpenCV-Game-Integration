import { describe, expect, it } from 'vitest';
import { openness, palmFacing, toFrame, type Landmark } from './landmarks';

const pose = (): Landmark[] => Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 1 }));

/** 3D hand in metres: wrist at the origin, fingers pointing up (−y). 'zy' turns the palm edge-on to the camera. */
function worldHand(curled: boolean, plane: 'xy' | 'zy' = 'xy'): Landmark[] {
  const lm: Landmark[] = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
  const put = (i: number, across: number, y: number, depth: number) => {
    lm[i] = plane === 'xy' ? { x: across, y, z: depth } : { x: depth, y, z: across };
  };
  [-0.02, -0.007, 0.007, 0.02].forEach((across, f) => {
    const k = 5 + f * 4;
    put(k, across, -0.04, 0);
    put(k + 1, across, -0.065, 0);
    if (curled) {
      put(k + 2, across, -0.065, 0.02);
      put(k + 3, across, -0.045, 0.02);
    } else {
      put(k + 2, across, -0.085, 0);
      put(k + 3, across, -0.1, 0);
    }
  });
  return lm;
}

describe('toFrame', () => {
  it('mirrors x, orders shoulders left-to-right on screen and summarises hands', () => {
    const p = pose();
    p[0] = { x: 0.4, y: 0.3, visibility: 1 };
    p[11] = { x: 0.6, y: 0.5, visibility: 1 };
    p[12] = { x: 0.4, y: 0.5, visibility: 1 };
    const hand: Landmark[] = Array.from({ length: 21 }, () => ({ x: 0.3, y: 0.6 }));
    hand[0] = { x: 0.3, y: 0.7 };
    const f = toFrame(1, [hand], p, [worldHand(false)]);
    expect(f.head!.x).toBeCloseTo(0.6);
    expect(f.shoulderL!.x).toBeCloseTo(0.4);
    expect(f.shoulderR!.x).toBeCloseTo(0.6);
    expect(f.hands[0].center.x).toBeCloseTo(0.7);
    expect(f.hands[0].center.y).toBeCloseTo(0.62);
    expect(f.hands[0].size).toBeCloseTo(0.1);
    expect(f.hands[0].open).toBeGreaterThan(0.9);
    expect(f.hands[0].facing).toBeGreaterThan(0.9);
  });

  it('drops points the model is unsure about', () => {
    const p = pose();
    p[0].visibility = 0.2;
    p[11].visibility = 0.2;
    const f = toFrame(0, [], p);
    expect(f.head).toBeNull();
    expect(f.shoulderL).toBeNull();
  });

  it('handles no person', () => {
    expect(toFrame(0, [], undefined)).toEqual({ t: 0, head: null, shoulderL: null, shoulderR: null, hands: [] });
  });
});

describe('hand shape', () => {
  it('tells an open hand from a fist in 3D', () => {
    expect(openness(worldHand(false))).toBeGreaterThan(0.9);
    expect(openness(worldHand(true))).toBeLessThan(0.1);
  });

  it('gives the same answer when the hand points another way', () => {
    expect(openness(worldHand(false, 'zy'))).toBeGreaterThan(0.9);
    expect(openness(worldHand(true, 'zy'))).toBeLessThan(0.1);
  });

  it('measures whether the palm faces the camera or is edge-on', () => {
    expect(palmFacing(worldHand(false, 'xy'))).toBeGreaterThan(0.9);
    expect(palmFacing(worldHand(false, 'zy'))).toBeLessThan(0.1);
  });
});
