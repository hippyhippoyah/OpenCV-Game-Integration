import { describe, expect, it } from 'vitest';
import { toFrame, type Landmark } from './landmarks';

const pose = (): Landmark[] => Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 1 }));

describe('toFrame', () => {
  it('mirrors x, orders shoulders left-to-right on screen and summarises hands', () => {
    const p = pose();
    p[0] = { x: 0.4, y: 0.3, visibility: 1 };
    p[11] = { x: 0.6, y: 0.5, visibility: 1 };
    p[12] = { x: 0.4, y: 0.5, visibility: 1 };
    const hand: Landmark[] = Array.from({ length: 21 }, () => ({ x: 0.3, y: 0.6 }));
    hand[0] = { x: 0.3, y: 0.7 };
    const f = toFrame(1, [hand], p);
    expect(f.head!.x).toBeCloseTo(0.6);
    expect(f.shoulderL!.x).toBeCloseTo(0.4);
    expect(f.shoulderR!.x).toBeCloseTo(0.6);
    expect(f.hands[0].center.x).toBeCloseTo(0.7);
    expect(f.hands[0].center.y).toBeCloseTo(0.62);
    expect(f.hands[0].size).toBeCloseTo(0.1);
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
