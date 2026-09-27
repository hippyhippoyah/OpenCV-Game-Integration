import { describe, expect, it } from 'vitest';
import { clamp, distToSeg, mulberry32 } from './math';

describe('math', () => {
  it('clamps', () => {
    expect(clamp(5, 0, 3)).toBe(3);
    expect(clamp(-1, 0, 3)).toBe(0);
  });

  it('measures distance to a segment, including past its ends', () => {
    const a = { x: 0, y: 0 }, b = { x: 10, y: 0 };
    expect(distToSeg({ x: 5, y: 3 }, a, b)).toBeCloseTo(3);
    expect(distToSeg({ x: 13, y: 4 }, a, b)).toBeCloseTo(5);
  });

  it('seeded random is repeatable', () => {
    const r1 = mulberry32(7), r2 = mulberry32(7);
    expect([r1(), r1()]).toEqual([r2(), r2()]);
  });
});
