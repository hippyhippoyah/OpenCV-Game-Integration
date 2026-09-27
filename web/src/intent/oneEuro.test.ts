import { describe, expect, it } from 'vitest';
import { OneEuro } from './oneEuro';

describe('OneEuro', () => {
  const noise = (i: number) => Math.sin(i * 12.9898) * 0.5; // deterministic jitter in [-0.5, 0.5]

  it('smooths jitter on a still signal', () => {
    const f = new OneEuro(1, 0.015);
    let out = 0, maxDev = 0;
    for (let i = 0; i < 120; i++) {
      out = f.filter(10 + noise(i), 1 / 30);
      if (i > 30) maxDev = Math.max(maxDev, Math.abs(out - 10));
    }
    expect(maxDev).toBeLessThan(0.25); // raw jitter is ±0.5
  });

  it('keeps up with a fast move', () => {
    const f = new OneEuro(1, 0.015);
    for (let i = 0; i < 30; i++) f.filter(0, 1 / 30);
    let out = 0;
    for (let i = 1; i <= 5; i++) out = f.filter(i * 40, 1 / 30); // 1200 units/s
    expect(out).toBeGreaterThan(170); // almost all the way to 200 within 5 frames
  });

  it('starts at the first value', () => {
    expect(new OneEuro(1, 0).filter(7, 1 / 30)).toBe(7);
  });
});
