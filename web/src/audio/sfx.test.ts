import { describe, expect, it } from 'vitest';
import { Sfx } from './sfx';

describe('Sfx', () => {
  it('places a sound left or right of you by where it happened, never fully in one ear', () => {
    expect(Sfx.panOf(0)).toBe(0);
    expect(Sfx.panOf(-35)).toBeCloseTo(-0.5);
    expect(Sfx.panOf(40, 40)).toBe(0);
    expect(Sfx.panOf(500)).toBe(0.8);
    expect(Sfx.panOf(-500)).toBe(-0.8);
  });
});
