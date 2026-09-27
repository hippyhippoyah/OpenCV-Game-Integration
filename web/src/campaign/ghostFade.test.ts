import { describe, expect, it } from 'vitest';
import { GHOST_FADE_S, GHOST_STALL_S, ghostAlpha } from './ghostFade';

describe('ghostAlpha', () => {
  it('stays full strength before the first success', () => {
    expect(ghostAlpha(1, 1 / 60, 0, 0, false)).toBe(1);
    expect(ghostAlpha(1, 1, 0, 10, false)).toBe(1); // even if you're slow, no success yet = full ghost
  });

  it('fades out over ~0.6s after the first success', () => {
    let a = 1;
    for (let t = 0; t < GHOST_FADE_S; t += 1 / 60) a = ghostAlpha(a, 1 / 60, 1, 0, false);
    expect(a).toBeCloseTo(0, 1);
  });

  it('reaches exactly 0 and stays there while progress continues', () => {
    let a = ghostAlpha(1, GHOST_FADE_S, 1, 0, false);
    expect(a).toBe(0);
    a = ghostAlpha(a, 1 / 60, 2, 0, false);
    expect(a).toBe(0);
  });

  it('fades back in after 4s with no progress', () => {
    let a = 0;
    // just under the stall threshold: still hidden
    a = ghostAlpha(a, 1 / 60, 1, GHOST_STALL_S - 0.5, false);
    expect(a).toBe(0);
    // past the threshold: starts fading back in
    a = ghostAlpha(a, 1 / 60, 1, GHOST_STALL_S + 0.1, false);
    expect(a).toBeGreaterThan(0);
    for (let t = 0; t < GHOST_FADE_S; t += 1 / 60) a = ghostAlpha(a, 1 / 60, 1, GHOST_STALL_S + t, false);
    expect(a).toBeCloseTo(1, 1);
  });

  it('is gone once the lesson is complete, fading out from wherever it was', () => {
    expect(ghostAlpha(1, GHOST_FADE_S, 1, 0, true)).toBe(0);
    expect(ghostAlpha(0, 1 / 60, 1, 0, true)).toBe(0);
    const mid = ghostAlpha(1, GHOST_FADE_S / 2, 1, 0, true);
    expect(mid).toBeGreaterThan(0);
    expect(mid).toBeLessThan(1);
  });
});
