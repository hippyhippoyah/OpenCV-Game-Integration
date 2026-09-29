import { describe, expect, it } from 'vitest';
import { FEATURES } from './config';
import { activeLessons } from './game/tutorial';
import { guardState, simulate, type Reach } from './sim/synthetic';
import { initialState, interpret } from './intent/interpret';
import { dist } from './math';

describe('switched-off features', () => {
  it('charged punches off: no charge lesson, and a fist held at the hip never charges', () => {
    FEATURES.chargedPunch = false;
    expect(activeLessons().map(l => l.id)).not.toContain('charge');
    const HIP: Reach = { out: 0.04, up: -0.42, fwd: -0.04 };
    const frames = simulate(t => guardState(1.5, { r: { reach: t < 1 ? { out: -0.08, up: 0.02, fwd: 0.25 } : HIP } }), 3, { seed: 1 });
    const cal = { head: frames[0].head!, sw: dist(frames[0].shoulderL!, frames[0].shoulderR!) };
    const s = initialState();
    expect(frames.map(f => interpret(f, cal, s)).every(o => (o.hands.r?.charge ?? 0) === 0)).toBe(true);
  });

  it('charged punches on: the lesson is there', () => {
    expect(activeLessons().map(l => l.id)).toContain('charge');
  });
});
