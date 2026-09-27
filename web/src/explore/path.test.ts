import { describe, expect, it } from 'vitest';
import { Look, pauses, Rail, WALK_SPEED } from './path';
import { STOPS } from '../campaign/chapter1';

describe('Rail', () => {
  it('pauses at each scroll and arena, in order', () => {
    const p = pauses();
    expect(p.map(x => x.kind)).toEqual(STOPS.flatMap(s => (s.scroll ? ['scroll', 'arena'] : ['arena'])));
    expect(p.map(x => x.at)).toEqual([...p.map(x => x.at)].sort((a, b) => a - b));
  });

  it('walks at walking speed and stops at the next pause', () => {
    const r = new Rail(0), p = pauses();
    expect(r.advance(1, p)).toBeNull();
    expect(r.d).toBeCloseTo(WALK_SPEED);
    let hit = null;
    for (let i = 0; i < 1000 && !hit; i++) hit = r.advance(0.1, p);
    expect(hit).toEqual(p[0]);
    expect(r.d).toBe(p[0].at);
    expect(r.advance(1, p)).toBeNull(); // stays until moved on past it
  });

  it('walks on past a pause once nudged, and can skip to the next', () => {
    const p = pauses(), r = new Rail(p[0].at + 0.01);
    r.skip(p);
    expect(r.d).toBeGreaterThan(p[1].at - 3);
    expect(r.d).toBeLessThan(p[1].at);
  });

  it('gives a position and heading along the path', () => {
    const a = new Rail(10).pose(), b = new Rail(11).pose();
    expect(Math.hypot(b.pos.x - a.pos.x, b.pos.z - a.pos.z)).toBeGreaterThan(0.5);
    expect(Number.isFinite(a.heading)).toBe(true);
  });
});

describe('Look', () => {
  it('turns with the mouse within limits and drifts back ahead', () => {
    const l = new Look();
    l.move(10000, -10000);
    expect(Math.abs(l.yaw)).toBeLessThanOrEqual(1.25);
    expect(Math.abs(l.pitch)).toBeLessThanOrEqual(0.6);
    const y = l.yaw;
    l.relax(1);
    expect(Math.abs(l.yaw)).toBeLessThan(Math.abs(y));
  });
});
