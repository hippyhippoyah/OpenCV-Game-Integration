import { describe, expect, it } from 'vitest';
import { MAP_H, MAP_W, pauses, pointAt, Rail, ROUTE, WALK_SPEED } from './path';
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

  it('can skip ahead to just short of a pause, and then walks into it', () => {
    const p = pauses(), r = new Rail(p[0].at + 0.01);
    r.skipTo(p[1].at);
    expect(r.d).toBeLessThan(p[1].at);
    expect(r.advance(0.01, p)).toEqual(p[1]);
  });

  it('draws the path on the map through every route point, inside the map', () => {
    for (const r of ROUTE) {
      const p = pointAt(r.d);
      expect(p.x).toBeCloseTo(r.x);
      expect(p.y).toBeCloseTo(r.y);
    }
    for (let d = 0; d <= 300; d += 5) {
      const p = pointAt(d);
      expect(p.x).toBeGreaterThan(0); expect(p.x).toBeLessThan(MAP_W);
      expect(p.y).toBeGreaterThan(0); expect(p.y).toBeLessThan(MAP_H);
    }
  });
});
