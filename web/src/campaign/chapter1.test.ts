import { describe, expect, it } from 'vitest';
import { LESSONS } from '../game/tutorial';
import { EPILOGUE, movesFor, PATH_LENGTH, SCROLLS, START_MOVES, STOPS } from './chapter1';

describe('Chapter 1', () => {
  it('has six stops down the path, in order, ending with the boss', () => {
    expect(STOPS.map(s => s.id)).toEqual(['courtyard', 'stairs', 'bridge', 'garden', 'gate', 'daro']);
    const at = STOPS.map(s => s.pathAt);
    expect([...at].sort((a, b) => a - b)).toEqual(at);
    expect(at.at(-1)).toBeLessThanOrEqual(PATH_LENGTH);
    expect(STOPS.at(-1)!.fight.boss).toBe(true);
  });

  it('every scroll is found once, just before the fight that needs it', () => {
    const found = STOPS.flatMap(s => (s.scroll ? [s.scroll] : [])).concat(STOPS.flatMap(s => (s.reward ? [s.reward] : [])));
    expect([...found].sort()).toEqual(Object.keys(SCROLLS).sort());
    for (const s of STOPS) if (s.scroll) expect(s.scrollAt!).toBeLessThan(s.pathAt);
  });

  it('practices and scrolls point at real lessons', () => {
    const ids = new Set(LESSONS.map(l => l.id));
    for (const s of STOPS) for (const p of s.practice) expect(ids.has(p), p).toBe(true);
    for (const sc of Object.values(SCROLLS)) expect(ids.has(sc.lessonId), sc.lessonId).toBe(true);
    expect(ids.has(EPILOGUE.lessonId)).toBe(true);
  });

  it('you start with punches and flurries; scrolls add the rest; never the X block or counters', () => {
    expect(START_MOVES).toEqual(['punch', 'flurry']);
    const all = movesFor(Object.keys(SCROLLS) as (keyof typeof SCROLLS)[]);
    for (const m of ['shield', 'palm', 'charge', 'wall', 'finisher'] as const) expect(all.has(m)).toBe(true);
    for (const m of ['xBlock', 'counter', 'oneTwo', 'volley', 'wallBreaker'] as const) expect(all.has(m)).toBe(false);
  });

  it('each stop only practises moves you have by then', () => {
    const have: string[] = [];
    for (const s of STOPS) {
      if (s.scroll) have.push(s.scroll);
      const moves = movesFor(have as never);
      for (const p of s.practice) {
        const needs = Object.values(SCROLLS).find(sc => sc.lessonId === p);
        if (needs) expect(have.includes(needs.id), `${s.id} practises ${p}`).toBe(true);
      }
      expect(s.newMove === null || moves.has(s.newMove)).toBe(true);
    }
  });
});
