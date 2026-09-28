import { describe, expect, it } from 'vitest';
import { STOPS } from './chapter1';
import { Progress } from './progress';
import {
  BUILDINGS, CHAPTER_BONUS, EMBERS_PER_FLAME, embersAvailable, embersEarned, embersSpent, MAX_LEVEL,
  nextCost, TEMPLE_MAX_COST, templeEffects, templeOpen, upgrade,
} from './temple';

/** Every stop finished with `flames`, and the chapter done if asked. */
function campaign(flames: number, done = true): Progress {
  const p = Progress.load(null);
  for (const s of STOPS) p.completeStop(s.id, flames);
  if (done) p.finishChapter();
  return p;
}

describe('embers', () => {
  it('a three-flame campaign pays exactly what maxing the temple costs: 10,000', () => {
    expect(TEMPLE_MAX_COST).toBe(10000);
    expect(embersEarned(campaign(3))).toBe(10000);
    expect(STOPS.length * 3 * EMBERS_PER_FLAME + CHAPTER_BONUS).toBe(10000);
  });

  it('each flame pays once: replaying for a better result pays only the new flames', () => {
    const p = Progress.load(null);
    p.completeStop('bridge', 2);
    expect(embersEarned(p)).toBe(800);
    p.completeStop('bridge', 1); // a worse replay keeps the best
    expect(embersEarned(p)).toBe(800);
    p.completeStop('bridge', 3);
    expect(embersEarned(p)).toBe(1200);
  });

  it('building spends embers, and the balance is worked out, never stored', () => {
    const p = campaign(1);
    const earned = embersEarned(p);
    expect(upgrade(p, 'shrine')).toBe(true);
    expect(embersSpent(p)).toBe(300);
    expect(embersAvailable(p)).toBe(earned - 300);
    const saved = JSON.parse(JSON.stringify(p.data));
    expect(JSON.stringify(saved)).not.toContain('ember');
  });

  it("can't build what you can't afford, or past level 3", () => {
    const p = campaign(0, false);
    expect(embersAvailable(p)).toBe(0);
    expect(upgrade(p, 'wall')).toBe(false);
    const rich = campaign(3);
    for (let i = 0; i < 5; i++) upgrade(rich, 'brazierL');
    expect(rich.level('brazierL')).toBe(MAX_LEVEL);
    expect(nextCost(rich, 'brazierL')).toBeNull();
  });

  it('a full campaign buys every level of every building, with nothing left over', () => {
    const p = campaign(3);
    for (const b of BUILDINGS) while (upgrade(p, b.id)) { /* keep building */ }
    expect(BUILDINGS.every(b => p.level(b.id) === MAX_LEVEL)).toBe(true);
    expect(embersAvailable(p)).toBe(0);
  });

  it('opens once Chapter 1 is finished', () => {
    expect(templeOpen(campaign(3, false))).toBe(false);
    expect(templeOpen(campaign(1))).toBe(true);
  });

  it('resetting the campaign resets the temple too', () => {
    const p = campaign(3);
    upgrade(p, 'wall');
    p.reset();
    expect(p.level('wall')).toBe(0);
  });
});

describe('temple effects', () => {
  it('nothing built: no help', () => {
    expect(templeEffects(campaign(3))).toEqual({ braziers: [], blockChance: 0, damageMult: 1 });
  });

  it('each level helps more', () => {
    const p = campaign(3);
    upgrade(p, 'brazierL');
    upgrade(p, 'wall');
    upgrade(p, 'shrine');
    const one = templeEffects(p);
    expect(one.braziers).toHaveLength(1);
    for (const b of BUILDINGS) while (upgrade(p, b.id)) { /* max */ }
    const max = templeEffects(p);
    expect(max.braziers).toHaveLength(2);
    expect(max.braziers[0].cd).toBeLessThan(one.braziers[0].cd);
    expect(max.blockChance).toBeGreaterThan(one.blockChance);
    expect(max.damageMult).toBeLessThan(one.damageMult);
  });
});
