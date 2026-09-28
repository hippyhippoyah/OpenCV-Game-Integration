import { describe, expect, it } from 'vitest';
import { Game } from '../game/game';
import { mulberry32 } from '../math';
import { botFight } from '../test/bot';
import { movesFor, SCROLLS, STOPS } from './chapter1';
import { Progress, type ScrollId } from './progress';
import { RAIDS } from './raids';
import { BUILDINGS, templeEffects, upgrade, type TempleEffects } from './temple';

const MOVES = movesFor(Object.keys(SCROLLS) as ScrollId[]);
const SEEDS = [1, 2, 3];

/** The temple as a three-flame campaign can build it: everything at level 3. */
function maxTemple(): TempleEffects {
  const p = Progress.load(null);
  for (const s of STOPS) p.completeStop(s.id, 3);
  p.finishChapter();
  for (const b of BUILDINGS) while (upgrade(p, b.id)) { /* max it */ }
  return templeEffects(p);
}

function raid(i: number, temple: TempleEffects | null, bot: boolean | 'attack', seed: number) {
  const g = new Game(mulberry32(seed), 70, true);
  g.scripted(); g.noDamage = false; g.allowed = MOVES; g.temple = temple;
  return botFight(g, RAIDS[i].fight, MOVES, bot);
}

describe('the raid ladder', () => {
  it('has six raids with their own ids', () => {
    expect(RAIDS).toHaveLength(6);
    expect(new Set(RAIDS.map(r => r.id)).size).toBe(6);
  });

  it('a player who only attacks wins every raid with a maxed temple', () => {
    const t = maxTemple();
    RAIDS.forEach((r, i) => { for (const seed of SEEDS) expect(raid(i, t, 'attack', seed), `${r.name} seed ${seed}`).toBe('won'); });
  });

  it('without the temple, the later raids beat a player who only attacks', () => {
    const lost = SEEDS.filter(seed => raid(4, null, 'attack', seed) === 'lost');
    expect(lost.length).toBeGreaterThanOrEqual(1);
  });

  it('the temple alone never wins the hard raids: you still have to fight', () => {
    const t = maxTemple();
    for (const i of [3, 4, 5]) for (const seed of SEEDS) expect(raid(i, t, false, seed), `${RAIDS[i].name}`).toBe('lost');
  });

  it('standing still with no temple loses every raid', () => {
    RAIDS.forEach((r, i) => expect(raid(i, null, false, 1), r.name).toBe('lost'));
  });

  it('a player who dodges and shields well can win every raid without a temple', () => {
    RAIDS.forEach((r, i) => expect(raid(i, null, true, 1), r.name).toBe('won'));
  });
});
