import { describe, expect, it } from 'vitest';
import { Game } from '../game/game';
import { mulberry32 } from '../math';
import { botFight } from '../test/bot';
import { movesFor, STOPS } from './chapter1';
import type { ScrollId } from './progress';

function play(stopIndex: number, smart: boolean): 'won' | 'lost' {
  const s = STOPS[stopIndex];
  const scrolls = STOPS.slice(0, stopIndex + 1).flatMap(x => (x.scroll ? [x.scroll] : [])) as ScrollId[];
  const moves = movesFor(scrolls);
  const g = new Game(mulberry32(stopIndex + 3), 70, true);
  g.scripted(); g.noDamage = false; g.allowed = moves;
  if (s.fight.boss) g.addBoss(0, 9);
  return botFight(g, s.fight, moves, smart);
}

describe('Chapter 1 fights', () => {
  STOPS.forEach((s, i) => {
    it(`${s.place} can be won with only the moves you have by then`, () => {
      expect(play(i, true)).toBe('won');
    });
  });

  it('standing still loses where the fight is meant to make you move', () => {
    for (const id of ['bridge', 'gate', 'daro']) {
      expect(play(STOPS.findIndex(s => s.id === id), false), id).toBe('lost');
    }
  });
});
