import { describe, expect, it } from 'vitest';
import { Game, type Enemy } from '../game/game';
import { mulberry32 } from '../math';
import { movesFor, STOPS } from './chapter1';
import type { ScrollId } from './progress';
import { FightRunner } from './scripts';
import type { HandState, Intent } from '../intent/interpret';

const base: Intent = { present: true, head: { x: 0, y: 0 }, hands: { l: null, r: null }, shoulders: { l: { x: -20, y: 20 }, r: { x: 20, y: 20 } }, punches: [], palms: [], shield: false, xBlock: false, casts: [], face: null, bodyTilt: 0 };

/** Where an enemy appears on screen (view units), to aim at it. */
const onScreen = (g: Game, e: Enemy) => { const s = 3 / (3 + e.z); return { x: (e.x - g.cam.x) * s, y: (e.y - g.cam.y) * s }; };

/** A raised, open hand for the shield: only the fields the shield check reads matter. */
const shieldHand = (x: number, y: number): HandState =>
  ({ pos: { x, y }, vel: { x: 0, y: 0 }, openness: 1, open: true, facing: 1, palm: null, source: 'hand', inView: true, elbow: null, extension: null, punchReady: true, punchRise: null, reach: null, reachBase: null, reachNoise: null, aimDir: null, charge: 0 });

function play(stopIndex: number, smart: boolean): 'won' | 'lost' {
  const s = STOPS[stopIndex];
  const scrolls = STOPS.slice(0, stopIndex + 1).flatMap(x => (x.scroll ? [x.scroll] : [])) as ScrollId[];
  const moves = movesFor(scrolls);
  const g = new Game(mulberry32(stopIndex + 3), 70, true);
  g.scripted(); g.noDamage = false; g.allowed = moves;
  const f = new FightRunner(g, s.fight);
  if (s.fight.boss) g.addBoss(0, 9);
  let head = { x: 0, y: 0 };
  for (let t = 0; t < 240; t += 1 / 60) {
    const i: Intent = { ...base, head, punches: [], palms: [], casts: [] };
    if (smart) {
      const next = g.incoming()[0];
      head = !next || next.safe ? head : next.kind === 'slab' ? { x: head.x, y: 20 } : { x: next.away * 30, y: 0 };
      if (!next) head = { x: head.x * 0.98, y: head.y * 0.9 };
      const target = g.enemies.find(e => e.hp > 0 && (e.boss ? true : !e.dummy || true));
      const frame = Math.round(t * 60);
      if (target && frame % 30 === 0) {
        const at = onScreen(g, target);
        const charged = moves.has('charge') && (target.hp >= 3 || !!target.boss) && frame % 60 === 0;
        i.punches = [{ hand: frame % 60 ? 'l' : 'r', at, shoulder: { x: 20, y: 20 }, dir: null, charged }];
      }
      if (moves.has('wall') && frame % 300 === 0) i.casts = [{ kind: 'wall', at: { x: 0, y: 10 } }];
      if (moves.has('palm') && target && frame % 90 === 45) i.palms = [{ kind: 'push', hand: 'r', at: onScreen(g, target), shoulder: { x: 20, y: 20 }, dir: null }];
      // Raise the shield when an orb is about to arrive: it needs both hands, and the bot's punches
      // don't use `hands` for anything, so this is free to hold up whenever it's useful.
      if (moves.has('shield')) {
        const soon = g.projs.some(p => p.kind === 'enemy' && p.z > 0 && p.z / -p.vz < 0.35);
        if (soon) { i.shield = true; i.hands = { l: shieldHand(head.x - 15, head.y), r: shieldHand(head.x + 15, head.y) }; }
      }
    }
    g.step(1 / 60, i);
    g.drainEvents();
    const out = s.fight.boss ? (g.state === 'over' ? 'lost' : g.boss ? 'fighting' : 'won') : f.update(1 / 60);
    if (out !== 'fighting') return out;
  }
  return 'lost';
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
