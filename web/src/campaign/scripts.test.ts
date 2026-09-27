import { describe, expect, it } from 'vitest';
import { Game } from '../game/game';
import { mulberry32 } from '../math';
import { FightRunner, type FightScript } from './scripts';

const fresh = () => { const g = new Game(mulberry32(1), 70, true); g.scripted(); g.noDamage = false; return g; };
const tick = (g: Game, f: FightRunner, seconds: number) => {
  let out = f.update(0);
  for (let t = 0; t < seconds && out === 'fighting'; t += 1 / 60) { g.step(1 / 60, idle); g.drainEvents(); out = f.update(1 / 60); }
  return out;
};
// a standing player who does nothing
const idle = { present: true, head: { x: 0, y: 0 }, hands: { l: null, r: null }, shoulders: null, punches: [], palms: [], shield: false, xBlock: false, casts: [], face: null, bodyTilt: 0 };

describe('FightRunner', () => {
  const two: FightScript = { groups: [{ at: 0, enemies: [{ kind: 'dummy', x: 0, z: 7 }] }, { at: 2, enemies: [{ kind: 'dummy', x: 10, z: 8 }] }], goal: { type: 'defeat' } };

  it('sends groups in at their times', () => {
    const g = fresh(), f = new FightRunner(g, two);
    tick(g, f, 1);
    expect(g.enemies).toHaveLength(1);
    tick(g, f, 1.5);
    expect(g.enemies).toHaveLength(2);
  });

  it('defeat: won once every group has come and fallen', () => {
    const g = fresh(), f = new FightRunner(g, two);
    tick(g, f, 1);
    g.enemies.forEach(e => { e.hp = 0; });
    expect(tick(g, f, 0.5)).toBe('fighting'); // the second group hasn't come yet
    tick(g, f, 1.5);
    g.enemies.forEach(e => { e.hp = 0; });
    expect(tick(g, f, 0.1)).toBe('won');
  });

  it('survive: won after the time, whatever is left', () => {
    const g = fresh(), f = new FightRunner(g, { groups: [{ at: 0, enemies: [{ kind: 'dummy', x: 0, z: 7 }] }], goal: { type: 'survive', seconds: 3 } });
    expect(tick(g, f, 2)).toBe('fighting');
    expect(tick(g, f, 1.5)).toBe('won');
    expect(f.progress).toBe(1);
  });

  it('lost when your health runs out', () => {
    const g = fresh(), f = new FightRunner(g, two);
    tick(g, f, 0.2);
    g.hp = 0;
    g.state = 'over';
    expect(f.update(1 / 60)).toBe('lost');
  });
});
