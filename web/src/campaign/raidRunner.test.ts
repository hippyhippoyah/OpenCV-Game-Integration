import { describe, expect, it } from 'vitest';
import { Game } from '../game/game';
import { mulberry32 } from '../math';
import { STOPS } from './chapter1';
import { Progress } from './progress';
import { raidOpen, RaidRunner } from './raidRunner';
import { RAIDS } from './raids';
import { upgrade } from './temple';

const done = () => { const p = Progress.load(null); for (const s of STOPS) p.completeStop(s.id, 3); p.finishChapter(); return p; };
const make = () => new Game(mulberry32(1), 70, true);
const until = (r: RaidRunner, state: string, ready = true) => { for (let t = 0; t < 30 && r.state !== state; t += 1 / 30) r.update(1 / 30, ready); return r.state; };

describe('RaidRunner', () => {
  it('waits for you to step back, counts down, then fights with the temple as built', () => {
    const p = done();
    upgrade(p, 'brazierL');
    const r = new RaidRunner(p, 0, make);
    expect(until(r, 'countdown', false)).toBe('handoff');
    expect(until(r, 'fight')).toBe('fight');
    expect(r.game!.temple!.braziers).toHaveLength(1);
    expect(r.game!.noDamage).toBe(false);
    expect(r.game!.label).toBe(RAIDS[0].name);
  });

  it('winning saves the raid with its flames and opens the next', () => {
    const p = done(), r = new RaidRunner(p, 0, make);
    until(r, 'fight');
    expect(raidOpen(p, 1)).toBe(false);
    r.fight!.elapsed = 99;
    for (let i = 0; i < 600 && r.state === 'fight'; i++) { r.game!.enemies.forEach(e => { e.hp = 0; }); r.update(1 / 30, true); }
    expect(r.state).toBe('result');
    expect(r.result!.flames).toBe(3);
    expect(p.raidFlames(RAIDS[0].id)).toBe(3);
    expect(raidOpen(p, 1)).toBe(true);
  });

  it('losing costs nothing: try again', () => {
    const p = done(), r = new RaidRunner(p, 0, make);
    until(r, 'fight');
    r.game!.state = 'over';
    r.update(1 / 30, true);
    expect(r.state).toBe('lost');
    r.retry();
    expect(r.state).toBe('fight');
    expect(p.raidDone(RAIDS[0].id)).toBe(false);
  });
});
