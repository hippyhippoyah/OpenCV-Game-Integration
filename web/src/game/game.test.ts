import { describe, expect, it } from 'vitest';
import { bodyHit, Game, TUNE, type Proj } from './game';
import type { HandsIntent, Intent } from '../intent/interpret';
import { mulberry32 } from '../math';

const hands = (cx: number, cy: number, spread: number): HandsIntent => ({
  l: { x: cx - spread / 2, y: cy }, r: { x: cx + spread / 2, y: cy },
  center: { x: cx, y: cy }, spread, vel: { x: 0, y: 0 },
});
const intent = (o: Partial<Intent> = {}): Intent =>
  ({ present: true, head: { x: 0, y: 0 }, hands: null, raised: false, throwNow: false, ...o });
const ready = () => intent({ hands: hands(0, 20, 5), raised: true });
const incoming = (x: number, y: number, id = 999): Proj =>
  ({ id, kind: 'enemy', x, y, z: 0.4, vx: 0, vy: 0, vz: -5, r: TUNE.enemyProjRadius, resolved: false });

function quietGame(): Game {
  const g = new Game(mulberry32(1));
  g.spawning = false;
  return g;
}
function run(g: Game, seconds: number, i: Intent): void {
  for (let t = 0; t < seconds; t += 1 / 60) g.step(1 / 60, i);
}

describe('Game', () => {
  it('summons fire when raised hands come together', () => {
    const g = quietGame();
    g.step(1 / 60, ready());
    expect(g.fire.held).toBe(true);
    expect(g.drainEvents().some(e => e.type === 'summon')).toBe(true);
  });

  it('does not summon with hands down, and drops fire when hands go low', () => {
    const g = quietGame();
    g.step(1 / 60, intent({ hands: hands(0, 50, 5), raised: false }));
    expect(g.fire.held).toBe(false);
    g.step(1 / 60, ready());
    g.step(1 / 60, intent({ hands: hands(0, 50, 5), raised: false }));
    expect(g.fire.held).toBe(false);
  });

  it('throws a fireball, then waits for the cooldown before re-summoning', () => {
    const g = quietGame();
    g.step(1 / 60, ready());
    g.step(1 / 60, { ...ready(), throwNow: true });
    expect(g.projs.filter(p => p.kind === 'player')).toHaveLength(1);
    expect(g.fire.held).toBe(false);
    run(g, 0.3, ready());
    expect(g.fire.held).toBe(false);
    run(g, 0.3, ready());
    expect(g.fire.held).toBe(true);
  });

  it('spreading hands turns fire into a shield that drains, breaks and recovers', () => {
    const g = quietGame();
    g.step(1 / 60, ready());
    const wide = intent({ hands: hands(0, 20, 30), raised: true });
    g.step(1 / 60, wide);
    expect(g.shield.on).toBe(true);
    expect(g.fire.held).toBe(false);
    run(g, 3.2, wide);
    expect(g.shield.on).toBe(false);
    expect(g.drainEvents().some(e => e.type === 'shieldBroken')).toBe(true);
    run(g, 1.5, intent());
    expect(g.shield.energy).toBeGreaterThan(0.2);
  });

  it('an attack at your face hurts', () => {
    const g = quietGame();
    g.projs.push(incoming(0, 0));
    run(g, 0.2, intent());
    expect(g.hp).toBe(TUNE.maxHp - TUNE.hitDamage);
  });

  it('leaning out of the way dodges it', () => {
    const g = quietGame();
    g.projs.push(incoming(0, 0));
    run(g, 0.2, intent({ head: { x: 25, y: 0 } }));
    expect(g.hp).toBe(TUNE.maxHp);
    expect(g.drainEvents().some(e => e.type === 'dodged')).toBe(true);
  });

  it('a shield over the attack blocks it', () => {
    const g = quietGame();
    g.projs.push(incoming(0, 0));
    run(g, 0.2, intent({ hands: hands(0, 0, 30), raised: true }));
    expect(g.hp).toBe(TUNE.maxHp);
    expect(g.drainEvents().some(e => e.type === 'blocked')).toBe(true);
  });

  it('fireballs home in on spirits; two hits banish one', () => {
    const g = quietGame();
    g.enemies.push({ id: 50, x: 0, y: 33, z: 4, hp: 2, t: 0, appear: 1, dying: 0, flash: 0,
      cd: 99, winding: false, wind: 0, side: 1, phase: 0 });
    for (let n = 0; n < 2; n++) {
      run(g, 0.6, ready());
      g.step(1 / 60, { ...ready(), throwNow: true });
      run(g, 0.6, ready());
    }
    expect(g.drainEvents().some(e => e.type === 'killEnemy')).toBe(true);
    expect(g.score).toBeGreaterThanOrEqual(100);
  });

  it('ends the game when health runs out', () => {
    const g = quietGame();
    for (let i = 0; i < 8; i++) {
      g.projs.push(incoming(0, 0, 1000 + i));
      run(g, 0.6, intent());
    }
    expect(g.state).toBe('over');
    expect(g.hp).toBe(0);
  });

  it('announces and spawns the first wave', () => {
    const g = new Game(mulberry32(2));
    expect(g.drainEvents()).toContainEqual({ type: 'wave', wave: 1 });
    run(g, 3, intent());
    expect(g.enemies.length).toBeGreaterThan(0);
  });

  it('flags incoming attacks that would hit if you stay still', () => {
    const g = quietGame();
    const p = incoming(0, 0);
    g.projs.push(p);
    expect(g.isThreat(p)).toBe(true);
    g.step(1 / 60, intent({ head: { x: 25, y: 0 } }));
    expect(g.isThreat(p)).toBe(false);
  });

  describe('practice mode', () => {
    it('sets up still dummies that never attack', () => {
      const g = new Game(mulberry32(3), 70, true);
      expect(g.drainEvents().some(e => e.type === 'wave')).toBe(false);
      const x0 = g.enemies.map(e => e.x);
      run(g, 10, intent());
      expect(g.enemies).toHaveLength(3);
      expect(g.enemies.every(e => e.dummy)).toBe(true);
      expect(g.enemies.map(e => e.x)).toEqual(x0);
      expect(g.projs).toHaveLength(0);
      expect(g.hp).toBe(TUNE.maxHp);
    });

    it('brings a knocked-down dummy back', () => {
      const g = new Game(mulberry32(3), 70, true);
      const d = g.enemies[0];
      d.hp = 0;
      run(g, 0.6, intent());
      expect(g.enemies.find(e => e.id === d.id)).toBeUndefined();
      run(g, 2, intent());
      expect(g.enemies).toHaveLength(3);
    });

    it('toggles between dummies and spirit waves', () => {
      const g = new Game(mulberry32(3));
      g.drainEvents();
      g.setPractice(true);
      expect(g.practice).toBe(true);
      expect(g.enemies.every(e => e.dummy)).toBe(true);
      g.setPractice(false);
      expect(g.enemies).toHaveLength(0);
      expect(g.drainEvents()).toContainEqual({ type: 'wave', wave: 1 });
    });
  });

  it('bodyHit covers head and torso only', () => {
    expect(bodyHit({ x: 0, y: 0 }, 4)).toBe(true);
    expect(bodyHit({ x: 0, y: 30 }, 4)).toBe(true);
    expect(bodyHit({ x: 30, y: 0 }, 4)).toBe(false);
  });
});
