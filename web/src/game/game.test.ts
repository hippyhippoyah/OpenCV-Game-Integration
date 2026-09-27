import { describe, expect, it } from 'vitest';
import { bodyHit, Game, TUNE, type Proj } from './game';
import type { HandState, Intent, Punch, Side } from '../intent/interpret';
import { mulberry32 } from '../math';

const SHOULDERS = { l: { x: -20, y: 20 }, r: { x: 20, y: 20 } };
const hs = (x: number, y: number, open = false): HandState =>
  ({ pos: { x, y }, vel: { x: 0, y: 0 }, openness: open ? 1 : 0, open, facing: 1, source: 'hand', inView: true, elbow: null, extension: null, punchReady: true });
const guard = () => ({ l: hs(-12, 22), r: hs(12, 22) });
const intent = (o: Partial<Intent> = {}): Intent =>
  ({ present: true, head: { x: 0, y: 0 }, hands: guard(), shoulders: SHOULDERS, punches: [], shield: false, face: null, bodyTilt: 0, ...o });
const punch = (hand: Side, x: number, y: number, dir: Punch['dir'] = null): Punch => ({ hand, at: { x, y }, shoulder: SHOULDERS[hand], dir });
const shieldUp = (y = 0) => intent({ hands: { l: hs(-15, y, true), r: hs(15, y, true) }, shield: true });
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
  describe('punch', () => {
    it('launches a fireball from the hand that opened', () => {
      const g = quietGame();
      g.step(1 / 60, intent({ punches: [punch('r', 10, 5)] }));
      expect(g.projs).toHaveLength(1);
      expect(g.projs[0].kind).toBe('player');
      expect(g.drainEvents()).toContainEqual({ type: 'punch', x: 10, y: 5, z: TUNE.launchZ });
    });

    it('each hand has its own short cooldown', () => {
      const g = quietGame();
      g.step(1 / 60, intent({ punches: [punch('r', 10, 5), punch('r', 10, 5), punch('l', -10, 5)] }));
      g.step(1 / 60, intent({ punches: [punch('r', 10, 5)] }));
      expect(g.projs).toHaveLength(2);
      run(g, TUNE.punchCooldownS, intent());
      g.step(1 / 60, intent({ punches: [punch('r', 10, 5)] }));
      expect(g.projs).toHaveLength(3);
    });

    it('goes the way you punch when nothing is near', () => {
      const g = quietGame();
      g.step(1 / 60, intent({ punches: [punch('l', -30, 0), punch('r', 30, 0)] }));
      const [left, right] = [...g.projs].sort((a, b) => a.x - b.x);
      expect(left.vx).toBeLessThan(0);
      expect(right.vx).toBeGreaterThan(0);
    });

    it('steers by the 3D direction of the arm when known', () => {
      const g = quietGame();
      g.step(1 / 60, intent({ punches: [punch('r', 20, 20, { x: -0.8, y: 0 })] }));
      expect(g.projs[0].vx).toBeLessThan(0); // punched across to the left from in front of the right shoulder
    });

    it('snaps onto a target near where you punch and knocks it down', () => {
      const g = new Game(mulberry32(3), 70, true);
      g.drainEvents();
      for (let n = 0; n < 2; n++) {
        g.step(1 / 60, intent({ punches: [punch('r', 0, 8)] }));
        run(g, 1, intent());
      }
      expect(g.drainEvents().some(e => e.type === 'killEnemy')).toBe(true);
    });
  });

  describe('shield', () => {
    it('is up while both hands are open', () => {
      const g = quietGame();
      g.step(1 / 60, shieldUp());
      expect(g.shield.on).toBe(true);
      g.step(1 / 60, intent());
      expect(g.shield.on).toBe(false);
    });

    it('needs both hands in view', () => {
      const g = quietGame();
      g.step(1 / 60, intent({ hands: { l: hs(-15, 0, true), r: null }, shield: true }));
      expect(g.shield.on).toBe(false);
    });

    it('never runs out while testing', () => {
      const g = quietGame();
      run(g, 10, shieldUp());
      expect(g.shield.on).toBe(true);
      expect(g.shield.energy).toBe(1);
    });

    it('blocks an attack it covers', () => {
      const g = quietGame();
      g.projs.push(incoming(0, 0));
      run(g, 0.2, shieldUp(0));
      expect(g.hp).toBe(TUNE.maxHp);
      expect(g.drainEvents().some(e => e.type === 'blocked')).toBe(true);
    });
  });

  describe('getting hit', () => {
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

    it('flags incoming attacks that would hit if you stay still', () => {
      const g = quietGame();
      const p = incoming(0, 0);
      g.projs.push(p);
      expect(g.isThreat(p)).toBe(true);
      g.step(1 / 60, intent({ head: { x: 25, y: 0 } }));
      expect(g.isThreat(p)).toBe(false);
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
  });

  it('announces and spawns the first wave', () => {
    const g = new Game(mulberry32(2));
    expect(g.drainEvents()).toContainEqual({ type: 'wave', wave: 1 });
    run(g, 3, intent());
    expect(g.enemies.length).toBeGreaterThan(0);
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
