import { describe, expect, it } from 'vitest';
import { bodyHit, FLOOR_Y, Game, TUNE, type Enemy, type Proj } from './game';
import type { HandState, Intent, Palm, Punch, Side } from '../intent/interpret';
import { mulberry32 } from '../math';

const SHOULDERS = { l: { x: -20, y: 20 }, r: { x: 20, y: 20 } };
const hs = (x: number, y: number, open = false): HandState =>
  ({ pos: { x, y }, vel: { x: 0, y: 0 }, openness: open ? 1 : 0, open, facing: 1, source: 'hand', inView: true, elbow: null, extension: null, punchReady: true, punchRise: null, reach: null, reachBase: null, reachNoise: null, aimDir: null, charge: 0 });
const guard = () => ({ l: hs(-12, 22), r: hs(12, 22) });
const intent = (o: Partial<Intent> = {}): Intent =>
  ({ present: true, head: { x: 0, y: 0 }, hands: guard(), shoulders: SHOULDERS, punches: [], palms: [], shield: false, xBlock: false, casts: [], face: null, bodyTilt: 0, ...o });
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
      expect(g.drainEvents()).toContainEqual({ type: 'punch', x: 10, y: 5, z: TUNE.launchZ, side: 'r' });
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

  describe('palm push', () => {
    const palm = (kind: Palm['kind'], x = 0, y = 10, hand: Side = 'r'): Palm => ({ kind, hand, at: { x, y }, shoulder: SHOULDERS[hand], dir: null });
    const foe = (id: number, x: number, z: number, hp = 3) =>
      ({ id, x, y: FLOOR_Y - 30, z, hp, t: 0, appear: 1, dying: 0, flash: 0, cd: 99, winding: false, wind: 0, side: 1 as const, phase: 0 });

    it('a push sends a pillar rolling forward that hits hard and passes through everything in its way', () => {
      const g = quietGame();
      g.enemies.push(foe(1, 0, 5), foe(2, 4, 9), foe(3, 60, 7));
      g.step(1 / 60, intent({ palms: [palm('push')] }));
      expect(g.pillars).toHaveLength(1);
      expect(g.drainEvents().some(e => e.type === 'pillar')).toBe(true);
      run(g, 2, intent());
      const hp = (id: number) => g.enemies.find(e => e.id === id)?.hp ?? 0;
      expect(hp(1)).toBe(3 - TUNE.palmDamage);
      expect(hp(2)).toBe(3 - TUNE.palmDamage);
      expect(hp(3)).toBe(3); // well off to the side
      expect(g.pillars).toHaveLength(0); // rolled off the end of the field
    });

    it('a push burns through incoming attacks', () => {
      const g = quietGame();
      g.projs.push({ ...incoming(0, 20), z: 4, vz: -2 });
      g.step(1 / 60, intent({ palms: [palm('push')] }));
      run(g, 1, intent());
      expect(g.projs.filter(p => p.kind === 'enemy')).toHaveLength(0);
      expect(g.hp).toBe(TUNE.maxHp);
    });

    it('each hand rests between pushes', () => {
      const g = quietGame();
      g.step(1 / 60, intent({ palms: [palm('push'), palm('push')] }));
      expect(g.pillars).toHaveLength(1);
      run(g, TUNE.palmCooldownS, intent());
      g.step(1 / 60, intent({ palms: [palm('push')] }));
      expect(g.pillars).toHaveLength(2);
    });
  });

  describe('pushing both palms', () => {
    const push = (x = 0) => intent({ casts: [{ kind: 'push', at: { x, y: 10 } }] });
    const wall = intent({ casts: [{ kind: 'wall', at: { x: 0, y: 10 } }] });
    const foe = (id: number, x: number, z: number) =>
      ({ id, x, y: FLOOR_Y - 30, z, hp: 3, t: 0, appear: 1, dying: 0, flash: 0, cd: 99, winding: false, wind: 0, side: 1 as const, phase: 0 });

    it('does nothing on its own, and says why', () => {
      const g = quietGame();
      g.step(1 / 60, push());
      expect(g.walls).toHaveLength(0);
      expect(g.drainEvents()).toContainEqual({ type: 'hint', text: expect.stringContaining('FIRE WALL') });
    });

    it('wall breaker: sends your standing fire wall rolling forward, burning enemies and blocking attacks', () => {
      const g = quietGame();
      g.enemies.push(foe(1, -20, 6), foe(2, 20, 9), foe(3, 150, 6));
      g.step(1 / 60, wall);
      g.projs.push({ ...incoming(0, 10), z: 5, vz: -3 });
      g.step(1 / 60, push());
      expect(g.walls).toHaveLength(1);
      expect(g.walls[0].vz).toBeGreaterThan(0);
      expect(g.drainEvents()).toContainEqual(expect.objectContaining({ type: 'combo', name: 'wallBreaker' }));
      run(g, 3, intent());
      const hp = (id: number) => g.enemies.find(e => e.id === id)?.hp ?? 0;
      expect(hp(1)).toBe(3 - TUNE.palmDamage);
      expect(hp(2)).toBe(3 - TUNE.palmDamage);
      expect(hp(3)).toBe(3);
      expect(g.projs.filter(p => p.kind === 'enemy')).toHaveLength(0);
      expect(g.hp).toBe(TUNE.maxHp);
      expect(g.walls).toHaveLength(0); // rolled off the end of the field
    });

  });

  describe('combos', () => {
    const foe = (id: number, x: number, z: number, hp = 5) =>
      ({ id, x, y: FLOOR_Y - 30, z, hp, t: 0, appear: 1, dying: 0, flash: 0, cd: 99, winding: false, wind: 0, side: 1 as const, phase: 0 });
    const combos = (g: Game) => g.drainEvents().flatMap(e => (e.type === 'combo' ? [e.name] : []));
    const jab = (hand: Side) => intent({ punches: [punch(hand, 0, 8)] });

    it('a charged punch throws a bigger, faster blue fireball that hits twice as hard', () => {
      const g = quietGame();
      g.enemies.push(foe(1, 0, 7));
      g.step(1 / 60, intent({ punches: [{ ...punch('r', 0, 8), charged: true }] }));
      expect(g.projs[0].shot).toBe('charged');
      expect(g.projs[0].r).toBeGreaterThan(TUNE.fireballRadius);
      expect(g.projs[0].vz).toBeGreaterThan(TUNE.fireballSpeed);
      expect(combos(g)).toEqual(['charged']);
      run(g, 1.5, intent());
      expect(g.enemies[0].hp).toBe(5 - TUNE.chargedDamage);
    });

    it('a charged punch goes for an enemy even when the fist points somewhere odd', () => {
      const g = quietGame();
      g.enemies.push(foe(1, 40, 8));
      // the fist reads far off to the lower left, pointing sideways
      g.step(1 / 60, intent({ punches: [{ ...punch('r', -40, 45, { x: -2, y: 1.5 }), charged: true }] }));
      run(g, 1.5, intent());
      expect(g.enemies[0].hp).toBe(5 - TUNE.chargedDamage);
    });

    it('with nobody there, a charged punch flies straight ahead', () => {
      const g = quietGame();
      g.step(1 / 60, intent({ punches: [{ ...punch('r', -40, 45, { x: -2, y: 1.5 }), charged: true }] }));
      // heads for the middle of the view, not off to the lower left where the fist pointed
      const p = g.projs[0], t = (TUNE.aimDepth - p.z) / p.vz;
      expect(Math.abs(p.x + p.vx * t)).toBeLessThan(1);
      expect(p.y + p.vy * t).toBeLessThan(FLOOR_Y - 30);
    });

    it('flurry: the third quick punch is a big fireball that also burns those nearby', () => {
      const g = quietGame();
      g.enemies.push(foe(1, 0, 7), foe(2, 15, 7), foe(3, 90, 7));
      g.step(1 / 60, jab('r'));
      run(g, 0.2, intent());
      g.step(1 / 60, jab('l'));
      run(g, 0.2, intent());
      g.step(1 / 60, jab('r'));
      const shots = g.projs.filter(p => p.kind === 'player').map(p => p.shot);
      expect(shots.at(-1)).toBe('flurry');
      expect(combos(g)).toEqual(['flurry']);
      run(g, 2, intent());
      expect(g.enemies.find(e => e.id === 2)!.hp).toBeLessThan(5); // splashed
      expect(g.enemies.find(e => e.id === 3)!.hp).toBe(5);
    });

    it('slow punches are no flurry', () => {
      const g = quietGame();
      for (let i = 0; i < 3; i++) { g.step(1 / 60, jab(i % 2 ? 'l' : 'r')); run(g, 0.6, intent()); }
      expect(combos(g)).toEqual([]);
    });

    it('counter: a punch just after the shield blocks something homes in and hits hard', () => {
      const g = quietGame();
      g.enemies.push(foe(1, 60, 8));
      g.projs.push(incoming(0, 10));
      run(g, 0.2, shieldUp(10));
      g.drainEvents();
      g.step(1 / 60, intent({ punches: [punch('r', -30, 0)] })); // aimed well off to the left
      expect(combos(g)).toEqual(['counter']);
      run(g, 1.5, intent());
      expect(g.enemies[0].hp).toBe(5 - TUNE.counterDamage);
    });

    it('one-two push: two jabs then a palm push make a wide pillar', () => {
      const g = quietGame();
      g.step(1 / 60, jab('l'));
      run(g, 0.15, intent());
      g.step(1 / 60, jab('r'));
      run(g, 0.2, intent());
      g.step(1 / 60, intent({ palms: [{ kind: 'push', hand: 'r', at: { x: 0, y: 10 }, shoulder: SHOULDERS.r, dir: null }] }));
      expect(combos(g)).toEqual(['oneTwo']);
      expect(g.pillars[0].halfW).toBe(TUNE.pillarHalfW * TUNE.oneTwoWidth);
    });

    it('pillar volley: a palm push from each hand, quickly, merge into one wide wave', () => {
      const g = quietGame();
      const push = (hand: Side) => intent({ palms: [{ kind: 'push', hand, at: { x: 0, y: 10 }, shoulder: SHOULDERS[hand], dir: null }] });
      g.step(1 / 60, push('r'));
      run(g, 0.3, intent());
      g.step(1 / 60, push('l'));
      expect(g.pillars).toHaveLength(1);
      expect(g.pillars[0].halfW).toBe(TUNE.pillarHalfW * TUNE.volleyWidth);
      expect(combos(g)).toEqual(['volley']);
    });

    it('the ultimate is a finisher: without two jabs first it does nothing and says how', () => {
      const g = quietGame();
      g.step(1 / 60, intent({ casts: [{ kind: 'ultimate', at: { x: 0, y: 10 } }] }));
      expect(g.blades).toHaveLength(0);
      expect(g.ultimateCharge).toBe(1);
      expect(g.drainEvents()).toContainEqual({ type: 'hint', text: expect.stringContaining('JAB, JAB') });
      g.step(1 / 60, jab('l'));
      run(g, 0.3, intent());
      g.step(1 / 60, jab('r'));
      run(g, 0.5, intent());
      g.step(1 / 60, intent({ casts: [{ kind: 'ultimate', at: { x: 0, y: 10 } }] }));
      expect(g.blades).toHaveLength(1);
      expect(combos(g)).toContain('finisher');
    });
  });

  describe('attacks you have to move out of', () => {
    const at = (x: number, y = 0) => intent({ head: { x, y } });
    const foe = (o: Partial<Enemy> = {}): Enemy =>
      ({ id: 7, x: 10, y: FLOOR_Y - 30, z: 8, hp: 2, t: 0, appear: 1, dying: 0, flash: 0, cd: 0, winding: false, wind: 0, side: 1, phase: 0, ...o });
    /** An earthbender about to raise a pillar; returns once it is shoved at you. */
    function pillarComing(): Game {
      const g = quietGame();
      g.enemies.push(foe({ earth: true }));
      g.step(1 / 60, at(0));
      expect(g.hazards.map(h => h.kind)).toEqual(['stonePillar']);
      expect(g.hazards[0].vz).toBe(0); // rising in front of him
      run(g, TUNE.pillarWindupS * 0.5, at(0));
      expect(g.hazards[0].rise).toBeGreaterThan(0.5);
      run(g, TUNE.pillarWindupS * 0.5 + 0.05, at(0));
      expect(g.hazards[0].vz).toBeLessThan(0); // shoved
      expect(g.drainEvents().some(e => e.type === 'stonePillar')).toBe(true);
      return g;
    }

    it('an earthbender raises a stone pillar, then shoves it along a lane beside you: stay and it hits', () => {
      const g = pillarComing();
      const lane = g.hazards[0].laneX;
      expect(Math.abs(lane)).toBe(TUNE.pillarOffset);
      run(g, 4, at(0));
      expect(g.hp).toBe(TUNE.maxHp - TUNE.hitDamage);
    });

    it('lean or step away from the pillar side and it misses you', () => {
      const g = pillarComing();
      const away = -Math.sign(g.hazards[0].laneX) * (TUNE.stonePillarHalfW + TUNE.bodyHalfW - TUNE.pillarOffset + 3);
      run(g, 4, at(away));
      expect(g.hp).toBe(TUNE.maxHp);
      expect(g.drainEvents().some(e => e.type === 'dodged')).toBe(true);
    });

    it('it comes slowly enough to see and react to', () => {
      const g = pillarComing();
      let t = 0;
      while (g.hazards.length && !g.hazards[0].resolved) { g.step(1 / 60, at(0)); t += 1 / 60; }
      expect(t).toBeGreaterThan(1.8);
    });

    it('knocking the earthbender down while he raises it crumbles the pillar', () => {
      const g = quietGame();
      g.enemies.push(foe({ earth: true }));
      g.step(1 / 60, at(0));
      g.enemies[0].hp = 0;
      g.step(1 / 60, at(0));
      expect(g.hazards).toHaveLength(0);
    });

    it('earthbenders only ever raise pillars (no rocks)', () => {
      const g = quietGame();
      g.enemies.push(foe({ earth: true }));
      let pillars = 0;
      for (let t = 0; t < 40; t += 1 / 60) {
        g.step(1 / 60, at(0));
        g.hp = TUNE.maxHp;
        expect(g.projs.filter(p => p.kind === 'enemy')).toHaveLength(0);
        pillars += g.drainEvents().filter(e => e.type === 'stonePillar').length;
      }
      expect(pillars).toBeGreaterThan(3);
    });

    it('the pillar runs straight down a lane clearly to one side of you', () => {
      const g = pillarComing();
      const h = g.hazards[0];
      expect(Math.abs(h.x)).toBe(TUNE.pillarOffset);
      run(g, 1, at(0));
      expect(g.hazards[0].x).toBe(h.laneX);
      // its near edge only just reaches past your centre
      expect(Math.abs(h.laneX) - TUNE.stonePillarHalfW).toBeGreaterThan(0);
    });

    it('tells you, the whole way in, whether you are out of its way yet', () => {
      const g = pillarComing();
      const side = Math.sign(g.hazards[0].laneX);
      expect(g.incoming()).toEqual([expect.objectContaining({ kind: 'stonePillar', safe: false, away: -side })]);
      run(g, 0.3, at(-side * 15));
      expect(g.incoming()[0].safe).toBe(true);
      expect(g.incoming()[0].closeness).toBeGreaterThan(0);
    });

    it('the shield and X block do not stop a pillar; a fire wall does', () => {
      const shielded = pillarComing();
      run(shielded, 4, { ...shieldUp(10), xBlock: true });
      expect(shielded.hp).toBe(TUNE.maxHp - TUNE.hitDamage);
      const walled = pillarComing();
      walled.step(1 / 60, intent({ casts: [{ kind: 'wall', at: { x: 0, y: 10 } }] }));
      run(walled, 4, at(0));
      expect(walled.hp).toBe(TUNE.maxHp);
    });

    it('a high sweep comes at head height: stand and it hits, duck and it passes over', () => {
      const sweep = () => {
        const g = quietGame();
        g.enemies.push(foe({ attack: 'slab' }));
        run(g, TUNE.windupS + 0.05, at(0));
        expect(g.hazards.map(h => h.kind)).toEqual(['slab']);
        return g;
      };
      const standing = sweep();
      run(standing, 3, at(0));
      expect(standing.hp).toBe(TUNE.maxHp - TUNE.hitDamage);
      const ducking = sweep();
      run(ducking, 3, at(0, TUNE.slabDuck + 2));
      expect(ducking.hp).toBe(TUNE.maxHp);
    });

    it('a high sweep always comes at standing head height, even if you were ducking when it was sent', () => {
      const g = quietGame();
      g.enemies.push(foe({ attack: 'slab' }));
      run(g, TUNE.windupS + 0.05, at(0, 30)); // crouched low the whole time
      expect(g.hazards[0].y).toBe(TUNE.slabY);
      run(g, 3, at(0, TUNE.slabDuck + 2)); // a normal duck gets under it
      expect(g.hp).toBe(TUNE.maxHp);
    });

    it('waves bring spirits and earthbenders, and keep them all near the middle of the screen', () => {
      const g = new Game(mulberry32(3));
      const kinds = new Set<string>();
      let widest = 0;
      for (let t = 0; t < 120; t += 1 / 60) {
        g.step(1 / 60, intent());
        g.hp = TUNE.maxHp;
        for (const e of g.enemies) {
          kinds.add(e.earth ? 'earthbender' : 'spirit');
          widest = Math.max(widest, Math.abs(e.x * (3 / (3 + e.z))));
        }
        for (const h of g.hazards) kinds.add(h.kind);
        if (g.projs.some(p => p.kind === 'enemy')) kinds.add('orb');
      }
      expect([...kinds].sort()).toEqual(['earthbender', 'orb', 'slab', 'spirit', 'stonePillar']);
      expect(widest).toBeLessThanOrEqual(g.viewHalfW * TUNE.enemyBand + 0.01);
    });
  });

  describe('fire wall', () => {
    const wall = (x = 0) => intent({ casts: [{ kind: 'wall', at: { x, y: 10 } }] });

    it('rises in front of you where your hands are and blocks attacks', () => {
      const g = quietGame();
      g.step(1 / 60, wall());
      expect(g.walls).toHaveLength(1);
      expect(g.walls[0].z).toBe(TUNE.wallDepth);
      g.projs.push({ ...incoming(0, 0), z: TUNE.wallDepth + 0.5 });
      run(g, 0.5, intent());
      expect(g.hp).toBe(TUNE.maxHp);
      expect(g.drainEvents().some(e => e.type === 'blocked')).toBe(true);
    });

    it('only covers its own width', () => {
      const g = quietGame();
      g.step(1 / 60, wall(-60));
      g.projs.push({ ...incoming(0, 0), z: TUNE.wallDepth + 0.5 });
      run(g, 0.8, intent()); // long enough to reach you
      expect(g.hp).toBe(TUNE.maxHp - TUNE.hitDamage);
    });

    it('lets your own fireballs through', () => {
      const g = quietGame();
      g.step(1 / 60, wall());
      g.step(1 / 60, intent({ punches: [punch('r', 0, 0)] }));
      run(g, 0.4, intent());
      expect(g.projs[0].z).toBeGreaterThan(TUNE.wallDepth);
    });

    it('burns out after a few seconds and has a short cooldown', () => {
      const g = quietGame();
      g.step(1 / 60, wall());
      g.step(1 / 60, wall());
      expect(g.walls).toHaveLength(1);
      run(g, TUNE.wallLifeS, intent());
      expect(g.walls).toHaveLength(0);
    });
  });

  describe('ultimate', () => {
    // a finisher: two jabs, then gather & fling
    const ultimate = intent({ punches: [punch('l', -10, 5), punch('r', 10, 5)], casts: [{ kind: 'ultimate', at: { x: 0, y: 10 } }] });
    const enemyAt = (id: number, z: number, x = 0) =>
      ({ id, x, y: 33, z, hp: 2, t: 0, appear: 1, dying: 0, flash: 0, cd: 99, winding: false, wind: 0, side: 1 as const, phase: 0 });

    it('sends out a flat blade of fire, below the hands, that grows until it has swept the field', () => {
      const g = quietGame();
      g.step(1 / 60, ultimate);
      expect(g.blades).toHaveLength(1);
      expect(g.blades[0].y).toBeGreaterThan(10);
      expect(g.blades[0].y).toBeLessThan(FLOOR_Y);
      const r0 = g.blades[0].r;
      run(g, 0.2, intent());
      expect(g.blades[0].r).toBeGreaterThan(r0);
      run(g, 2, intent());
      expect(g.blades).toHaveLength(0);
      expect(g.drainEvents().some(e => e.type === 'ultimate')).toBe(true);
    });

    it('cuts down near enemies before far ones, then every enemy and incoming attack', () => {
      const g = quietGame();
      g.enemies.push(enemyAt(1, 4), enemyAt(2, 11, 150));
      g.projs.push({ ...incoming(0, 0), z: 8, vz: -1 });
      g.step(1 / 60, ultimate);
      run(g, 0.35, intent());
      expect(g.enemies.find(e => e.id === 1)!.hp).toBe(0);
      expect(g.enemies.find(e => e.id === 2)!.hp).toBeGreaterThan(0);
      run(g, 1, intent());
      expect(g.enemies.every(e => e.hp <= 0)).toBe(true);
      expect(g.projs.filter(p => p.kind === 'enemy')).toHaveLength(0);
      expect(g.score).toBeGreaterThanOrEqual(200);
      expect(g.hp).toBe(TUNE.maxHp);
    });

    it('has to recharge before it can be used again', () => {
      const g = new Game(mulberry32(3), 70, true);
      expect(g.ultimateCharge).toBe(1);
      g.step(1 / 60, ultimate);
      expect(g.ultimateCharge).toBeLessThan(0.01);
      run(g, 3, intent()); // dummies are cut down, then come back
      expect(g.enemies.every(e => e.hp > 0)).toBe(true);
      g.step(1 / 60, intent({ casts: ultimate.casts })); // still recharging: nothing
      run(g, 1, intent());
      expect(g.enemies.every(e => e.hp > 0)).toBe(true);
      run(g, TUNE.ultimateCooldownS, intent());
      expect(g.ultimateCharge).toBe(1);
      g.step(1 / 60, ultimate);
      run(g, 1, intent());
      expect(g.enemies.every(e => e.hp <= 0)).toBe(true);
    });
  });

  describe('X block', () => {
    const crossed = () => intent({ hands: { l: hs(6, 14), r: hs(-6, 14) }, xBlock: true });

    it('blocks attacks at your head and your body while the arms are crossed', () => {
      const g = quietGame();
      g.projs.push(incoming(0, 0), incoming(0, 30, 1000));
      run(g, 0.2, crossed());
      expect(g.hp).toBe(TUNE.maxHp);
      expect(g.drainEvents().filter(e => e.type === 'blocked')).toHaveLength(2);
    });

    it('shows no threats while it is up', () => {
      const g = quietGame();
      const p = incoming(0, 0);
      g.projs.push(p);
      g.step(1 / 60, crossed());
      expect(g.xBlock).toBe(true);
      expect(g.isThreat(p)).toBe(false);
    });
  });

  describe('aim preview', () => {
    it('points where a punch from that fist would go, and the punch goes there', () => {
      const g = new Game(mulberry32(3), 70, true);
      g.drainEvents();
      const i = intent();
      g.step(1 / 60, i);
      const preview = g.previewAim('r', { x: 0, y: 8 }, null);
      expect(preview.target).not.toBeNull();
      g.step(1 / 60, intent({ punches: [punch('r', 0, 8)] }));
      run(g, 1, intent());
      expect(g.drainEvents().some(e => (e.type === 'hitEnemy' || e.type === 'killEnemy'))).toBe(true);
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

  describe('moves you have (campaign)', () => {
    const jab = (hand: Side) => intent({ punches: [punch(hand, 0, 8)] });
    it('every move is allowed by default', () => {
      expect(quietGame().allowed).toBeNull();
    });

    it('ignores moves you have not learned', () => {
      const g = quietGame();
      g.allowed = new Set(['punch']);
      g.step(1 / 60, intent({ palms: [{ kind: 'push', hand: 'r', at: { x: 0, y: 10 }, shoulder: SHOULDERS.r, dir: null }] }));
      g.step(1 / 60, intent({ casts: [{ kind: 'wall', at: { x: 0, y: 10 } }] }));
      g.step(1 / 60, shieldUp(10));
      expect(g.pillars).toHaveLength(0);
      expect(g.walls).toHaveLength(0);
      expect(g.shield.on).toBe(false);
      g.step(1 / 60, jab('r'));
      expect(g.projs).toHaveLength(1);
    });

    it('a charged punch without the charge scroll is an ordinary punch', () => {
      const g = quietGame();
      g.allowed = new Set(['punch']);
      g.step(1 / 60, intent({ punches: [{ ...punch('r', 0, 8), charged: true }] }));
      expect(g.projs[0].shot).toBe('normal');
    });

    it('no flurry, counter or finisher without them', () => {
      const g = quietGame();
      g.allowed = new Set(['punch']);
      for (let i = 0; i < 3; i++) { g.step(1 / 60, jab(i % 2 ? 'l' : 'r')); run(g, 0.2, intent()); }
      expect(g.projs.map(p => p.shot)).not.toContain('flurry');
      g.step(1 / 60, intent({ casts: [{ kind: 'ultimate', at: { x: 0, y: 10 } }] }));
      expect(g.blades).toHaveLength(0);
    });
  });

  it('bodyHit covers head and torso only', () => {
    expect(bodyHit({ x: 0, y: 0 }, 4)).toBe(true);
    expect(bodyHit({ x: 0, y: 30 }, 4)).toBe(true);
    expect(bodyHit({ x: 30, y: 0 }, 4)).toBe(false);
  });
});
