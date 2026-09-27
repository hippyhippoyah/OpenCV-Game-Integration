import type { HandsIntent, Intent } from '../intent/interpret';
import { distToSeg, type Vec2 } from '../math';

/** An object at depth z appears at scale FOCAL / (FOCAL + z). */
export const FOCAL = 3;
/** Floor height below the eyes, world units. */
export const FLOOR_Y = 63;

export const TUNE = {
  maxHp: 100, hitDamage: 14, invulnS: 0.5,
  summonSpread: 9, shieldSpread: 22, dropBelowY: 45,
  fireCooldownS: 0.5, fireballSpeed: 11, fireballRadius: 4.5, aimAssist: 0.8,
  shieldDrainPerS: 0.33, shieldRegenPerS: 0.22, shieldBlockCost: 0.18, shieldBrokenS: 1.2, shieldReach: 8,
  enemyHp: 2, enemyProjRadius: 4.2, windupS: 1, waveBreakS: 2.2,
};

export interface Enemy {
  id: number; x: number; y: number; z: number; hp: number;
  t: number; appear: number; dying: number; flash: number;
  cd: number; winding: boolean; wind: number; side: 1 | -1; phase: number;
}

export interface Proj {
  id: number; kind: 'player' | 'enemy';
  x: number; y: number; z: number; vx: number; vy: number; vz: number; r: number;
  resolved: boolean;
}

type PositionedType = 'summon' | 'extinguish' | 'throw' | 'blocked' | 'playerHit' | 'dodged' | 'hitEnemy' | 'killEnemy' | 'clash';
export type GameEvent =
  | { type: PositionedType; x: number; y: number; z: number }
  | { type: 'shieldBroken' | 'gameOver' }
  | { type: 'wave'; wave: number };

export type Rand = () => number;

/** Head + torso hitbox. `v` is relative to the eyes. */
export function bodyHit(v: Vec2, r: number): boolean {
  if (Math.hypot(v.x, v.y) < 8 + r * 0.5) return true;
  return Math.abs(v.x) < 12 + r * 0.5 && v.y > 6 && v.y < 70;
}

/** Where an incoming projectile crosses the player's plane (z = 0). */
export function arrival(p: Proj): Vec2 {
  const T = p.z / -p.vz;
  return { x: p.x + p.vx * T, y: p.y + p.vy * T };
}

export class Game {
  state: 'play' | 'over' = 'play';
  hp = TUNE.maxHp;
  score = 0;
  wave = 0;
  cam: Vec2 = { x: 0, y: 0 };
  hands: HandsIntent | null = null;
  fire = { held: false, cool: 0 };
  shield = { on: false, energy: 1, broken: 0 };
  inv = 0;
  enemies: Enemy[] = [];
  projs: Proj[] = [];
  /** Tests turn this off to control enemies by hand. */
  spawning = true;

  private events: GameEvent[] = [];
  private toSpawn = 0;
  private spawnT = 0;
  private waveBreak = 0;
  private nextId = 1;

  constructor(private rand: Rand = Math.random, public viewHalfW = 70) {
    this.startWave();
  }

  step(dt: number, intent: Intent): void {
    this.cam = { ...intent.head };
    this.hands = intent.hands;
    this.inv = Math.max(0, this.inv - dt);
    this.updateHands(dt, intent);
    if (this.state !== 'play') return;
    this.updateWaves(dt);
    this.updateEnemies(dt);
    this.updateProjs(dt);
  }

  drainEvents(): GameEvent[] {
    const e = this.events;
    this.events = [];
    return e;
  }

  /** A view-space point (e.g. a hand) in world space. */
  handWorld(p: Vec2): Vec2 {
    return { x: p.x + this.cam.x, y: p.y + this.cam.y };
  }

  /** Would this incoming attack hit you if you stayed exactly as you are? */
  isThreat(p: Proj): boolean {
    const a = arrival(p), v = { x: a.x - this.cam.x, y: a.y - this.cam.y };
    return bodyHit(v, p.r) && !this.shieldCovers(v, p.r);
  }

  private rnd(a: number, b: number): number {
    return a + this.rand() * (b - a);
  }

  private emit(type: PositionedType, x: number, y: number, z: number): void {
    this.events.push({ type, x, y, z });
  }

  private shieldCovers(v: Vec2, r: number): boolean {
    return this.shield.on && this.hands !== null && distToSeg(v, this.hands.l, this.hands.r) < TUNE.shieldReach + r;
  }

  private updateHands(dt: number, intent: Intent): void {
    const h = this.hands, fire = this.fire, sh = this.shield;
    fire.cool = Math.max(0, fire.cool - dt);
    sh.broken = Math.max(0, sh.broken - dt);

    sh.on = h !== null && intent.raised && h.spread >= TUNE.shieldSpread && sh.energy > 0 && sh.broken <= 0;
    if (sh.on) {
      fire.held = false; // the fireball spreads into the shield
      sh.energy = Math.max(0, sh.energy - dt * TUNE.shieldDrainPerS);
      if (sh.energy <= 0) {
        sh.on = false;
        sh.broken = TUNE.shieldBrokenS;
        this.events.push({ type: 'shieldBroken' });
      }
    } else {
      sh.energy = Math.min(1, sh.energy + dt * TUNE.shieldRegenPerS);
    }

    if (h === null) {
      if (fire.held) this.extinguish();
      return;
    }
    if (!fire.held && !sh.on && fire.cool <= 0 && intent.raised && h.spread < TUNE.summonSpread) {
      fire.held = true;
      const c = this.handWorld(h.center);
      this.emit('summon', c.x, c.y, 0);
    }
    if (fire.held && h.center.y > TUNE.dropBelowY) this.extinguish();
    if (intent.throwNow && fire.held) this.throwFire(h);
  }

  private extinguish(): void {
    this.fire.held = false;
    const c = this.hands ? this.handWorld(this.hands.center) : this.cam;
    this.emit('extinguish', c.x, c.y, 0);
  }

  private throwFire(h: HandsIntent): void {
    const w = this.handWorld(h.center), vz = TUNE.fireballSpeed;
    let vx = h.vel.x * 0.5, vy = h.vel.y * 0.5 - 3;
    const tgt = this.pickTarget(h.center.x + h.vel.x * 0.25);
    if (tgt) {
      const T = (tgt.z - 0.3) / vz;
      vx += ((tgt.x - w.x) / T - vx) * TUNE.aimAssist;
      vy += ((tgt.y - w.y) / T - vy) * TUNE.aimAssist;
    }
    this.projs.push({ id: this.nextId++, kind: 'player', x: w.x, y: w.y, z: 0.3, vx, vy, vz, r: TUNE.fireballRadius, resolved: false });
    this.fire.held = false;
    this.fire.cool = TUNE.fireCooldownS;
    this.emit('throw', w.x, w.y, 0.3);
  }

  /** The living enemy whose on-screen x is closest to `viewX`. */
  private pickTarget(viewX: number): Enemy | null {
    let best: Enemy | null = null, bestD = Infinity;
    for (const e of this.enemies) {
      if (e.hp <= 0) continue;
      const d = Math.abs((e.x - this.cam.x) * (FOCAL / (FOCAL + e.z)) - viewX);
      if (d < bestD) { bestD = d; best = e; }
    }
    return best;
  }

  private startWave(): void {
    this.wave++;
    this.toSpawn = 2 + this.wave;
    this.spawnT = 0.6;
    this.events.push({ type: 'wave', wave: this.wave });
  }

  private updateWaves(dt: number): void {
    if (!this.spawning) return;
    const alive = this.enemies.filter(e => e.hp > 0).length;
    if (this.toSpawn > 0) {
      this.spawnT -= dt;
      if (this.spawnT <= 0 && alive < 2 + Math.ceil(this.wave / 2)) {
        this.spawnEnemy();
        this.toSpawn--;
        this.spawnT = this.rnd(0.8, 1.8);
      }
    } else if (this.enemies.length === 0) {
      this.waveBreak += dt;
      if (this.waveBreak > TUNE.waveBreakS) {
        this.waveBreak = 0;
        this.startWave();
      }
    }
  }

  private spawnEnemy(): void {
    const z = this.rnd(6.5, 11), s = FOCAL / (FOCAL + z);
    this.enemies.push({
      id: this.nextId++, x: this.cam.x + (this.rnd(-1, 1) * this.viewHalfW * 0.85) / s, y: FLOOR_Y - 30, z,
      hp: TUNE.enemyHp, t: 0, appear: 0, dying: 0, flash: 0,
      cd: this.rnd(1.2, 2.6), winding: false, wind: 0, side: this.rand() < 0.5 ? -1 : 1, phase: this.rnd(0, 6),
    });
  }

  private updateEnemies(dt: number): void {
    const baseCd = Math.max(1.4, 3.4 - this.wave * 0.25);
    for (let i = this.enemies.length - 1; i >= 0; i--) {
      const e = this.enemies[i];
      e.t += dt;
      e.appear = Math.min(1, e.appear + dt * 1.2);
      e.flash = Math.max(0, e.flash - dt * 4);
      if (e.hp <= 0) {
        e.dying += dt * 2.2;
        if (e.dying >= 1) this.enemies.splice(i, 1);
        continue;
      }
      e.x += (Math.sin(e.t * 0.5 + e.phase) * 8 * dt) / (FOCAL / (FOCAL + e.z));
      if (e.appear < 1) continue;
      if (!e.winding) {
        e.cd -= dt;
        if (e.cd <= 0) { e.winding = true; e.wind = 0; }
      } else {
        e.wind += dt / TUNE.windupS;
        if (e.wind >= 1) {
          this.enemyThrow(e);
          e.winding = false;
          e.cd = this.rnd(baseCd, baseCd + 1.5);
          e.side = e.side === 1 ? -1 : 1;
        }
      }
    }
  }

  /** Aim at where your head/chest is now; moving afterwards is how you dodge. */
  private enemyThrow(e: Enemy): void {
    const tx = this.cam.x + this.rnd(-4, 4), ty = this.cam.y + this.rnd(-3, 12);
    const hx = e.x + e.side * 11, hy = e.y - 14, z = e.z - 0.1;
    const vz = -(4.6 + this.wave * 0.35), T = z / -vz;
    this.projs.push({ id: this.nextId++, kind: 'enemy', x: hx, y: hy, z, vx: (tx - hx) / T, vy: (ty - hy) / T, vz, r: TUNE.enemyProjRadius, resolved: false });
  }

  private updateProjs(dt: number): void {
    const dead = new Set<number>();
    for (const p of this.projs) {
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z += p.vz * dt;
    }
    for (const p of this.projs) {
      if (dead.has(p.id)) continue;
      if (p.kind === 'player') {
        for (const e of this.enemies) {
          if (e.hp <= 0 || Math.abs(p.z - e.z) > 0.7) continue;
          if (Math.abs(p.x - e.x) < p.r + 7 && Math.abs(p.y - e.y) < p.r + 22) {
            e.hp -= 1;
            e.flash = 1;
            dead.add(p.id);
            if (e.hp <= 0) { this.score += 100; this.emit('killEnemy', p.x, p.y, p.z); }
            else this.emit('hitEnemy', p.x, p.y, p.z);
            break;
          }
        }
        if (dead.has(p.id)) continue;
        for (const q of this.projs) {
          if (q.kind !== 'enemy' || dead.has(q.id) || Math.abs(p.z - q.z) > 0.8) continue;
          if (Math.hypot(p.x - q.x, p.y - q.y) < p.r + q.r + 2) {
            dead.add(p.id);
            dead.add(q.id);
            this.score += 25;
            this.emit('clash', q.x, q.y, q.z);
            break;
          }
        }
        if (p.z > 14) dead.add(p.id);
      } else {
        if (!p.resolved && p.z <= 0.2) {
          p.resolved = true;
          if (this.resolveIncoming(p)) dead.add(p.id);
        }
        if (p.z < -1.5) dead.add(p.id);
      }
    }
    if (dead.size) this.projs = this.projs.filter(p => !dead.has(p.id));
  }

  /** Returns true if the projectile was absorbed (blocked or hit you). */
  private resolveIncoming(q: Proj): boolean {
    const v = { x: q.x - this.cam.x, y: q.y - this.cam.y };
    if (this.shieldCovers(v, q.r)) {
      this.shield.energy = Math.max(0, this.shield.energy - TUNE.shieldBlockCost);
      this.score += 15;
      this.emit('blocked', q.x, q.y, 0);
      return true;
    }
    if (this.inv <= 0 && bodyHit(v, q.r)) {
      this.hp = Math.max(0, this.hp - TUNE.hitDamage);
      this.inv = TUNE.invulnS;
      this.emit('playerHit', q.x, q.y, 0);
      if (this.hp <= 0) {
        this.state = 'over';
        this.events.push({ type: 'gameOver' });
      }
      return true;
    }
    if (Math.hypot(v.x, v.y - 10) < 30) {
      this.score += 10;
      this.emit('dodged', q.x, q.y, 0);
    }
    return false;
  }
}
