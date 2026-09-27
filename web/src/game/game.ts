import type { Cast, Intent, Palm, Punch, Side } from '../intent/interpret';
import { distToSeg, lerp, type Vec2 } from '../math';

/** An object at depth z appears at scale FOCAL / (FOCAL + z). */
export const FOCAL = 3;
/** Floor height below the eyes, world units. */
export const FLOOR_Y = 63;
const depthScale = (z: number) => FOCAL / (FOCAL + z);

export const TUNE = {
  maxHp: 100, hitDamage: 14, invulnS: 0.5,
  punchCooldownS: 0.1, fireballSpeed: 12, fireballRadius: 4, launchZ: 0.3,
  /**
   * How far the shoulder→hand direction bends a shot beyond where the hand opened. Sideways it
   * helps cross punches and hooks; vertically it mostly overshoots, so it is kept small.
   */
  aimSkewX: 0.5, aimSkewY: 0.2, aimDepth: 10,
  /**
   * With a 3D arm direction: aim at the hand's position pushed along the punch angle (view units per
   * unit of tangent), blended with the 2D aim by aimDirWeight.
   */
  aimDirScale: 40, aimDirWeight: 0.6,
  /** Shots snap (by aimAssist) onto a target within assistRadius view units of the aim point. */
  aimAssist: 1, assistRadius: 22,
  /** Fire wall: stands at wallDepth where your hands were, blocks attacks crossing it, burns for wallLifeS. */
  wallDepth: 2.5, wallHalfWidth: 55, wallLifeS: 4, wallCooldownS: 1,
  /**
   * Ultimate: a flat blade of fire spreads out from the hands at their height, cutting down every
   * enemy and incoming attack it reaches, then recharges. Its reach grows at bladeSpeed depth
   * units/s; sideways, bladeWidthPerDepth world units count as one depth unit (a wide, flat disc).
   */
  ultimateCooldownS: 12, bladeSpeed: 16, bladeWidthPerDepth: 30, bladeMaxR: 18,
  /**
   * How far below the hands the blade sweeps, as a fraction of the way to the floor. At hand height
   * the disc would be almost at eye level and look like a thin line; lower, you see it as a layer.
   */
  bladeDrop: 0.45,
  /**
   * Palm push, the heavy attack: a pillar of fire (pillarHalfW wide, pillarHeight tall) rolls
   * forward at pillarSpeed, through every enemy and attack in its way, palmDamage per hit (a punch
   * does 1); then that hand rests for palmCooldownS.
   */
  palmDamage: 2, palmCooldownS: 0.8,
  pillarSpeed: 9, pillarHalfW: 9, pillarHeight: 75, pillarMaxZ: 14,
  /** Both palms pushed: a fire wall pushWallHalfW wide rolls forward at pushWallSpeed, burning (palmDamage) what it passes. */
  pushWallHalfW: 40, pushWallSpeed: 7, pushWallStartZ: 1.2, pushWallCooldownS: 2.5,
  /**
   * Enemies: water spirits and (earthShare of them) earthbenders. Spirits throw water orbs, or
   * (slabShare) a high sweep: a sheet of water crossing the whole field at the height your eyes
   * were — duck at least slabDuck. Earthbenders open by raising a stone pillar (over their
   * pillarWindupS wind-up) in a lane just to one side of where you stand — its centre pillarOffset
   * off your centre, pillarHalfW wide — and shoving it at you at stonePillarSpeed: lean or step the
   * other way (your body is bodyHalfW wide). After that they throw rocks, raising another pillar
   * now and then (pillarAgain). Shield and X block don't stop pillars or sweeps; a fire wall does.
   */
  earthShare: 0.35, slabShare: 0.3, slabSpeed: 6, slabDuck: 14, bodyHalfW: 12,
  pillarWindupS: 1.4, stonePillarSpeed: 3.5, stonePillarHalfW: 13, pillarHeightStone: 70, pillarOffset: 10, pillarAgain: 0.3,
  /** Enemies stay within this fraction of the screen's half-width of its centre (easier to aim at). */
  enemyBand: 0.4,
  /** Testing: the shield never drains or breaks. */
  shieldInfinite: true,
  shieldDrainPerS: 0.33, shieldRegenPerS: 0.22, shieldBlockCost: 0.18, shieldBrokenS: 1.2, shieldReach: 8,
  enemyHp: 2, enemyProjRadius: 4.2, windupS: 1, waveBreakS: 2.2,
};

export interface Enemy {
  id: number; x: number; y: number; z: number; hp: number;
  t: number; appear: number; dying: number; flash: number;
  cd: number; winding: boolean; wind: number; side: 1 | -1; phase: number;
  /** The attack it is winding up (picked when the wind-up starts). */
  attack?: AttackKind;
  /** Earthbender (stone pillars and rocks) instead of a water spirit. */
  earth?: boolean;
  /** Earthbender: has already opened with a pillar. */
  opened?: boolean;
  /** Practice target: never moves or attacks, respawns in its slot. */
  dummy?: boolean;
  slot?: number;
}

export interface Proj {
  id: number; kind: 'player' | 'enemy';
  /** An earthbender's rock rather than a water orb. */
  rock?: boolean;
  x: number; y: number; z: number; vx: number; vy: number; vz: number; r: number;
  resolved: boolean;
}

type PositionedType = 'blocked' | 'playerHit' | 'dodged' | 'hitEnemy' | 'killEnemy' | 'clash' | 'wall' | 'ultimate' | 'cut' | 'wallPush' | 'slab';
export type GameEvent =
  | { type: PositionedType; x: number; y: number; z: number }
  | { type: 'punch' | 'pillar'; x: number; y: number; z: number; side: Side }
  | { type: 'stonePillar'; x: number; y: number; z: number; side: 1 | -1 }
  | { type: 'shieldBroken' | 'gameOver' }
  | { type: 'wave'; wave: number };

export type Rand = () => number;

/** The ultimate's spinning disc of fire: centred at the hands (world x, y), reach r in depth units. */
export interface Blade { id: number; x: number; y: number; r: number }

/** A palm push: a column of fire rolling forward from the floor, `hit` = enemies it already burned. */
export interface Pillar { id: number; x: number; z: number; vx: number; age: number; hit: number[] }

export type AttackKind = 'orb' | 'rock' | 'slab' | 'pillar';

/**
 * An attack you have to move out of, travelling toward you at depth z. A stone pillar rises in
 * front of its earthbender (rise 0 → 1, `owner` while it rises), then slides toward `laneX`, where
 * it arrives; a slab (high sweep) crosses the whole field at height y.
 */
export interface Hazard {
  id: number; kind: 'stonePillar' | 'slab'; x: number; y: number; z: number; vz: number; resolved: boolean;
  laneX: number; side: 1 | -1; startX: number; startZ: number; rise: number; owner: number | null;
}

/** A wall of fire across the courtyard: standing (vz 0), or rolling forward (a wall push) burning what it passes. */
export interface Wall { id: number; x: number; z: number; halfW: number; life: number; vz: number; hit: number[] }

/** Where practice dummies stand (world x, depth). */
const DUMMY_SLOTS = [{ x: -36, z: 6 }, { x: 0, z: 9 }, { x: 36, z: 6 }];
const DUMMY_RESPAWN_S = 1.5;

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
  hands: Intent['hands'] = { l: null, r: null };
  shield = { on: false, energy: 1, broken: 0 };
  inv = 0;
  enemies: Enemy[] = [];
  projs: Proj[] = [];
  walls: Wall[] = [];
  blades: Blade[] = [];
  pillars: Pillar[] = [];
  hazards: Hazard[] = [];
  /** Forearms crossed: everything that reaches you is blocked. */
  xBlock = false;
  /** Last known shoulder positions (view space), for aiming. */
  shoulders: Record<Side, Vec2> = { l: { x: -20, y: 20 }, r: { x: 20, y: 20 } };
  /** Seconds until the ultimate is ready again. */
  ultimateIn = 0;
  /** Tests turn this off to control enemies by hand. */
  spawning = true;
  /** Dummies instead of attacking spirits. */
  practice = false;

  private events: GameEvent[] = [];
  private toSpawn = 0;
  /** An earthbender has come this wave (every wave brings at least one). */
  private waveEarth = false;
  private spawnT = 0;
  private waveBreak = 0;
  private nextId = 1;
  private dummyTimers = DUMMY_SLOTS.map(() => 0);
  private punchCool: Record<Side, number> = { l: 0, r: 0 };
  private wallCool = 0;
  private palmCool: Record<Side, number> = { l: 0, r: 0 };
  private pushWallCool = 0;

  constructor(private rand: Rand = Math.random, public viewHalfW = 70, practice = false) {
    if (practice) this.setPractice(true);
    else this.startWave();
  }

  /** Switch between practice dummies and spirit waves, clearing the field. */
  setPractice(on: boolean): void {
    this.practice = on;
    this.enemies = [];
    this.projs = this.projs.filter(p => p.kind === 'player');
    this.hazards = [];
    if (on) {
      this.dummyTimers = DUMMY_SLOTS.map(() => 0);
      this.spawnDummies(0);
    } else {
      this.wave = 0;
      this.waveBreak = 0;
      this.startWave();
    }
  }

  step(dt: number, intent: Intent): void {
    this.cam = { ...intent.head };
    this.hands = intent.hands;
    this.xBlock = intent.xBlock;
    if (intent.shoulders) this.shoulders = intent.shoulders;
    this.inv = Math.max(0, this.inv - dt);
    this.punchCool = { l: Math.max(0, this.punchCool.l - dt), r: Math.max(0, this.punchCool.r - dt) };
    this.wallCool = Math.max(0, this.wallCool - dt);
    this.pushWallCool = Math.max(0, this.pushWallCool - dt);
    this.palmCool = { l: Math.max(0, this.palmCool.l - dt), r: Math.max(0, this.palmCool.r - dt) };
    this.ultimateIn = Math.max(0, this.ultimateIn - dt);
    this.updateShield(dt, intent.shield);
    if (this.state !== 'play') return;
    for (const p of intent.punches) this.punch(p);
    for (const c of intent.casts) this.cast(c);
    for (const p of intent.palms ?? []) this.palm(p);
    this.updateWalls(dt);
    this.updateBlades(dt);
    this.updatePillars(dt);
    this.updateWaves(dt);
    this.updateEnemies(dt);
    this.updateProjs(dt);
    this.updateHazards(dt);
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
    if (this.xBlock) return false;
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
    const { l, r: rh } = this.hands;
    return this.shield.on && !!l && !!rh && distToSeg(v, l.pos, rh.pos) < TUNE.shieldReach + r;
  }

  private updateShield(dt: number, wanted: boolean): void {
    const sh = this.shield;
    sh.broken = Math.max(0, sh.broken - dt);
    sh.on = wanted && !!this.hands.l && !!this.hands.r && sh.energy > 0 && sh.broken <= 0;
    if (!sh.on) {
      sh.energy = Math.min(1, sh.energy + dt * TUNE.shieldRegenPerS);
    } else if (!TUNE.shieldInfinite) {
      sh.energy = Math.max(0, sh.energy - dt * TUNE.shieldDrainPerS);
      if (sh.energy <= 0) {
        sh.on = false;
        sh.broken = TUNE.shieldBrokenS;
        this.events.push({ type: 'shieldBroken' });
      }
    }
  }

  /** Fire leaves the opened hand toward where it points: its screen position, bent further along shoulder → hand. */
  /**
   * Where a punch from `at` (view space) would go: the fist's screen position, bent by the punch
   * direction, then snapped onto a nearby target. Returns the world point it flies to (at `depth`)
   * and the target, if any. The game and the on-screen aim reticle both use this.
   */
  aimFor(at: Vec2, shoulder: Vec2, dir: Vec2 | null): { point: Vec2; depth: number; target: Enemy | null } {
    let aim = { x: at.x + (at.x - shoulder.x) * TUNE.aimSkewX, y: at.y + (at.y - shoulder.y) * TUNE.aimSkewY };
    if (dir) {
      const along = { x: at.x + dir.x * TUNE.aimDirScale, y: at.y + dir.y * TUNE.aimDirScale };
      aim = { x: lerp(aim.x, along.x, TUNE.aimDirWeight), y: lerp(aim.y, along.y, TUNE.aimDirWeight) };
    }
    let depth = TUNE.aimDepth;
    const target = this.pickTarget(aim);
    if (target) {
      const s = depthScale(target.z);
      aim = { x: lerp(aim.x, (target.x - this.cam.x) * s, TUNE.aimAssist), y: lerp(aim.y, (target.y - this.cam.y) * s, TUNE.aimAssist) };
      depth = target.z;
    }
    // the world point that appears at `aim` on screen at that depth
    const s = depthScale(depth);
    return { point: { x: this.cam.x + aim.x / s, y: Math.min(FLOOR_Y - 4, this.cam.y + aim.y / s) }, depth, target };
  }

  /** What a punch from this hand, as it is now, would hit (for the aim reticle). */
  previewAim(side: Side, at: Vec2, dir: Vec2 | null): { point: Vec2; depth: number; target: Enemy | null } {
    return this.aimFor(at, this.shoulders[side], dir);
  }

  private punch(p: Punch): void {
    if (this.punchCool[p.hand] > 0) return;
    this.punchCool[p.hand] = TUNE.punchCooldownS;
    const start = this.handWorld(p.at);
    const { point: target, depth } = this.aimFor(p.at, p.shoulder, p.dir);
    const T = (depth - TUNE.launchZ) / TUNE.fireballSpeed;
    this.projs.push({
      id: this.nextId++, kind: 'player', x: start.x, y: start.y, z: TUNE.launchZ,
      vx: (target.x - start.x) / T, vy: (target.y - start.y) / T, vz: TUNE.fireballSpeed, r: TUNE.fireballRadius, resolved: false,
    });
    this.events.push({ type: 'punch', x: start.x, y: start.y, z: TUNE.launchZ, side: p.hand });
  }

  private palm(p: Palm): void {
    if (this.palmCool[p.hand] > 0) return;
    this.palmCool[p.hand] = TUNE.palmCooldownS;
    // rolls from in front of the hand toward where the palm points
    const { point, depth } = this.aimFor(p.at, p.shoulder, p.dir);
    const start = this.handWorld(p.at).x, z = TUNE.launchZ;
    const vx = ((point.x - start) / Math.max(1, depth - z)) * TUNE.pillarSpeed;
    this.pillars.push({ id: this.nextId++, x: start, z, vx, age: 0, hit: [] });
    this.events.push({ type: 'pillar', x: start, y: FLOOR_Y, z, side: p.hand });
  }

  /** Standing walls burn down; rolling ones move forward, burning each enemy they pass once. */
  private updateWalls(dt: number): void {
    for (const w of this.walls) {
      w.life -= dt;
      if (!w.vz) continue;
      const z0 = w.z;
      w.z += w.vz * dt;
      for (const e of this.enemies) {
        if (e.hp <= 0 || w.hit.includes(e.id) || e.z < z0 - 0.5 || e.z > w.z + 0.5 || Math.abs(e.x - w.x) > w.halfW + 7) continue;
        w.hit.push(e.id);
        this.burn(e);
      }
    }
    this.walls = this.walls.filter(w => w.life > 0 && w.z < TUNE.pillarMaxZ);
  }

  /** A wall that something moving from z0 to z just met (both may be moving). */
  private wallMet(x: number, r: number, z0: number, z: number, dt: number): Wall | undefined {
    return this.walls.find(w => {
      const wz0 = w.z - w.vz * dt;
      return z0 - wz0 > 0 && z - w.z <= 0 && Math.abs(x - w.x) <= w.halfW + r;
    });
  }

  /**
   * Raise pillars while their earthbender winds up (a pillar crumbles if he falls), move pillars and
   * sweeps toward you; walls stop them; when they arrive, did you get out of the way?
   */
  private updateHazards(dt: number): void {
    const dead = new Set<number>();
    for (const h of this.hazards) {
      if (h.owner !== null) {
        const e = this.enemies.find(x => x.id === h.owner);
        if (!e || e.hp <= 0 || !e.winding) {
          dead.add(h.id);
          this.emit('cut', h.x, FLOOR_Y - 20, h.z);
        } else h.rise = Math.min(1, e.wind / 0.6);
        continue;
      }
      const z0 = h.z;
      h.z += h.vz * dt;
      // a stone pillar is shoved from in front of its earthbender toward its lane beside you
      if (h.kind === 'stonePillar') h.x = lerp(h.laneX, h.startX, Math.max(0, h.z) / h.startZ);
      // a wall stops it if it stands between the attack and you
      const wall = this.wallMet(h.kind === 'stonePillar' ? h.x : this.cam.x, h.kind === 'stonePillar' ? TUNE.stonePillarHalfW : 0, z0, h.z, dt);
      if (wall) {
        dead.add(h.id);
        this.score += 15;
        this.emit('blocked', h.x, h.kind === 'stonePillar' ? FLOOR_Y - 20 : h.y, wall.z);
        continue;
      }
      if (!h.resolved && h.z <= 0.2) {
        h.resolved = true;
        const hit = h.kind === 'stonePillar'
          ? Math.abs(this.cam.x - h.laneX) < TUNE.stonePillarHalfW + TUNE.bodyHalfW
          : this.cam.y - h.y < TUNE.slabDuck;
        if (!hit) {
          this.score += 20;
          this.emit('dodged', this.cam.x, this.cam.y + 10, 0);
        } else if (this.inv <= 0) {
          this.hurt(this.cam.x, h.kind === 'stonePillar' ? this.cam.y + 20 : h.y);
        }
      }
      if (h.z < -1.5) dead.add(h.id);
    }
    if (dead.size) this.hazards = this.hazards.filter(h => !dead.has(h.id));
  }

  private hurt(x: number, y: number): void {
    this.hp = Math.max(0, this.hp - TUNE.hitDamage);
    this.inv = TUNE.invulnS;
    this.emit('playerHit', x, y, 0);
    if (this.hp <= 0) {
      this.state = 'over';
      this.events.push({ type: 'gameOver' });
    }
  }

  /** Burn an enemy with a pillar. */
  private burn(e: Enemy): void {
    e.hp -= TUNE.palmDamage;
    e.flash = 1;
    if (e.hp <= 0) { this.score += 100; this.emit('killEnemy', e.x, e.y, e.z); }
    else this.emit('hitEnemy', e.x, e.y, e.z);
  }

  /** Roll each pillar forward, burning every enemy it passes (once) and every attack it meets. */
  private updatePillars(dt: number): void {
    for (const c of this.pillars) {
      const z0 = c.z;
      c.x += c.vx * dt;
      c.z += TUNE.pillarSpeed * dt;
      c.age += dt;
      for (const e of this.enemies) {
        if (e.hp <= 0 || c.hit.includes(e.id) || e.z < z0 - 0.5 || e.z > c.z + 0.5 || Math.abs(e.x - c.x) > TUNE.pillarHalfW + 7) continue;
        c.hit.push(e.id);
        this.burn(e);
      }
      this.projs = this.projs.filter(p => {
        if (p.kind !== 'enemy' || Math.abs(p.z - c.z) > 0.8 || Math.abs(p.x - c.x) > TUNE.pillarHalfW + p.r) return true;
        this.score += 25;
        this.emit('clash', p.x, p.y, p.z);
        return false;
      });
    }
    this.pillars = this.pillars.filter(c => c.z < TUNE.pillarMaxZ);
  }

  /** 0 = just used … 1 = ready. */
  get ultimateCharge(): number {
    return 1 - this.ultimateIn / TUNE.ultimateCooldownS;
  }

  private cast(c: Cast): void {
    if (c.kind === 'wall') {
      if (this.wallCool > 0) return;
      this.wallCool = TUNE.wallCooldownS;
      // stand the wall where the hands appear on screen, at its depth
      const x = this.cam.x + c.at.x / depthScale(TUNE.wallDepth);
      this.walls.push({ id: this.nextId++, x, z: TUNE.wallDepth, halfW: TUNE.wallHalfWidth, life: TUNE.wallLifeS, vz: 0, hit: [] });
      this.emit('wall', x, FLOOR_Y, TUNE.wallDepth);
      return;
    }
    if (c.kind === 'push') {
      if (this.pushWallCool > 0) return;
      this.pushWallCool = TUNE.pushWallCooldownS;
      const z = TUNE.pushWallStartZ, x = this.cam.x + c.at.x / depthScale(z);
      this.walls.push({ id: this.nextId++, x, z, halfW: TUNE.pushWallHalfW, life: Infinity, vz: TUNE.pushWallSpeed, hit: [] });
      this.emit('wallPush', x, FLOOR_Y, z);
      return;
    }
    if (this.ultimateIn > 0) return;
    this.ultimateIn = TUNE.ultimateCooldownS;
    const at = this.handWorld(c.at);
    const y = at.y + (FLOOR_Y - at.y) * TUNE.bladeDrop;
    this.blades.push({ id: this.nextId++, x: at.x, y, r: 0 });
    this.emit('ultimate', at.x, at.y, 0);
  }

  /** Grow each blade and cut down whatever its edge has reached. */
  private updateBlades(dt: number): void {
    const reach = (b: Blade, x: number, z: number) => Math.hypot((x - b.x) / TUNE.bladeWidthPerDepth, z) <= b.r;
    for (const b of this.blades) {
      b.r += TUNE.bladeSpeed * dt;
      for (const e of this.enemies) {
        if (e.hp <= 0 || !reach(b, e.x, e.z)) continue;
        e.hp = 0;
        e.flash = 1;
        this.score += 100;
        this.emit('killEnemy', e.x, e.y, e.z);
      }
      this.projs = this.projs.filter(p => {
        if (p.kind !== 'enemy' || !reach(b, p.x, Math.max(p.z, 0))) return true;
        this.emit('cut', p.x, p.y, p.z);
        return false;
      });
    }
    this.blades = this.blades.filter(b => b.r < TUNE.bladeMaxR);
  }

  /** The living enemy that appears closest to `aim` on screen, if within assist range. */
  private pickTarget(aim: Vec2): Enemy | null {
    let best: Enemy | null = null, bestD = TUNE.assistRadius;
    for (const e of this.enemies) {
      if (e.hp <= 0) continue;
      const s = depthScale(e.z);
      const d = Math.hypot((e.x - this.cam.x) * s - aim.x, (e.y - this.cam.y) * s - aim.y);
      if (d < bestD) { bestD = d; best = e; }
    }
    return best;
  }

  private startWave(): void {
    this.wave++;
    this.toSpawn = 2 + this.wave;
    this.waveEarth = false;
    this.spawnT = 0.6;
    this.events.push({ type: 'wave', wave: this.wave });
  }

  private updateWaves(dt: number): void {
    if (!this.spawning) return;
    if (this.practice) {
      this.spawnDummies(dt);
      return;
    }
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

  /** Fill empty dummy slots once their respawn timer runs out. */
  private spawnDummies(dt: number): void {
    DUMMY_SLOTS.forEach((slot, i) => {
      if (this.enemies.some(e => e.slot === i)) return;
      this.dummyTimers[i] -= dt;
      if (this.dummyTimers[i] > 0) return;
      this.dummyTimers[i] = DUMMY_RESPAWN_S;
      this.enemies.push({
        id: this.nextId++, x: slot.x, y: FLOOR_Y - 30, z: slot.z, hp: TUNE.enemyHp, t: 0, appear: 0, dying: 0, flash: 0,
        cd: Infinity, winding: false, wind: 0, side: 1, phase: 0, dummy: true, slot: i,
      });
    });
  }

  private spawnEnemy(): void {
    const z = this.rnd(6.5, 11), s = depthScale(z);
    // this.toSpawn still counts this one: if no earthbender yet, one of the last two is
    const earth = (this.toSpawn <= 2 && !this.waveEarth) || this.rand() < TUNE.earthShare;
    this.waveEarth ||= earth;
    // near the middle of the screen, away from where the others stand
    let x = 0;
    for (let tries = 0; tries < 6; tries++) {
      x = (this.rnd(-1, 1) * this.viewHalfW * TUNE.enemyBand) / s;
      if (this.enemies.every(e => e.hp <= 0 || Math.abs(e.x * depthScale(e.z) - x * s) > 10)) break;
    }
    this.enemies.push({
      id: this.nextId++, x, y: FLOOR_Y - 30, z,
      hp: TUNE.enemyHp, t: 0, appear: 0, dying: 0, flash: 0,
      cd: this.rnd(1.2, 2.6), winding: false, wind: 0, side: this.rand() < 0.5 ? -1 : 1, phase: this.rnd(0, 6),
      earth,
    });
  }

  /** Most you can drift sideways (world x) at depth z and stay near the middle of the screen. */
  private bandAt(z: number): number {
    return (this.viewHalfW * TUNE.enemyBand) / depthScale(z);
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
      if (e.dummy) continue;
      // sway gently, staying near the middle of the screen
      const band = this.bandAt(e.z);
      e.x = Math.max(-band, Math.min(band, e.x + (Math.sin(e.t * 0.5 + e.phase) * 5 * dt) / depthScale(e.z)));
      if (e.appear < 1) continue;
      if (!e.winding) {
        e.cd -= dt;
        if (e.cd <= 0) {
          e.winding = true;
          e.wind = 0;
          e.attack ??= this.pickAttack(e);
          if (e.attack === 'pillar') this.raisePillar(e);
        }
      } else {
        e.wind += dt / (e.attack === 'pillar' ? TUNE.pillarWindupS : TUNE.windupS);
        if (e.wind >= 1) {
          this.enemyAttack(e);
          e.winding = false;
          e.attack = undefined;
          e.cd = this.rnd(baseCd, baseCd + 1.5);
          e.side = e.side === 1 ? -1 : 1;
        }
      }
    }
  }

  private pickAttack(e: Enemy): AttackKind {
    if (e.earth) return !e.opened || this.rand() < TUNE.pillarAgain ? 'pillar' : 'rock';
    return this.rand() < TUNE.slabShare ? 'slab' : 'orb';
  }

  /** An earthbender stomps: a stone pillar starts rising in front of him, aimed at a lane beside you. */
  private raisePillar(e: Enemy): void {
    const side: 1 | -1 = this.rand() < 0.5 ? -1 : 1, z = e.z - 0.6, x = e.x + side * 16;
    this.hazards.push({
      id: this.nextId++, kind: 'stonePillar', x, y: FLOOR_Y, z, vz: 0, resolved: false,
      laneX: this.cam.x + side * TUNE.pillarOffset, side, startX: x, startZ: z, rise: 0, owner: e.id,
    });
  }

  private enemyAttack(e: Enemy): void {
    const kind = e.attack ?? 'orb';
    if (kind === 'orb' || kind === 'rock') { this.enemyThrow(e, kind === 'rock'); return; }
    if (kind === 'pillar') {
      // shove the raised pillar at you
      e.opened = true;
      const h = this.hazards.find(x => x.owner === e.id);
      if (!h) return;
      h.owner = null;
      h.rise = 1;
      h.vz = -TUNE.stonePillarSpeed;
      this.events.push({ type: 'stonePillar', x: h.x, y: FLOOR_Y, z: h.z, side: h.side });
      return;
    }
    const z = e.z - 0.1;
    this.hazards.push({ id: this.nextId++, kind, x: e.x, y: this.cam.y, z, vz: -TUNE.slabSpeed, resolved: false, laneX: 0, side: 1, startX: e.x, startZ: z, rise: 1, owner: null });
    this.emit('slab', e.x, this.cam.y, z);
  }

  /** Aim at where your head/chest is now; moving afterwards is how you dodge. */
  private enemyThrow(e: Enemy, rock = false): void {
    const tx = this.cam.x + this.rnd(-4, 4), ty = this.cam.y + this.rnd(-3, 12);
    const hx = e.x + e.side * 11, hy = e.y - 14, z = e.z - 0.1;
    const vz = -(4.6 + this.wave * 0.35), T = z / -vz;
    this.projs.push({ id: this.nextId++, kind: 'enemy', rock, x: hx, y: hy, z, vx: (tx - hx) / T, vy: (ty - hy) / T, vz, r: TUNE.enemyProjRadius, resolved: false });
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
        const wall = this.wallMet(p.x, p.r, p.z - p.vz * dt, p.z, dt);
        if (wall) {
          dead.add(p.id);
          this.score += 15;
          this.emit('blocked', p.x, p.y, wall.z);
          continue;
        }
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
    if (this.xBlock && bodyHit(v, q.r * 2)) {
      this.score += 15;
      this.emit('blocked', q.x, q.y, 0);
      return true;
    }
    if (this.shieldCovers(v, q.r)) {
      this.shield.energy = Math.max(0, this.shield.energy - TUNE.shieldBlockCost);
      this.score += 15;
      this.emit('blocked', q.x, q.y, 0);
      return true;
    }
    if (this.inv <= 0 && bodyHit(v, q.r)) {
      this.hurt(q.x, q.y);
      return true;
    }
    if (Math.hypot(v.x, v.y - 10) < 30) {
      this.score += 10;
      this.emit('dodged', q.x, q.y, 0);
    }
    return false;
  }
}
