import type { Cast, Intent, Palm, Punch, Side } from '../intent/interpret';
import { distToSeg, lerp, type Vec2 } from '../math';
import { BOSS, bossDamage, newBossState, updateBoss, type BossState } from './boss';

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
  /**
   * Breath: fire comes from your breath. Every attack spends some (a jab very little, a wall a
   * lot); it comes back breathRegen per second, breathRestRegen once you've not attacked for
   * breathRestS. An attack you haven't the breath for fizzles. The finisher uses the ultimate bar.
   */
  breathMax: 100, breathRegen: 12, breathRestRegen: 30, breathRestS: 0.8,
  breathPunch: 6, breathCharged: 16, breathPalm: 12, breathWall: 25, breathWallPush: 10,
  ultimateCooldownS: 7, bladeSpeed: 16, bladeWidthPerDepth: 30, bladeMaxR: 18,
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
   * (slabShare) a high sweep: a wave of water crossing the whole field, always at slabY (your eye
   * height standing up, even if you were ducking when it was sent) — duck at least slabDuck below
   * it. Earthbenders raise stone pillars (over their pillarWindupS
   * wind-up) out of the ground in a lane to one side of where you stand — its centre pillarOffset
   * off your centre, stonePillarHalfW wide, so it clearly runs down your left or right — and shove
   * them straight down that lane at stonePillarSpeed: lean or step the other way (your body is
   * bodyHalfW wide). Shield and X block don't stop pillars or sweeps; a fire wall does.
   */
  earthShare: 0.35, slabShare: 0.3, slabSpeed: 6, slabY: 0, slabDuck: 14, bodyHalfW: 12,
  pillarWindupS: 1.4, stonePillarSpeed: 3.5, stonePillarHalfW: 20, pillarHeightStone: 70, pillarOffset: 22,
  /** Enemies stay within this fraction of the screen's half-width of its centre (easier to aim at). */
  enemyBand: 0.4,
  /**
   * Combos and charged punches.
   * - Charged punch (a fist pulled back and held): a blue fireball, chargedSpeed× faster and
   *   chargedRadius× bigger, doing chargedDamage.
   * - Flurry: flurryCount punches within flurryWindowS — the last is a big fireball
   *   (flurryRadius×, flurryDamage) that also burns enemies within flurrySplash of the one it hits.
   * - Counter: a punch within counterWindowS of the flame shield blocking something — it homes in
   *   (snaps to the nearest enemy) at counterSpeed×, doing counterDamage.
   * - One-two push: a palm push within oneTwoWindowS of two punches (the second within
   *   oneTwoGapS) — a pillar oneTwoWidth× wide doing oneTwoDamage.
   * - Pillar volley: a palm push with the other hand within volleyWindowS of one — the two merge into
   *   a wave volleyWidth× a pillar's width doing volleyDamage.
   * - Wall breaker: pushing both palms while your own fire wall stands sends it rolling forward.
   *   (Pushing both palms does nothing otherwise.)
   * - Finisher: the ultimate — open hands held together until they catch fire, then spread — when
   *   the ultimate bar is full.
   */
  chargedSpeed: 1.3, chargedRadius: 1.6, chargedDamage: 2,
  flurryCount: 3, flurryWindowS: 1, flurryRadius: 1.8, flurryDamage: 2, flurrySplash: 25,
  counterWindowS: 0.6, counterSpeed: 1.5, counterDamage: 2,
  oneTwoWindowS: 1.2, oneTwoGapS: 0.8, oneTwoWidth: 2, oneTwoDamage: 3,
  volleyWindowS: 0.6, volleyWidth: 3, volleyDamage: 3,
  /** Testing: the shield never drains or breaks. */
  shieldReach: 8,
  enemyHp: 2, enemyProjRadius: 4.2, windupS: 1, waveBreakS: 2.2,
};

export interface Enemy {
  id: number; x: number; y: number; z: number; hp: number;
  t: number; appear: number; dying: number; flash: number;
  cd: number; winding: boolean; wind: number; side: 1 | -1; phase: number;
  /** The attack it is winding up (picked when the wind-up starts). */
  attack?: AttackKind;
  /** Earthbender (stone pillars) instead of a water spirit. */
  earth?: boolean;
  /** Scripted (tutorial): always uses this attack. */
  only?: AttackKind;
  /** Scripted (tutorial): which of the lesson's enemies this is. */
  tag?: string;
  /** Scripted (tutorial): seconds between attacks. */
  pace?: number;
  /** Practice target: never moves or attacks, respawns in its slot. */
  dummy?: boolean;
  slot?: number;
  /** The chapter boss (Daro Stonefist). */
  boss?: BossState;
  /** Full health, for bosses' health bars. */
  maxHp?: number;
}

/** What kind of fireball a player's shot is (they look and hit differently). */
export type Shot = 'normal' | 'charged' | 'flurry' | 'counter';

export interface Proj {
  id: number; kind: 'player' | 'enemy';
  x: number; y: number; z: number; vx: number; vy: number; vz: number; r: number;
  resolved: boolean;
  /** Player shots: which kind, and how much it hurts (default normal, 1). */
  shot?: Shot;
  damage?: number;
}

export type ComboName = 'charged' | 'flurry' | 'counter' | 'oneTwo' | 'volley' | 'wallBreaker' | 'finisher';

/** Every move the player can have; the campaign unlocks them one scroll at a time. */
export type MoveName = 'punch' | 'flurry' | 'shield' | 'palm' | 'charge' | 'wall' | 'finisher'
  | 'xBlock' | 'counter' | 'oneTwo' | 'volley' | 'wallBreaker';

type PositionedType = 'blocked' | 'playerHit' | 'dodged' | 'hitEnemy' | 'killEnemy' | 'clash' | 'wall' | 'ultimate' | 'cut' | 'wallPush' | 'slab' | 'fizzle';
export type GameEvent =
  | { type: PositionedType; x: number; y: number; z: number }
  | { type: 'punch' | 'pillar'; x: number; y: number; z: number; side: Side }
  | { type: 'stonePillar'; x: number; y: number; z: number; side: 1 | -1 }
  | { type: 'combo'; name: ComboName; x: number; y: number; z: number; side?: Side }
  /** A move that didn't go off, and what it needs (e.g. the wall push needs a wall). */
  | { type: 'hint'; text: string }
  | { type: 'gameOver' }
  | { type: 'wave'; wave: number };

export type Rand = () => number;

/** The ultimate's spinning disc of fire: centred at the hands (world x, y), reach r in depth units. */
export interface Blade { id: number; x: number; y: number; r: number }

/** A palm push: a column of fire rolling forward from the floor, `hit` = enemies it already burned. */
export interface Pillar { id: number; x: number; z: number; vx: number; age: number; hit: number[]; hand: Side; halfW: number; damage: number }

export type AttackKind = 'orb' | 'slab' | 'pillar';

/** An attack on its way that you can only move out of: are you out of its way right now? */
export interface Incoming {
  kind: 'stonePillar' | 'slab';
  safe: boolean;
  /** Stone pillar: which way to move to get out of its lane (−1 left, 1 right). */
  away: 1 | -1;
  /** 0 = just started … 1 = arriving. */
  closeness: number;
}

/**
 * An attack you have to move out of, travelling toward you at depth z. A stone pillar rises out of
 * the ground in its lane at `laneX` (rise 0 → 1, `owner` while its earthbender raises it), then
 * slides straight down the lane to you; a slab (high sweep) crosses the whole field at height y.
 */
export interface Hazard {
  id: number; kind: 'stonePillar' | 'slab'; x: number; y: number; z: number; vz: number; resolved: boolean;
  laneX: number; side: 1 | -1; startX: number; startZ: number; rise: number; owner: number | null;
  halfW?: number; look?: 'boulder';
}

/** A wall of fire across the courtyard: standing (vz 0), or rolling forward (a wall push) burning what it passes. */
export interface Wall { id: number; x: number; z: number; halfW: number; life: number; vz: number; hit: number[]; maxZ?: number }

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
  shield = { on: false };
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
  /** Finisher: how gathered your open hands are, 0 → 1 (1 = spread them to cast). */
  gather = 0;
  /** Breath left for attacks (see TUNE.breathMax), when you last spent some, and the last "out of breath" hint. */
  breath: number = TUNE.breathMax;
  private breathSpentT = -Infinity;
  private breathHintT = -Infinity;
  /** Tests turn this off to control enemies by hand. */
  spawning = true;
  /** Dummies instead of attacking spirits. */
  practice = false;
  /** Tutorial: attacks still land (and show) but cost no health. */
  noDamage = false;
  /** The moves you have (campaign); null = every move (waves, training, tutorial). */
  allowed: Set<MoveName> | null = null;
  /** Shown instead of the wave number (e.g. "Tutorial"). */
  label: string | null = null;

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
  /** Game time, and what just happened, for combos. */
  private time = 0;
  private recentPunches: { t: number; hand: Side }[] = [];
  private lastFlurryT = -Infinity;
  private lastShieldBlockT = -Infinity;

  constructor(private rand: Rand = Math.random, public viewHalfW = 70, practice = false) {
    if (practice) this.setPractice(true);
    else this.startWave();
  }

  /** Do you have this move? */
  has(move: MoveName): boolean {
    return this.allowed === null || this.allowed.has(move);
  }

  /**
   * Hand the field over to a script (the tutorial): nothing spawns by itself, the field is cleared,
   * and you can't lose.
   */
  scripted(): void {
    this.spawning = false;
    this.practice = false;
    this.noDamage = true;
    this.clearField();
  }

  /** Remove every enemy and every attack in flight. */
  clearField(): void {
    this.enemies = [];
    this.projs = [];
    this.hazards = [];
    this.pillars = [];
    this.blades = [];
    this.walls = [];
  }

  /** Put an enemy on the field (scripts): a dummy, a water spirit or an earthbender. */
  addEnemy(o: { kind: 'dummy' | 'spirit' | 'earth'; x: number; z: number; only?: AttackKind; tag?: string; hp?: number; cd?: number; pace?: number }): Enemy {
    const e: Enemy = {
      id: this.nextId++, x: o.x, y: FLOOR_Y - 30, z: o.z, hp: o.hp ?? TUNE.enemyHp, t: 0, appear: 0, dying: 0, flash: 0,
      cd: o.kind === 'dummy' ? Infinity : o.cd ?? 1.5, winding: false, wind: 0, side: 1, phase: this.rnd(0, 6),
      dummy: o.kind === 'dummy' || undefined, earth: o.kind === 'earth' || undefined, only: o.only, tag: o.tag, pace: o.pace,
    };
    this.enemies.push(e);
    return e;
  }

  /** The boss on the field, if any. */
  get boss(): Enemy | null { return this.enemies.find(e => e.boss && e.hp > 0) ?? null; }

  /** Put Daro Stonefist on the field. */
  addBoss(x: number, z: number): Enemy {
    const e = this.addEnemy({ kind: 'earth', x, z, hp: BOSS.hp, cd: Infinity });
    e.boss = newBossState();
    e.maxHp = BOSS.hp;
    return e;
  }

  /** A coin toss from the game's own random source (bosses). */
  rngBool(): boolean { return this.rand() < 0.5; }

  /** A stone pillar raised and shoved at once down the lane at laneX (the boss's attacks). */
  sendPillar(e: Enemy, laneX: number, halfW?: number): void {
    const z = e.z - 0.6, side: 1 | -1 = laneX >= this.cam.x ? 1 : -1;
    this.hazards.push({ id: this.nextId++, kind: 'stonePillar', x: laneX, y: FLOOR_Y, z, vz: -TUNE.stonePillarSpeed, resolved: false, laneX, side, startX: laneX, startZ: z, rise: 1, owner: null, halfW });
    this.events.push({ type: 'stonePillar', x: laneX, y: FLOOR_Y, z, side });
  }

  /** A sweep at head height (a boulder, for the boss). */
  sendSlab(e: Enemy, look?: 'boulder'): void {
    const z = e.z - 0.1;
    this.hazards.push({ id: this.nextId++, kind: 'slab', x: e.x, y: TUNE.slabY, z, vz: -TUNE.slabSpeed, resolved: false, laneX: 0, side: 1, startX: e.x, startZ: z, rise: 1, owner: null, look });
    this.emit('slab', e.x, TUNE.slabY, z);
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
    this.xBlock = intent.xBlock && this.has('xBlock');
    if (intent.shoulders) this.shoulders = intent.shoulders;
    this.inv = Math.max(0, this.inv - dt);
    this.punchCool = { l: Math.max(0, this.punchCool.l - dt), r: Math.max(0, this.punchCool.r - dt) };
    this.wallCool = Math.max(0, this.wallCool - dt);
    this.pushWallCool = Math.max(0, this.pushWallCool - dt);
    this.time += dt;
    this.palmCool = { l: Math.max(0, this.palmCool.l - dt), r: Math.max(0, this.palmCool.r - dt) };
    this.ultimateIn = Math.max(0, this.ultimateIn - dt);
    const rested = this.time - this.breathSpentT >= TUNE.breathRestS;
    this.breath = Math.min(TUNE.breathMax, this.breath + (rested ? TUNE.breathRestRegen : TUNE.breathRegen) * dt);
    this.updateShield(intent.shield && this.has('shield'));
    this.gather = this.has('finisher') ? intent.gather ?? 0 : 0;
    if (this.state !== 'play') return;
    if (this.has('punch')) for (const p of intent.punches) this.punch(this.has('charge') ? p : { ...p, charged: false });
    for (const c of intent.casts) {
      const needs: MoveName = c.kind === 'wall' ? 'wall' : c.kind === 'push' ? 'wallBreaker' : 'finisher';
      if (this.has(needs)) this.cast(c);
    }
    if (this.has('palm')) for (const p of intent.palms ?? []) this.palm(p);
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

  /** The shield is up whenever both open hands are held up for it: no meter, it never breaks. */
  private updateShield(wanted: boolean): void {
    this.shield.on = wanted && !!this.hands.l && !!this.hands.r;
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

  /** Spend breath on an attack at view point `at`; without enough, it fizzles (a puff of smoke) and returns false. */
  private spend(cost: number, at: Vec2): boolean {
    if (this.breath >= cost) {
      this.breath -= cost;
      this.breathSpentT = this.time;
      return true;
    }
    const w = this.handWorld(at);
    this.emit('fizzle', w.x, w.y, TUNE.launchZ);
    if (this.time - this.breathHintT > 1.5) {
      this.breathHintT = this.time;
      this.events.push({ type: 'hint', text: 'OUT OF BREATH — ease off a moment' });
    }
    return false;
  }

  private punch(p: Punch): void {
    if (this.punchCool[p.hand] > 0) return;
    if (!this.spend(p.charged ? TUNE.breathCharged : TUNE.breathPunch, p.at)) { this.punchCool[p.hand] = TUNE.punchCooldownS; return; }
    this.punchCool[p.hand] = TUNE.punchCooldownS;
    const now = this.time;
    this.recentPunches = [...this.recentPunches.filter(x => now - x.t <= Math.max(TUNE.flurryWindowS, TUNE.oneTwoWindowS)), { t: now, hand: p.hand }];
    const start = this.handWorld(p.at);
    let { point: target, depth } = this.aimFor(p.at, p.shoulder, p.dir);
    // what kind of shot: a charged fist, the end of a flurry, or a counter just after blocking
    const flurry = this.has('flurry') && this.recentPunches.filter(x => x.t > this.lastFlurryT && now - x.t <= TUNE.flurryWindowS).length >= TUNE.flurryCount;
    const counter = this.has('counter') && now - this.lastShieldBlockT <= TUNE.counterWindowS;
    const shot: Shot = p.charged ? 'charged' : counter ? 'counter' : flurry ? 'flurry' : 'normal';
    if (flurry) this.lastFlurryT = now;
    if (counter) this.lastShieldBlockT = -Infinity;
    if (shot === 'counter') {
      // homes in on the nearest enemy
      const e = this.nearestEnemy();
      if (e) { target = { x: e.x, y: e.y }; depth = e.z; }
    } else if (shot === 'charged') {
      // thrown from the chest, where the fist's distance reads worst: don't trust where it points —
      // go for the enemy it's aimed at, else the nearest, else straight ahead
      const aimed = this.aimFor(p.at, p.shoulder, p.dir).target;
      const e = aimed ?? this.nearestEnemy();
      if (e) { target = { x: e.x, y: e.y }; depth = e.z; }
      else { target = { x: this.cam.x, y: this.cam.y + 10 }; depth = TUNE.aimDepth; }
    }
    const speed = TUNE.fireballSpeed * (shot === 'charged' ? TUNE.chargedSpeed : shot === 'counter' ? TUNE.counterSpeed : 1);
    const r = TUNE.fireballRadius * (shot === 'charged' ? TUNE.chargedRadius : shot === 'flurry' ? TUNE.flurryRadius : 1);
    const damage = shot === 'charged' ? TUNE.chargedDamage : shot === 'flurry' ? TUNE.flurryDamage : shot === 'counter' ? TUNE.counterDamage : 1;
    const T = (depth - TUNE.launchZ) / speed;
    this.projs.push({
      id: this.nextId++, kind: 'player', x: start.x, y: start.y, z: TUNE.launchZ,
      vx: (target.x - start.x) / T, vy: (target.y - start.y) / T, vz: speed, r, resolved: false, shot, damage,
    });
    this.events.push({ type: 'punch', x: start.x, y: start.y, z: TUNE.launchZ, side: p.hand });
    if (shot !== 'normal') this.events.push({ type: 'combo', name: shot, x: start.x, y: start.y, z: TUNE.launchZ, side: p.hand });
  }

  /** The living enemy nearest you (by depth, then sideways). */
  private nearestEnemy(): Enemy | null {
    let best: Enemy | null = null;
    for (const e of this.enemies) if (e.hp > 0 && (!best || e.z + Math.abs(e.x - this.cam.x) / 30 < best.z + Math.abs(best.x - this.cam.x) / 30)) best = e;
    return best;
  }

  private palm(p: Palm): void {
    if (this.palmCool[p.hand] > 0) return;
    if (!this.spend(TUNE.breathPalm, p.at)) { this.palmCool[p.hand] = TUNE.palmCooldownS; return; }
    this.palmCool[p.hand] = TUNE.palmCooldownS;
    const now = this.time, start = this.handWorld(p.at).x, z = TUNE.launchZ;
    this.events.push({ type: 'pillar', x: start, y: FLOOR_Y, z, side: p.hand });
    // pillar volley: the other hand's pillar went out just now — the two merge into a wave
    const partner = !this.has('volley') ? undefined : this.pillars.find(c => c.hand !== p.hand && c.age <= TUNE.volleyWindowS && c.halfW < TUNE.pillarHalfW * TUNE.volleyWidth);
    if (partner) {
      partner.x = (partner.x + start) / 2;
      partner.halfW = TUNE.pillarHalfW * TUNE.volleyWidth;
      partner.damage = TUNE.volleyDamage;
      partner.hit = [];
      this.events.push({ type: 'combo', name: 'volley', x: partner.x, y: FLOOR_Y, z: partner.z, side: p.hand });
      return;
    }
    // one-two push: two punches just before
    const jabs = this.recentPunches.filter(x => now - x.t <= TUNE.oneTwoWindowS);
    const oneTwo = this.has('oneTwo') && jabs.length >= 2 && now - jabs[jabs.length - 1].t <= TUNE.oneTwoGapS;
    if (oneTwo) this.recentPunches = [];
    // rolls from in front of the hand toward where the palm points
    const { point, depth } = this.aimFor(p.at, p.shoulder, p.dir);
    const vx = ((point.x - start) / Math.max(1, depth - z)) * TUNE.pillarSpeed;
    this.pillars.push({
      id: this.nextId++, x: start, z, vx, age: 0, hit: [], hand: p.hand,
      halfW: TUNE.pillarHalfW * (oneTwo ? TUNE.oneTwoWidth : 1), damage: oneTwo ? TUNE.oneTwoDamage : TUNE.palmDamage,
    });
    if (oneTwo) this.events.push({ type: 'combo', name: 'oneTwo', x: start, y: FLOOR_Y, z, side: p.hand });
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
        this.burn(e, TUNE.palmDamage, 'wall');
      }
    }
    this.walls = this.walls.filter(w => w.life > 0 && w.z < (w.maxZ ?? TUNE.pillarMaxZ));
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
      // a wall stops it if it stands between the attack and you
      const wall = this.wallMet(h.kind === 'stonePillar' ? h.x : this.cam.x, h.kind === 'stonePillar' ? h.halfW ?? TUNE.stonePillarHalfW : 0, z0, h.z, dt);
      if (wall) {
        dead.add(h.id);
        this.score += 15;
        this.emit('blocked', h.x, h.kind === 'stonePillar' ? FLOOR_Y - 20 : h.y, wall.z);
        continue;
      }
      if (!h.resolved && h.z <= 0.2) {
        h.resolved = true;
        const hit = !this.outOfWay(h);
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

  /** Out of this attack's way where you stand now: beside a pillar's lane, or ducked under a sweep. */
  private outOfWay(h: Hazard): boolean {
    return h.kind === 'stonePillar'
      ? Math.abs(this.cam.x - h.laneX) >= (h.halfW ?? TUNE.stonePillarHalfW) + TUNE.bodyHalfW
      : this.cam.y - h.y >= TUNE.slabDuck;
  }

  /** Attacks on their way you have to move out of, nearest first, and whether you are clear of each. */
  incoming(): Incoming[] {
    return this.hazards
      .filter(h => !h.resolved)
      .sort((a, b) => (a.owner !== null ? 1 : 0) - (b.owner !== null ? 1 : 0) || a.z - b.z)
      .map(h => ({
        kind: h.kind,
        safe: this.outOfWay(h),
        away: (h.laneX > this.cam.x ? -1 : 1) as 1 | -1,
        closeness: h.owner !== null ? 0 : Math.max(0, Math.min(1, 1 - h.z / h.startZ)),
      }));
  }

  private hurt(x: number, y: number): void {
    if (!this.noDamage) this.hp = Math.max(0, this.hp - TUNE.hitDamage);
    this.inv = TUNE.invulnS;
    this.emit('playerHit', x, y, 0);
    if (this.hp <= 0) {
      this.state = 'over';
      this.events.push({ type: 'gameOver' });
    }
  }

  /** Burn an enemy with a pillar (or a rolling wall, `shot: 'wall'`). */
  private burn(e: Enemy, damage = TUNE.palmDamage, shot: 'pillar' | 'wall' = 'pillar'): void {
    const dealt = e.boss ? bossDamage(e, shot, damage) : damage;
    e.hp -= dealt;
    e.flash = 1;
    if (dealt === 0) this.emit('blocked', e.x, e.y, e.z);
    else if (e.hp <= 0) { this.score += 100; this.emit('killEnemy', e.x, e.y, e.z); }
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
        if (e.hp <= 0 || c.hit.includes(e.id) || e.z < z0 - 0.5 || e.z > c.z + 0.5 || Math.abs(e.x - c.x) > c.halfW + 7) continue;
        c.hit.push(e.id);
        this.burn(e, c.damage);
      }
      this.projs = this.projs.filter(p => {
        if (p.kind !== 'enemy' || Math.abs(p.z - c.z) > 0.8 || Math.abs(p.x - c.x) > c.halfW + p.r) return true;
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
      if (!this.spend(TUNE.breathWall, c.at)) return;
      // stand the wall where the hands appear on screen, at its depth
      const x = this.cam.x + c.at.x / depthScale(TUNE.wallDepth);
      this.walls.push({ id: this.nextId++, x, z: TUNE.wallDepth, halfW: TUNE.wallHalfWidth, life: TUNE.wallLifeS, vz: 0, hit: [] });
      this.emit('wall', x, FLOOR_Y, TUNE.wallDepth);
      return;
    }
    if (c.kind === 'push') {
      // wall breaker: your standing fire wall rolls forward
      const wall = this.walls.find(w => w.vz === 0);
      if (wall) {
        if (!this.spend(TUNE.breathWallPush, c.at)) return;
        wall.vz = TUNE.pushWallSpeed;
        wall.life = Infinity;
        wall.hit = [];
        this.emit('wallPush', wall.x, FLOOR_Y, wall.z);
        this.events.push({ type: 'combo', name: 'wallBreaker', x: wall.x, y: FLOOR_Y, z: wall.z });
        return;
      }
      this.events.push({ type: 'hint', text: 'WALL PUSH: RAISE A FIRE WALL FIRST' });
      return;
    }
    if (this.ultimateIn > 0) {
      this.events.push({ type: 'hint', text: `FINISHER RECHARGING — ${Math.ceil(this.ultimateIn)}s` });
      return;
    }
    this.ultimateIn = TUNE.ultimateCooldownS;
    const at = this.handWorld(c.at);
    const y = at.y + (FLOOR_Y - at.y) * TUNE.bladeDrop;
    this.blades.push({ id: this.nextId++, x: at.x, y, r: 0 });
    this.emit('ultimate', at.x, at.y, 0);
    this.events.push({ type: 'combo', name: 'finisher', x: at.x, y: at.y, z: 0 });
  }

  /** Grow each blade and cut down whatever its edge has reached. */
  private updateBlades(dt: number): void {
    const reach = (b: Blade, x: number, z: number) => Math.hypot((x - b.x) / TUNE.bladeWidthPerDepth, z) <= b.r;
    for (const b of this.blades) {
      b.r += TUNE.bladeSpeed * dt;
      for (const e of this.enemies) {
        if (e.hp <= 0 || !reach(b, e.x, e.z)) continue;
        if (e.boss) { e.hp -= bossDamage(e, 'blade', 10); e.flash = 1; if (e.hp <= 0) { this.score += 100; this.emit('killEnemy', e.x, e.y, e.z); } else this.emit('hitEnemy', e.x, e.y, e.z); continue; }
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
      if (e.boss) { updateBoss(this, e, dt); continue; }
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
          e.cd = e.pace ?? this.rnd(baseCd, baseCd + 1.5);
          e.side = e.side === 1 ? -1 : 1;
        }
      }
    }
  }

  private pickAttack(e: Enemy): AttackKind {
    if (e.only) return e.only;
    if (e.earth) return 'pillar';
    return this.rand() < TUNE.slabShare ? 'slab' : 'orb';
  }

  /** An earthbender stomps: a stone pillar starts rising out of the ground in a lane to one side of you. */
  private raisePillar(e: Enemy): void {
    const side: 1 | -1 = this.rand() < 0.5 ? -1 : 1, z = e.z - 0.6, x = this.cam.x + side * TUNE.pillarOffset;
    this.hazards.push({
      id: this.nextId++, kind: 'stonePillar', x, y: FLOOR_Y, z, vz: 0, resolved: false,
      laneX: x, side, startX: x, startZ: z, rise: 0, owner: e.id,
    });
  }

  private enemyAttack(e: Enemy): void {
    const kind = e.attack ?? 'orb';
    if (kind === 'orb') { this.enemyThrow(e); return; }
    if (kind === 'pillar') {
      // shove the raised pillar down its lane
      const h = this.hazards.find(x => x.owner === e.id);
      if (!h) return;
      h.owner = null;
      h.rise = 1;
      h.vz = -TUNE.stonePillarSpeed;
      this.events.push({ type: 'stonePillar', x: h.x, y: FLOOR_Y, z: h.z, side: h.side });
      return;
    }
    const z = e.z - 0.1;
    this.hazards.push({ id: this.nextId++, kind, x: e.x, y: TUNE.slabY, z, vz: -TUNE.slabSpeed, resolved: false, laneX: 0, side: 1, startX: e.x, startZ: z, rise: 1, owner: null });
    this.emit('slab', e.x, TUNE.slabY, z);
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
            const dealt = e.boss ? bossDamage(e, p.shot ?? 'normal', p.damage ?? 1) : p.damage ?? 1;
            e.hp -= dealt;
            e.flash = 1;
            dead.add(p.id);
            if (dealt === 0) this.emit('blocked', p.x, p.y, p.z);
            else if (e.hp <= 0) { this.score += 100; this.emit('killEnemy', p.x, p.y, p.z); }
            else this.emit('hitEnemy', p.x, p.y, p.z);
            // a flurry's big fireball bursts, burning those standing nearby
            if (p.shot === 'flurry') {
              for (const o of this.enemies) {
                if (o !== e && o.hp > 0 && Math.abs(o.x - e.x) <= TUNE.flurrySplash && Math.abs(o.z - e.z) <= 2) this.burn(o, 1);
              }
            }
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
      this.lastShieldBlockT = this.time;
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
