import type { Enemy, Game } from './game';

/**
 * Daro Stonefist. `hp` is his health; phases change at phase2At / phase3At of it. He attacks
 * every attackEveryS[phase-1] seconds; in phase 1 and 2 he raises a stone wall (wallHp breaking
 * hits) every wallEveryS; in phase 3, after each attack, he's winded for windedS (hits double).
 * Twin pillars come down lanes twinOffset either side of you, twinHalfW wide.
 */
export const BOSS = { hp: 40, phase2At: 0.6, phase3At: 0.25, wallHp: 1, wallEveryS: 7, attackEveryS: [3, 2.4, 1.9], windedS: 2, twinOffset: 36, twinHalfW: 16 };

export interface BossState {
  phase: 1 | 2 | 3;
  /** Stone wall hits left (0 = none up). */
  wall: number;
  wallT: number;
  attackT: number;
  /** Seconds left winded. */
  winded: number;
  spiritT: number;
  /** Alternates which side single pillars come down. */
  side: 1 | -1;
}

export function newBossState(): BossState {
  return { phase: 1, wall: 0, wallT: BOSS.wallEveryS * 0.5, attackT: 2, winded: 0, spiritT: 6, side: 1 };
}

export function bossPhase(e: { hp: number; maxHp?: number }): 1 | 2 | 3 {
  const k = e.hp / (e.maxHp ?? BOSS.hp);
  return k > BOSS.phase2At ? 1 : k > BOSS.phase3At ? 2 : 3;
}

/** His turn: walls, attacks, spirits joining, being winded. */
export function updateBoss(g: Game, e: Enemy, dt: number): void {
  const b = e.boss!;
  b.phase = bossPhase(e);
  b.winded = Math.max(0, b.winded - dt);
  if (b.winded > 0) return;
  if (b.phase < 3) {
    b.wallT -= dt;
    if (b.wallT <= 0 && b.wall === 0) { b.wall = BOSS.wallHp; b.wallT = BOSS.wallEveryS; }
  }
  if (b.phase >= 2) {
    b.spiritT -= dt;
    if (b.spiritT <= 0 && g.enemies.filter(x => !x.boss && x.hp > 0).length < 2) {
      g.addEnemy({ kind: 'spirit', x: e.x + b.side * 25, z: e.z + 1, only: 'orb', pace: 2.2, cd: 1.5 });
      b.spiritT = 9;
    }
  }
  b.attackT -= dt;
  if (b.attackT > 0) return;
  b.attackT = BOSS.attackEveryS[b.phase - 1];
  if (b.phase === 1) {
    g.sendPillar(e, g.cam.x + b.side * 22);
    b.side = b.side === 1 ? -1 : 1;
  } else if (g.rngBool()) {
    g.sendPillar(e, g.cam.x - BOSS.twinOffset, BOSS.twinHalfW);
    g.sendPillar(e, g.cam.x + BOSS.twinOffset, BOSS.twinHalfW);
  } else {
    g.sendSlab(e, 'boulder');
  }
  if (b.phase === 3) b.winded = BOSS.windedS;
}

/**
 * How much a hit does to him: the stone wall stops fireballs (a charged punch or pillar breaks it
 * instead); winded, everything lands double.
 */
export function bossDamage(e: Enemy, shot: 'normal' | 'charged' | 'flurry' | 'counter' | 'pillar' | 'wall' | 'blade', dmg: number): number {
  const b = e.boss!;
  if (b.wall > 0 && shot !== 'blade') {
    if (shot === 'charged' || shot === 'pillar' || shot === 'wall') b.wall = Math.max(0, b.wall - 1);
    return 0;
  }
  return b.winded > 0 ? dmg * 2 : dmg;
}
