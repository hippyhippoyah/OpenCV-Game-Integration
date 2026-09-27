import { describe, expect, it } from 'vitest';
import { Game, TUNE } from './game';
import { BOSS, bossPhase } from './boss';
import { mulberry32 } from '../math';

const idle = { present: true, head: { x: 0, y: 0 }, hands: { l: null, r: null }, shoulders: { l: { x: -20, y: 20 }, r: { x: 20, y: 20 } }, punches: [], palms: [], shield: false, xBlock: false, casts: [], face: null, bodyTilt: 0 };
const arena = () => { const g = new Game(mulberry32(2), 70, true); g.scripted(); g.noDamage = false; return g; };
const run = (g: Game, s: number, i = idle) => { for (let t = 0; t < s; t += 1 / 60) g.step(1 / 60, i); };
const punchAt = (x: number, y: number, charged = false) => ({ ...idle, punches: [{ hand: 'r' as const, at: { x, y }, shoulder: { x: 20, y: 20 }, dir: null, charged }] });

describe('Daro Stonefist', () => {
  it('has phases at 60% and 25% health', () => {
    expect(bossPhase({ hp: 40, maxHp: 40 })).toBe(1);
    expect(bossPhase({ hp: 23, maxHp: 40 })).toBe(2);
    expect(bossPhase({ hp: 9, maxHp: 40 })).toBe(3);
  });

  it('attacks with pillars in phase 1', () => {
    const g = arena();
    g.addBoss(0, 9);
    run(g, BOSS.attackEveryS[0] + 1.6);
    expect(g.hazards.some(h => h.kind === 'stonePillar')).toBe(true);
  });

  it('his stone wall stops fireballs, but a charged punch breaks it', () => {
    const g = arena(), b = g.addBoss(0, 9);
    b.boss!.wall = BOSS.wallHp;
    const s = 3 / (3 + 9), at = { x: 0, y: (b.y) * s };
    g.step(1 / 60, punchAt(at.x, at.y));
    run(g, 1.2);
    expect(b.hp).toBe(BOSS.hp);
    expect(b.boss!.wall).toBe(BOSS.wallHp);
    g.step(1 / 60, punchAt(at.x, at.y, true));
    run(g, 1.2);
    expect(b.boss!.wall).toBe(0);
  });

  it('twin pillars in phase 2 leave the middle safe', () => {
    const g = arena(), b = g.addBoss(0, 9);
    b.hp = Math.floor(BOSS.hp * 0.5);
    for (let i = 0; i < 20 && !g.hazards.some(h => h.kind === 'stonePillar' && h.halfW === BOSS.twinHalfW); i++) run(g, 0.5);
    const twins = g.hazards.filter(h => h.kind === 'stonePillar');
    expect(twins.map(h => Math.sign(h.laneX)).sort()).toEqual([-1, 1]);
    // standing in the middle is out of both lanes (boulders and spirits may still come; not pillars)
    expect(g.incoming().filter(i => i.kind === 'stonePillar').every(i => i.safe)).toBe(true);
    expect(TUNE.maxHp).toBeGreaterThan(0);
  });

  it('winded in phase 3: hits land double', () => {
    const g = arena(), b = g.addBoss(0, 9);
    b.hp = 8;
    b.boss!.winded = BOSS.windedS;
    b.boss!.wall = 0;
    const s = 3 / (3 + 9);
    g.step(1 / 60, punchAt(0, b.y * s));
    run(g, 1.2);
    expect(b.hp).toBe(6);
  });
});
