# Campaign Chapter 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Campaign Chapter 1 — auto-walking a three.js mountain path, finding scrolls that unlock moves, and webcam fights (with ghost-hand practice) ending in the boss Daro Stonefist.

**Architecture:** Pure-logic modules (move gating in `Game`, progress, fight scripts, boss, chapter data, the campaign state machine, the path) are built test-first. Rendering is split: the explore half is a new three.js scene on its own canvas; fights reuse the existing 2D `Renderer` with ghost hands added. `main.ts` gains a `campaign` phase that switches between the two halves and their inputs.

**Tech Stack:** TypeScript 5.9, Vite 8, Vitest 5 (node env, no DOM in tested modules), three.js (new dependency), MediaPipe (existing).

**Spec:** `docs/superpowers/specs/2026-09-27-campaign-chapter-1-design.md`

## Global Constraints

- All work under `web/`; run commands from `web/`. Never stage or commit `../socket_pose.py`.
- Tested modules must not touch the DOM or WebGL (Vitest runs in `node`).
- Match existing style: 2-space indent, single quotes, JSDoc on exported things, comment density like `src/game/game.ts`.
- `localStorage` access always wrapped in try/catch; the game must work without it.
- Waves, Training and the Tutorial behave exactly as before (every move allowed there).
- Not in the campaign: X block, shield counter, one-two push, pillar volley, wall breaker.
- Commit after each task with a message ending in the attribution line:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- `npx tsc --noEmit -p .` and `npx vitest run` must pass at the end of every task.

## File Structure

| File | Responsibility |
|---|---|
| `src/game/game.ts` (modify) | `MoveName`, `Game.allowed` gating; boss hooks; per-hazard width and boulder look |
| `src/game/boss.ts` (create) | Daro Stonefist: phases, stone wall, twin pillars, boulder, winded |
| `src/game/tutorial.ts` (modify) | `Tutorial` takes a lesson list (so a campaign practice can run one lesson) |
| `src/campaign/progress.ts` (create) | Scrolls found, stops done, flames; load/save through a storage adapter |
| `src/campaign/scripts.ts` (create) | Fight scripts (timed enemy groups, goals) and `FightRunner` |
| `src/campaign/chapter1.ts` (create) | Chapter data: scrolls, stops, Ren's lines, path stops, practices, fights |
| `src/campaign/runner.ts` (create) | The campaign state machine (walk ⇄ fight) |
| `src/explore/path.ts` (create) | Rail along the path: distance, pose, pauses, skip; mouse-look limits |
| `src/explore/world3d.ts` (create) | three.js scene: terrain, places, lights, fog, sky, particles, scrolls, arenas |
| `src/render/ghost.ts` (create) | Ghost-hand keyframes per move (pure) |
| `src/render/renderer.ts` (modify) | Draw ghost hands; boss and its health, stone wall, boulder look; place tint |
| `src/campaign/ui.ts` (create) | DOM: controls strip, notes, prompt, Ren line, handoff card, countdown, result, map, scrolls, boss bar |
| `index.html`, `src/style.css` (modify) | Markup and styles for the above; `#world` canvas |
| `src/main.ts` (modify) | Campaign phase, input switching, pointer lock, menu entry |

---

### Task 1: Move gating in the game

**Files:**
- Modify: `src/game/game.ts`
- Test: `src/game/game.test.ts`

**Interfaces:**
- Produces: `export type MoveName = 'punch' | 'flurry' | 'shield' | 'palm' | 'charge' | 'wall' | 'finisher' | 'xBlock' | 'counter' | 'oneTwo' | 'volley' | 'wallBreaker';` and `Game.allowed: Set<MoveName> | null` (null = every move, the default).

- [ ] **Step 1: Write the failing tests** — append inside `describe('Game', …)` in `src/game/game.test.ts`:

```ts
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
```

- [ ] **Step 2: Run to see them fail**

Run: `npx vitest run src/game/game.test.ts -t "moves you have"`
Expected: FAIL (`allowed` is undefined; locked moves still happen).

- [ ] **Step 3: Implement** in `src/game/game.ts`:

Add after `export type ComboName = …`:

```ts
/** Every move the player can have; the campaign unlocks them one scroll at a time. */
export type MoveName = 'punch' | 'flurry' | 'shield' | 'palm' | 'charge' | 'wall' | 'finisher'
  | 'xBlock' | 'counter' | 'oneTwo' | 'volley' | 'wallBreaker';
```

Add a field next to `noDamage`:

```ts
  /** The moves you have (campaign); null = every move (waves, training, tutorial). */
  allowed: Set<MoveName> | null = null;

  /** Do you have this move? */
  has(move: MoveName): boolean {
    return this.allowed === null || this.allowed.has(move);
  }
```

In `step()`, replace the lines that read `this.xBlock = intent.xBlock;` and `this.updateShield(dt, intent.shield);` and the three `for (const … of intent.…)` loops with:

```ts
    this.xBlock = intent.xBlock && this.has('xBlock');
```
```ts
    this.updateShield(dt, intent.shield && this.has('shield'));
```
```ts
    if (this.has('punch')) for (const p of intent.punches) this.punch(this.has('charge') ? p : { ...p, charged: false });
    for (const c of intent.casts) {
      const needs: MoveName = c.kind === 'wall' ? 'wall' : c.kind === 'push' ? 'wallBreaker' : 'finisher';
      if (this.has(needs)) this.cast(c);
    }
    if (this.has('palm')) for (const p of intent.palms ?? []) this.palm(p);
```

In `punch()`, gate the combos: change

```ts
    const flurry = this.recentPunches.filter(x => x.t > this.lastFlurryT && now - x.t <= TUNE.flurryWindowS).length >= TUNE.flurryCount;
    const counter = now - this.lastShieldBlockT <= TUNE.counterWindowS;
```
to
```ts
    const flurry = this.has('flurry') && this.recentPunches.filter(x => x.t > this.lastFlurryT && now - x.t <= TUNE.flurryWindowS).length >= TUNE.flurryCount;
    const counter = this.has('counter') && now - this.lastShieldBlockT <= TUNE.counterWindowS;
```

In `palm()`, gate volley and one-two: change `const partner = this.pillars.find(` to `const partner = !this.has('volley') ? undefined : this.pillars.find(`, and `const oneTwo = jabs.length >= 2 && …` to `const oneTwo = this.has('oneTwo') && jabs.length >= 2 && now - jabs[jabs.length - 1].t <= TUNE.oneTwoGapS;`.

- [ ] **Step 4: Run all tests**

Run: `npx vitest run && npx tsc --noEmit -p .`
Expected: all pass (existing tests keep `allowed = null`).

- [ ] **Step 5: Commit**

```bash
git add src/game/game.ts src/game/game.test.ts
git commit -m "feat(web): moves can be locked (Game.allowed) for the campaign

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Campaign progress (scrolls, stops, flames)

**Files:**
- Create: `src/campaign/progress.ts`
- Test: `src/campaign/progress.test.ts`

**Interfaces:**
- Consumes: `MoveName` (Task 1).
- Produces:
```ts
export type ScrollId = 'flameShield' | 'risingPillar' | 'heldBreath' | 'burningWall' | 'finalFlame';
export interface StoreLike { getItem(k: string): string | null; setItem(k: string, v: string): void }
export interface ProgressData { version: 1; scrolls: ScrollId[]; stops: Record<string, { flames: number }>; chapterDone: boolean }
export class Progress {
  static load(store: StoreLike | null): Progress;
  readonly data: ProgressData;
  hasScroll(id: ScrollId): boolean;
  addScroll(id: ScrollId): boolean;          // true if new
  completeStop(stopId: string, flames: number): void; // keeps the best
  flames(stopId: string): number;
  isDone(stopId: string): boolean;
  finishChapter(): void;
  save(): void;
  reset(): void;
}
export const PROGRESS_KEY = 'firebending.campaign.v1';
```

- [ ] **Step 1: Write the failing test** `src/campaign/progress.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { Progress, PROGRESS_KEY, type StoreLike } from './progress';

const memory = (): StoreLike & { data: Record<string, string> } => {
  const data: Record<string, string> = {};
  return { data, getItem: k => data[k] ?? null, setItem: (k, v) => { data[k] = v; } };
};

describe('Progress', () => {
  it('starts empty', () => {
    const p = Progress.load(memory());
    expect(p.data.scrolls).toEqual([]);
    expect(p.isDone('courtyard')).toBe(false);
  });

  it('remembers scrolls, stops (best flames) and the chapter across loads', () => {
    const store = memory();
    const p = Progress.load(store);
    expect(p.addScroll('flameShield')).toBe(true);
    expect(p.addScroll('flameShield')).toBe(false);
    p.completeStop('courtyard', 2);
    p.completeStop('courtyard', 1);
    p.finishChapter();
    p.save();
    const q = Progress.load(store);
    expect(q.hasScroll('flameShield')).toBe(true);
    expect(q.flames('courtyard')).toBe(2);
    expect(q.data.chapterDone).toBe(true);
  });

  it('starts fresh when storage is missing, broken or throws', () => {
    expect(Progress.load(null).data.scrolls).toEqual([]);
    const bad = memory();
    bad.data[PROGRESS_KEY] = '{not json';
    expect(Progress.load(bad).data.scrolls).toEqual([]);
    const throwing: StoreLike = { getItem: () => { throw new Error('denied'); }, setItem: () => { throw new Error('denied'); } };
    const p = Progress.load(throwing);
    p.addScroll('heldBreath');
    expect(() => p.save()).not.toThrow();
  });

  it('can be reset', () => {
    const store = memory();
    const p = Progress.load(store);
    p.addScroll('flameShield');
    p.reset();
    expect(Progress.load(store).data.scrolls).toEqual([]);
  });
});
```

- [ ] **Step 2: Run to see it fail**

Run: `npx vitest run src/campaign/progress.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `src/campaign/progress.ts`:

```ts
/** Scrolls are moves, found along the campaign path. */
export type ScrollId = 'flameShield' | 'risingPillar' | 'heldBreath' | 'burningWall' | 'finalFlame';

/** The bit of `localStorage` progress needs (tests pass an in-memory one). */
export interface StoreLike { getItem(k: string): string | null; setItem(k: string, v: string): void }

export interface ProgressData {
  version: 1;
  scrolls: ScrollId[];
  /** Finished stops and the most flames earned there. */
  stops: Record<string, { flames: number }>;
  chapterDone: boolean;
}

export const PROGRESS_KEY = 'firebending.campaign.v1';

const fresh = (): ProgressData => ({ version: 1, scrolls: [], stops: {}, chapterDone: false });

/** Campaign progress, kept on this device. Storage failing never breaks the game. */
export class Progress {
  private constructor(private store: StoreLike | null, public data: ProgressData) {}

  static load(store: StoreLike | null): Progress {
    try {
      const raw = store?.getItem(PROGRESS_KEY);
      const d = raw ? (JSON.parse(raw) as ProgressData) : null;
      if (d && d.version === 1 && Array.isArray(d.scrolls) && d.stops) return new Progress(store, d);
    } catch { /* unreadable: start fresh */ }
    return new Progress(store, fresh());
  }

  hasScroll(id: ScrollId): boolean { return this.data.scrolls.includes(id); }

  /** Returns true if it's a new scroll. */
  addScroll(id: ScrollId): boolean {
    if (this.hasScroll(id)) return false;
    this.data.scrolls.push(id);
    return true;
  }

  /** A stop finished with `flames` (0–3); the best is kept. */
  completeStop(stopId: string, flames: number): void {
    this.data.stops[stopId] = { flames: Math.max(flames, this.data.stops[stopId]?.flames ?? 0) };
  }

  flames(stopId: string): number { return this.data.stops[stopId]?.flames ?? 0; }
  isDone(stopId: string): boolean { return stopId in this.data.stops; }
  finishChapter(): void { this.data.chapterDone = true; }

  save(): void {
    try { this.store?.setItem(PROGRESS_KEY, JSON.stringify(this.data)); } catch { /* storage full or blocked */ }
  }

  reset(): void {
    this.data = fresh();
    this.save();
  }
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run src/campaign/progress.test.ts && npx tsc --noEmit -p .`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/campaign/progress.ts src/campaign/progress.test.ts
git commit -m "feat(web): campaign progress (scrolls, stops, flames) saved on the device

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Fight scripts and the fight runner

**Files:**
- Create: `src/campaign/scripts.ts`
- Test: `src/campaign/scripts.test.ts`

**Interfaces:**
- Consumes: `Game.addEnemy`, `Game.scripted`, `Game.clearField`, `Game.noDamage`, `Game.state`, `Game.enemies`, `AttackKind` (existing).
- Produces:
```ts
export interface ScriptEnemy { kind: 'dummy' | 'spirit' | 'earth'; x: number; z: number; only?: AttackKind; pace?: number; cd?: number; hp?: number }
export interface Group { at: number; enemies: ScriptEnemy[] }
export type Goal = { type: 'defeat' } | { type: 'survive'; seconds: number };
export interface FightScript { groups: Group[]; goal: Goal; boss?: boolean }
export type FightOutcome = 'fighting' | 'won' | 'lost';
export class FightRunner {
  constructor(g: Game, script: FightScript);
  readonly elapsed: number;
  update(dt: number): FightOutcome;
  /** 0..1 for the HUD: groups defeated or time survived. */
  get progress(): number;
}
```

- [ ] **Step 1: Write the failing test** `src/campaign/scripts.test.ts`:

```ts
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
```

- [ ] **Step 2: Run to see it fail**

Run: `npx vitest run src/campaign/scripts.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `src/campaign/scripts.ts`:

```ts
import type { AttackKind, Game } from '../game/game';

/** One enemy in a fight script (the same options as `Game.addEnemy`). */
export interface ScriptEnemy { kind: 'dummy' | 'spirit' | 'earth'; x: number; z: number; only?: AttackKind; pace?: number; cd?: number; hp?: number }
/** Enemies that arrive `at` seconds into the fight. */
export interface Group { at: number; enemies: ScriptEnemy[] }
export type Goal = { type: 'defeat' } | { type: 'survive'; seconds: number };
/** A campaign fight: who comes when, and what wins it. `boss` fights are driven by the boss instead. */
export interface FightScript { groups: Group[]; goal: Goal; boss?: boolean }
export type FightOutcome = 'fighting' | 'won' | 'lost';

/** Runs a fight script on a scripted game: sends groups in on time and says when it's won or lost. */
export class FightRunner {
  elapsed = 0;
  private sent = 0;

  constructor(private g: Game, private script: FightScript) {}

  update(dt: number): FightOutcome {
    if (this.g.state === 'over') return 'lost';
    this.elapsed += dt;
    const groups = this.script.groups;
    while (this.sent < groups.length && groups[this.sent].at <= this.elapsed) {
      for (const e of groups[this.sent].enemies) this.g.addEnemy({ ...e, tag: `g${this.sent}` });
      this.sent++;
    }
    const goal = this.script.goal;
    if (goal.type === 'survive') return this.elapsed >= goal.seconds ? 'won' : 'fighting';
    const standing = this.g.enemies.some(e => e.hp > 0);
    return this.sent === groups.length && !standing ? 'won' : 'fighting';
  }

  get progress(): number {
    const goal = this.script.goal;
    if (goal.type === 'survive') return Math.min(1, this.elapsed / goal.seconds);
    const total = this.script.groups.reduce((n, g) => n + g.enemies.length, 0) || 1;
    const standing = this.g.enemies.filter(e => e.hp > 0).length;
    const toCome = this.script.groups.slice(this.sent).reduce((n, g) => n + g.enemies.length, 0);
    return 1 - (standing + toCome) / total;
  }
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run src/campaign && npx tsc --noEmit -p .`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/campaign/scripts.ts src/campaign/scripts.test.ts
git commit -m "feat(web): campaign fight scripts (timed groups, defeat/survive goals)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Daro Stonefist (the boss)

**Files:**
- Create: `src/game/boss.ts`
- Modify: `src/game/game.ts`
- Test: `src/game/boss.test.ts`

**Interfaces:**
- Consumes: `Game` internals exposed by this task.
- Produces:
  - In `game.ts`: `Enemy.boss?: BossState` (from `boss.ts`), `Enemy.maxHp?: number`; `Hazard.halfW?: number`, `Hazard.look?: 'boulder'`; public `Game.sendPillar(e: Enemy, laneX: number, halfW?: number): void` (raise-and-shove immediately, for the boss), `Game.sendSlab(e: Enemy, look?: 'boulder'): void`; `Game.boss: Enemy | null` getter; `Game.addBoss(x: number, z: number): Enemy`.
  - In `boss.ts`:
```ts
export const BOSS = { hp: 40, phase2At: 0.6, phase3At: 0.25, wallHp: 1, wallEveryS: 7, attackEveryS: [3, 2.4, 1.9], windedS: 2, twinOffset: 36, twinHalfW: 16 };
export interface BossState { phase: 1 | 2 | 3; wall: number; wallT: number; attackT: number; winded: number; spiritT: number }
export function newBossState(): BossState;
export function bossPhase(e: { hp: number; maxHp?: number }): 1 | 2 | 3;
export function updateBoss(g: Game, e: Enemy, dt: number): void;
export function bossDamage(e: Enemy, shot: 'normal' | 'charged' | 'flurry' | 'counter' | 'pillar' | 'wall' | 'blade', dmg: number): number; // after stone wall / winded
```

- [ ] **Step 1: Write the failing test** `src/game/boss.test.ts`:

```ts
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
```

- [ ] **Step 2: Run to see it fail**

Run: `npx vitest run src/game/boss.test.ts`
Expected: FAIL (module / `addBoss` not found).

- [ ] **Step 3: Implement `src/game/boss.ts`:**

```ts
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
```

- [ ] **Step 4: Hook the boss into `src/game/game.ts`:**

1. Imports at the top: `import { BOSS, bossDamage, newBossState, updateBoss, type BossState } from './boss';`
2. `Enemy` gains:
```ts
  /** The chapter boss (Daro Stonefist). */
  boss?: BossState;
  /** Full health, for bosses' health bars. */
  maxHp?: number;
```
3. `Hazard` gains `halfW?: number; look?: 'boulder';` and `outOfWay` uses `(h.halfW ?? TUNE.stonePillarHalfW)`; so does the wall check in `updateHazards` (`h.kind === 'stonePillar' ? h.halfW ?? TUNE.stonePillarHalfW : 0`).
4. Public helpers (next to `addEnemy`):
```ts
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
```
5. In `updateEnemies`, right after `if (e.dummy) continue;` add `if (e.boss) { updateBoss(this, e, dt); continue; }` (the boss doesn't sway or use the normal attack cycle).
6. Damage paths go through `bossDamage` when the enemy is the boss:
   - Player fireball hit: replace `e.hp -= p.damage ?? 1;` with `e.hp -= e.boss ? bossDamage(e, p.shot ?? 'normal', p.damage ?? 1) : p.damage ?? 1;` and when that returns 0 and a wall stood, emit `'blocked'` instead of `'hitEnemy'` (compute `const dealt = …;` then `if (dealt === 0) this.emit('blocked', p.x, p.y, p.z); else if (e.hp <= 0) … else …`).
   - `burn(e, damage)`: `e.hp -= e.boss ? bossDamage(e, 'pillar', damage) : damage;` (rolling walls call `burn` too; pass `'wall'` from `updateWalls` by adding a third `shot` parameter to `burn` defaulting to `'pillar'`).
   - Blade (`updateBlades`): don't one-shot the boss — `if (e.boss) { e.hp -= bossDamage(e, 'blade', 10); … continue; }` before `e.hp = 0`.
7. Hazard `x` for pillars with a `halfW` is drawn by the renderer from `h.halfW` (Task 9).

- [ ] **Step 5: Run tests**

Run: `npx vitest run && npx tsc --noEmit -p .`
Expected: PASS (including the existing suite).

- [ ] **Step 6: Commit**

```bash
git add src/game/boss.ts src/game/boss.test.ts src/game/game.ts
git commit -m "feat(web): Daro Stonefist — phases, stone wall, twin pillars, boulder, winded

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Chapter 1 data (and one-lesson practices)

**Files:**
- Modify: `src/game/tutorial.ts` (lesson list parameter)
- Create: `src/campaign/chapter1.ts`
- Test: `src/campaign/chapter1.test.ts`

**Interfaces:**
- Consumes: `LESSONS`, `Lesson`, `Tutorial` (existing), `FightScript` (Task 3), `ScrollId` (Task 2), `MoveName` (Task 1).
- Produces:
  - `new Tutorial(g, start = 0, lessons: Lesson[] = LESSONS)`.
  - In `chapter1.ts`:
```ts
export interface ScrollDef { id: ScrollId; name: string; moves: MoveName[]; lessonId: string }
export interface StopDef {
  id: string; place: string; ren: string[];          // Ren's lines on arrival
  scroll?: ScrollId;                                  // found just before the arena
  practice: string[];                                 // lesson ids to practise first (ghost hands)
  fight: FightScript;
  newMove: MoveName | null;                           // for the third flame
  tint: { sky: string; light: string };
  pathAt: number;                                     // distance along the path of the arena (m)
  scrollAt?: number;                                  // distance of its scroll
  reward?: ScrollId;                                  // found after winning (the boss)
}
export const SCROLLS: Record<ScrollId, ScrollDef>;
export const START_MOVES: MoveName[];                 // ['punch', 'flurry']
export const STOPS: StopDef[];
export const PATH_LENGTH: number;
export function movesFor(scrolls: ScrollId[]): Set<MoveName>;
export const EPILOGUE: { lessonId: string; ren: string[] };
```

- [ ] **Step 1: Let `Tutorial` take a lesson list.** In `src/game/tutorial.ts` change the class to keep `private lessons: Lesson[]`:

```ts
  constructor(private g: Game, start = 0, private lessons: Lesson[] = LESSONS) {
    g.scripted();
    this.go(start);
  }

  get lesson(): Lesson { return this.lessons[this.index]; }
```
and replace every other `LESSONS` inside the class (`go`, `next`) with `this.lessons`. Run `npx vitest run src/game/tutorial.test.ts` — Expected: PASS (default unchanged).

- [ ] **Step 2: Write the failing test** `src/campaign/chapter1.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { LESSONS } from '../game/tutorial';
import { EPILOGUE, movesFor, PATH_LENGTH, SCROLLS, START_MOVES, STOPS } from './chapter1';

describe('Chapter 1', () => {
  it('has six stops down the path, in order, ending with the boss', () => {
    expect(STOPS.map(s => s.id)).toEqual(['courtyard', 'stairs', 'bridge', 'garden', 'gate', 'daro']);
    const at = STOPS.map(s => s.pathAt);
    expect([...at].sort((a, b) => a - b)).toEqual(at);
    expect(at.at(-1)).toBeLessThanOrEqual(PATH_LENGTH);
    expect(STOPS.at(-1)!.fight.boss).toBe(true);
  });

  it('every scroll is found once, just before the fight that needs it', () => {
    const found = STOPS.flatMap(s => (s.scroll ? [s.scroll] : [])).concat(STOPS.flatMap(s => (s.reward ? [s.reward] : [])));
    expect([...found].sort()).toEqual(Object.keys(SCROLLS).sort());
    for (const s of STOPS) if (s.scroll) expect(s.scrollAt!).toBeLessThan(s.pathAt);
  });

  it('practices and scrolls point at real lessons', () => {
    const ids = new Set(LESSONS.map(l => l.id));
    for (const s of STOPS) for (const p of s.practice) expect(ids.has(p), p).toBe(true);
    for (const sc of Object.values(SCROLLS)) expect(ids.has(sc.lessonId), sc.lessonId).toBe(true);
    expect(ids.has(EPILOGUE.lessonId)).toBe(true);
  });

  it('you start with punches and flurries; scrolls add the rest; never the X block or counters', () => {
    expect(START_MOVES).toEqual(['punch', 'flurry']);
    const all = movesFor(Object.keys(SCROLLS) as (keyof typeof SCROLLS)[]);
    for (const m of ['shield', 'palm', 'charge', 'wall', 'finisher'] as const) expect(all.has(m)).toBe(true);
    for (const m of ['xBlock', 'counter', 'oneTwo', 'volley', 'wallBreaker'] as const) expect(all.has(m)).toBe(false);
  });

  it('each stop only practises moves you have by then', () => {
    const have: string[] = [];
    for (const s of STOPS) {
      if (s.scroll) have.push(s.scroll);
      const moves = movesFor(have as never);
      for (const p of s.practice) {
        const needs = Object.values(SCROLLS).find(sc => sc.lessonId === p);
        if (needs) expect(have.includes(needs.id), `${s.id} practises ${p}`).toBe(true);
      }
      expect(s.newMove === null || moves.has(s.newMove)).toBe(true);
    }
  });
});
```

- [ ] **Step 3: Run to see it fail**

Run: `npx vitest run src/campaign/chapter1.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 4: Implement** `src/campaign/chapter1.ts`:

```ts
import type { MoveName } from '../game/game';
import type { ScrollId } from './progress';
import type { FightScript } from './scripts';

export interface ScrollDef { id: ScrollId; name: string; moves: MoveName[]; lessonId: string }

/** A stop on the path: arrive, maybe find a scroll, practise, fight. Distances are metres along the path. */
export interface StopDef {
  id: string;
  place: string;
  /** Ren's lines on arrival. */
  ren: string[];
  /** The scroll found on the way into this stop (at scrollAt). */
  scroll?: ScrollId;
  scrollAt?: number;
  /** Tutorial lesson ids practised (with ghost hands) before the fight. */
  practice: string[];
  fight: FightScript;
  /** The move that earns this stop's third flame. */
  newMove: MoveName | null;
  /** Colours for the fight view in this place. */
  tint: { sky: string; light: string };
  pathAt: number;
  /** A scroll found after winning here (the boss). */
  reward?: ScrollId;
}

export const SCROLLS: Record<ScrollId, ScrollDef> = {
  flameShield: { id: 'flameShield', name: 'Flame Shield', moves: ['shield'], lessonId: 'shield' },
  risingPillar: { id: 'risingPillar', name: 'Rising Pillar', moves: ['palm'], lessonId: 'palm' },
  heldBreath: { id: 'heldBreath', name: 'Held Breath', moves: ['charge'], lessonId: 'charge' },
  burningWall: { id: 'burningWall', name: 'Burning Wall', moves: ['wall'], lessonId: 'wall' },
  finalFlame: { id: 'finalFlame', name: 'Final Flame', moves: ['finisher'], lessonId: 'ultimate' },
};

/** What you can do before finding any scroll. */
export const START_MOVES: MoveName[] = ['punch', 'flurry'];

export function movesFor(scrolls: ScrollId[]): Set<MoveName> {
  return new Set<MoveName>([...START_MOVES, ...scrolls.flatMap(id => SCROLLS[id].moves)]);
}

export const PATH_LENGTH = 300;

const spirit = (x: number, z: number, only: 'orb' | 'slab', pace: number) => ({ kind: 'spirit' as const, x, z, only, pace, cd: 1.5, hp: 2 });
const earth = (x: number, z: number, pace: number) => ({ kind: 'earth' as const, x, z, only: 'pillar' as const, pace, cd: 1.5, hp: 2 });

export const STOPS: StopDef[] = [
  {
    id: 'courtyard', place: 'Temple Courtyard', pathAt: 30, newMove: 'flurry',
    ren: ['The Spirit Moon rises, and I am far away. You must keep the Flame.', 'Fists up. Let the fire come from your breath.'],
    practice: ['move', 'punch', 'flurry'],
    fight: { goal: { type: 'defeat' }, groups: [
      { at: 0, enemies: [spirit(-20, 9, 'orb', 3.5)] },
      { at: 4, enemies: [spirit(15, 10, 'orb', 3.5), spirit(-5, 11, 'orb', 3.5)] },
    ] },
    tint: { sky: '#2a1a3a', light: '#ffb070' },
  },
  {
    id: 'stairs', place: 'The Long Stairs', pathAt: 80, scroll: 'flameShield', scrollAt: 68, newMove: 'shield',
    ren: ['Spirits on the stairs. When you cannot step aside, stand your ground.'],
    practice: ['shield'],
    fight: { goal: { type: 'defeat' }, groups: [
      { at: 0, enemies: [spirit(-15, 8, 'orb', 1.6), spirit(15, 8, 'orb', 1.9)] },
      { at: 8, enemies: [spirit(0, 9, 'slab', 3), spirit(-20, 10, 'orb', 1.8)] },
    ] },
    tint: { sky: '#1f1a36', light: '#ff9a60' },
  },
  {
    id: 'bridge', place: 'Bamboo Bridge', pathAt: 135, scroll: 'risingPillar', scrollAt: 122, newMove: 'palm',
    ren: ["Daro's scouts. Earth moves slowly — read it, then answer with fire."],
    practice: ['pillar', 'palm'],
    fight: { goal: { type: 'defeat' }, groups: [
      { at: 0, enemies: [earth(0, 9, 3.2)] },
      { at: 5, enemies: [spirit(-10, 7, 'orb', 2.5), spirit(0, 9, 'orb', 2.5), spirit(10, 11, 'orb', 2.5)] },
      { at: 12, enemies: [earth(-15, 10, 3), earth(15, 9, 3.4)] },
    ] },
    tint: { sky: '#162235', light: '#ffc080' },
  },
  {
    id: 'garden', place: 'Stone Garden', pathAt: 190, scroll: 'heldBreath', scrollAt: 176, newMove: 'charge',
    ren: ['Some spirits are old and hard. Hold your breath, gather your fire, then strike once.'],
    practice: ['charge'],
    fight: { goal: { type: 'defeat' }, groups: [
      { at: 0, enemies: [{ ...spirit(0, 8, 'orb', 2.2), hp: 3 }, { ...spirit(-20, 10, 'orb', 2.6), hp: 3 }] },
      { at: 10, enemies: [earth(15, 9, 3), { ...spirit(-10, 9, 'slab', 3), hp: 3 }] },
    ] },
    tint: { sky: '#1d2430', light: '#ffb890' },
  },
  {
    id: 'gate', place: 'The Village Gate', pathAt: 245, scroll: 'burningWall', scrollAt: 232, newMove: 'wall',
    ren: ['They are at the gate. Raise a wall and let nothing through.'],
    practice: ['wall'],
    fight: { goal: { type: 'survive', seconds: 45 }, groups: [
      { at: 0, enemies: [earth(-15, 9, 2.8), spirit(15, 8, 'orb', 2)] },
      { at: 15, enemies: [earth(15, 10, 2.6), spirit(-10, 9, 'orb', 1.8)] },
      { at: 30, enemies: [earth(0, 9, 2.4), spirit(-20, 8, 'slab', 3), spirit(20, 8, 'orb', 1.8)] },
    ] },
    tint: { sky: '#2a1620', light: '#ff8a50' },
  },
  {
    id: 'daro', place: 'Daro Stonefist', pathAt: 262, newMove: null, reward: 'finalFlame',
    ren: ['Daro Stonefist. Strong, and slow to anger — and slower to tire. Break his wall. Stay out of his lanes.'],
    practice: [],
    fight: { goal: { type: 'defeat' }, boss: true, groups: [] },
    tint: { sky: '#2e1418', light: '#ff7a40' },
  },
];

/** After Daro falls: the Final Flame scroll and a first try of the finisher. */
export const EPILOGUE = {
  lessonId: 'ultimate',
  ren: ['He falls back — but Kuzan will come himself now.', 'Take my last scroll. Jab, jab… then gather the fire, and let it fly.'],
};
```

- [ ] **Step 5: Run tests**

Run: `npx vitest run && npx tsc --noEmit -p .`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/game/tutorial.ts src/campaign/chapter1.ts src/campaign/chapter1.test.ts
git commit -m "feat(web): chapter 1 data — scrolls, stops, Ren's lines, practices, fights

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The path (auto-walk rail and mouse look)

**Files:**
- Create: `src/explore/path.ts`
- Test: `src/explore/path.test.ts`

**Interfaces:**
- Consumes: `STOPS`, `PATH_LENGTH` (Task 5).
- Produces:
```ts
export interface V3 { x: number; y: number; z: number }
export interface Pause { at: number; kind: 'scroll' | 'arena'; stop: number }
export const WALK_SPEED: number;             // m/s
export const WAYPOINTS: V3[];                 // the path's centre line (world metres)
export function pauses(): Pause[];            // from STOPS, ascending
export class Rail {
  d: number;                                  // distance walked
  constructor(start?: number);
  /** Walk on up to the next pause after `d`; returns the pause reached this step, if any. */
  advance(dt: number, from: Pause[]): Pause | null;
  /** Jump to just before the next pause. */
  skip(from: Pause[]): void;
  pose(): { pos: V3; heading: number };       // heading = yaw (radians) along the path
}
export class Look { yaw: number; pitch: number; move(dx: number, dy: number): void; relax(dt: number): void }
```

- [ ] **Step 1: Write the failing test** `src/explore/path.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { Look, pauses, Rail, WALK_SPEED } from './path';
import { STOPS } from '../campaign/chapter1';

describe('Rail', () => {
  it('pauses at each scroll and arena, in order', () => {
    const p = pauses();
    expect(p.map(x => x.kind)).toEqual(STOPS.flatMap(s => (s.scroll ? ['scroll', 'arena'] : ['arena'])));
    expect(p.map(x => x.at)).toEqual([...p.map(x => x.at)].sort((a, b) => a - b));
  });

  it('walks at walking speed and stops at the next pause', () => {
    const r = new Rail(0), p = pauses();
    expect(r.advance(1, p)).toBeNull();
    expect(r.d).toBeCloseTo(WALK_SPEED);
    let hit = null;
    for (let i = 0; i < 1000 && !hit; i++) hit = r.advance(0.1, p);
    expect(hit).toEqual(p[0]);
    expect(r.d).toBe(p[0].at);
    expect(r.advance(1, p)).toBeNull(); // stays until moved on past it
  });

  it('walks on past a pause once nudged, and can skip to the next', () => {
    const p = pauses(), r = new Rail(p[0].at + 0.01);
    r.skip(p);
    expect(r.d).toBeGreaterThan(p[1].at - 3);
    expect(r.d).toBeLessThan(p[1].at);
  });

  it('gives a position and heading along the path', () => {
    const a = new Rail(10).pose(), b = new Rail(11).pose();
    expect(Math.hypot(b.pos.x - a.pos.x, b.pos.z - a.pos.z)).toBeGreaterThan(0.5);
    expect(Number.isFinite(a.heading)).toBe(true);
  });
});

describe('Look', () => {
  it('turns with the mouse within limits and drifts back ahead', () => {
    const l = new Look();
    l.move(10000, -10000);
    expect(Math.abs(l.yaw)).toBeLessThanOrEqual(1.25);
    expect(Math.abs(l.pitch)).toBeLessThanOrEqual(0.6);
    const y = l.yaw;
    l.relax(1);
    expect(Math.abs(l.yaw)).toBeLessThan(Math.abs(y));
  });
});
```

- [ ] **Step 2: Run to see it fail**

Run: `npx vitest run src/explore/path.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `src/explore/path.ts`:

```ts
import { PATH_LENGTH, STOPS } from '../campaign/chapter1';

export interface V3 { x: number; y: number; z: number }
/** A place on the path where the walk stops for you: a scroll to pick up, or an arena. */
export interface Pause { at: number; kind: 'scroll' | 'arena'; stop: number }

/** Walking pace along the path, metres per second. */
export const WALK_SPEED = 3.2;

/**
 * The path's centre line, from the temple courtyard (top) down the mountain to the village gate.
 * x/z across the ground, y up (metres). The places (see world3d.ts) sit along it.
 */
export const WAYPOINTS: V3[] = [
  { x: 0, y: 40, z: 0 }, { x: 0, y: 40, z: -30 },            // courtyard
  { x: 12, y: 32, z: -50 }, { x: 24, y: 22, z: -70 },        // the long stairs
  { x: 26, y: 20, z: -85 }, { x: 20, y: 18, z: -135 },       // bamboo bridge
  { x: 5, y: 12, z: -160 }, { x: -10, y: 8, z: -190 },       // stone garden
  { x: -12, y: 4, z: -225 }, { x: -8, y: 2, z: -262 },       // the village gate
  { x: -6, y: 1, z: -300 },
];

/** Cumulative distance at each waypoint, scaled so the last is PATH_LENGTH. */
const CUM = (() => {
  const raw = [0];
  for (let i = 1; i < WAYPOINTS.length; i++) {
    const a = WAYPOINTS[i - 1], b = WAYPOINTS[i];
    raw.push(raw[i - 1] + Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z));
  }
  const k = PATH_LENGTH / raw[raw.length - 1];
  return raw.map(d => d * k);
})();

export function pauses(): Pause[] {
  return STOPS.flatMap((s, i): Pause[] => [
    ...(s.scroll && s.scrollAt !== undefined ? [{ at: s.scrollAt, kind: 'scroll' as const, stop: i }] : []),
    { at: s.pathAt, kind: 'arena', stop: i },
  ]).sort((a, b) => a.at - b.at);
}

/** Point on the path `d` metres along it. */
export function pointAt(d: number): V3 {
  const t = Math.max(0, Math.min(PATH_LENGTH, d));
  let i = 1;
  while (i < CUM.length - 1 && CUM[i] < t) i++;
  const a = WAYPOINTS[i - 1], b = WAYPOINTS[i], k = (t - CUM[i - 1]) / (CUM[i] - CUM[i - 1] || 1);
  return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k, z: a.z + (b.z - a.z) * k };
}

/** Auto-walk along the path, stopping at each pause. */
export class Rail {
  constructor(public d = 0) {}

  advance(dt: number, from: Pause[]): Pause | null {
    const next = from.find(p => p.at > this.d + 1e-6);
    const onOne = from.find(p => Math.abs(p.at - this.d) < 1e-6);
    if (onOne) return null; // waiting here until moved on
    const target = next ? next.at : PATH_LENGTH;
    this.d = Math.min(target, this.d + WALK_SPEED * dt);
    return next && this.d >= next.at ? next : null;
  }

  /** Move on past the pause you're standing at (after it's done). */
  leave(): void { this.d += 0.02; }

  skip(from: Pause[]): void {
    const next = from.find(p => p.at > this.d + 1e-6);
    if (next) this.d = Math.max(this.d, next.at - 2);
  }

  pose(): { pos: V3; heading: number } {
    const a = pointAt(this.d), b = pointAt(this.d + 1.5);
    return { pos: a, heading: Math.atan2(-(b.x - a.x), -(b.z - a.z)) };
  }
}

/** Looking around with the mouse (radians, limited), drifting back to straight ahead. */
export class Look {
  yaw = 0;
  pitch = 0;

  move(dx: number, dy: number): void {
    this.yaw = Math.max(-1.2, Math.min(1.2, this.yaw - dx * 0.0025));
    this.pitch = Math.max(-0.55, Math.min(0.55, this.pitch - dy * 0.0025));
  }

  relax(dt: number): void {
    const k = Math.exp(-dt * 0.8);
    this.yaw *= k;
    this.pitch *= k;
  }
}
```

Note: the `advance` test expects `advance` to return `null` once at a pause; the runner calls `leave()` when a pause is finished. Adjust the third test to call `r.leave()` instead of starting at `p[0].at + 0.01` if preferred — both leave `d` just past the pause.

- [ ] **Step 4: Run tests**

Run: `npx vitest run src/explore && npx tsc --noEmit -p .`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/explore/path.ts src/explore/path.test.ts
git commit -m "feat(web): campaign path — auto-walk rail with pauses, mouse look

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The campaign runner (state machine)

**Files:**
- Create: `src/campaign/runner.ts`
- Test: `src/campaign/runner.test.ts`

**Interfaces:**
- Consumes: `Progress` (T2), `FightRunner` (T3), `STOPS`, `SCROLLS`, `EPILOGUE`, `movesFor` (T5), `Rail`, `pauses` (T6), `Tutorial`, `LESSONS` (existing), `Game`, `GameEvent`, `MoveName`.
- Produces:
```ts
export type CampaignState = 'walk' | 'scroll' | 'arena' | 'handoff' | 'countdown' | 'practice' | 'fight' | 'result' | 'lost' | 'end';
export interface Note { kind: 'scroll' | 'stop' | 'info'; text: string }
export interface Result { stop: number; flames: number; reasons: string[] }
export class CampaignRunner {
  state: CampaignState; stop: number; rail: Rail; game: Game | null; practice: Tutorial | null; fight: FightRunner | null;
  result: Result | null; countdown: number; handoffFor: number; ren: string[] | null;
  constructor(progress: Progress, makeGame: () => Game);
  readonly notes: Note[];                     // new notifications; UI drains them
  update(dt: number, cameraReady: boolean, events: GameEvent[]): void;
  interact(): void;                            // E / click
  skip(): void;                                // Space
  back(): void;                                // Esc during handoff/countdown → arena
  retry(): void;                               // after losing
  walkOn(): void;                              // after a result
  replay(stop: number): void;                  // from the map
  get allowed(): Set<MoveName>;
  get ghostMove(): string | null;              // the lesson id to demonstrate with ghost hands, or null
}
```

Behaviour: `walk` advances the rail; reaching a scroll pause → `scroll` (Ren's arrival lines shown); `interact` in `scroll` adds the scroll (note "New move learned: … — scroll added to your Scrolls (Tab)"), leaves the pause, back to `walk`. Reaching an arena → `arena` (Ren's lines); `interact` → `handoff`; `handoff` needs `cameraReady` continuously for 1 s → `countdown` (3 s) → `practice` (if the stop has practices: a `Tutorial` over the stop's lessons, `noDamage` true, ghost move = current lesson) else `fight`. Practice finished → `fight`: `g.clearField()`, `g.noDamage = false`, `g.allowed = allowed`, `FightRunner` (boss stop: `g.addBoss(0, 9)` and the fight is won when the boss falls). Won → `result` (flames: 1 finished, +1 if `g.hp >= 70`, +1 if the stop's `newMove` was used — detected from events: `punch` for punch/flurry via a `combo`/`flurry`, `pillar` for palm, `combo charged` for charge, `wall` for wall, shield `blocked` while shield on). `walkOn` saves progress and returns to `walk` (boss stop: goes to `scroll` for the reward, then a finisher practice, then `end` with `progress.finishChapter()`). Lost → `lost`; `retry` restarts the fight only.

- [ ] **Step 1: Write the failing test** `src/campaign/runner.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { Game } from '../game/game';
import { mulberry32 } from '../math';
import { Progress } from './progress';
import { CampaignRunner } from './runner';
import { STOPS } from './chapter1';

const make = () => { const g = new Game(mulberry32(1), 70, true); return g; };
const fresh = () => new CampaignRunner(Progress.load(null), make);
/** Advance until the state changes (or give up). */
function until(r: CampaignRunner, state: string, ready = true, seconds = 200) {
  for (let t = 0; t < seconds && r.state !== state; t += 1 / 30) r.update(1 / 30, ready, r.game?.drainEvents() ?? []);
  return r.state;
}
/** Finish whatever practice or fight is running, as if the player did it. */
function winFight(r: CampaignRunner) {
  for (let i = 0; i < 20000 && (r.state === 'practice' || r.state === 'fight'); i++) {
    if (r.state === 'practice') r.practice!.completedFor = 99, r.practice!.finished = true;
    r.game!.enemies.forEach(e => { e.hp = 0; });
    r.update(1 / 30, true, r.game!.drainEvents());
  }
}

describe('CampaignRunner', () => {
  it('walks to the first arena, needs the camera for a second, counts down, then practises and fights', () => {
    const r = fresh();
    expect(r.state).toBe('walk');
    expect(until(r, 'arena')).toBe('arena');
    expect(r.ren).toEqual(STOPS[0].ren);
    r.interact();
    expect(r.state).toBe('handoff');
    until(r, 'countdown', false, 2);
    expect(r.state).toBe('handoff'); // no camera: waits
    expect(until(r, 'countdown')).toBe('countdown');
    expect(until(r, 'practice')).toBe('practice');
    expect(r.ghostMove).toBe(STOPS[0].practice[0]);
  });

  it('Esc during the handoff goes back to the arena', () => {
    const r = fresh();
    until(r, 'arena');
    r.interact();
    r.back();
    expect(r.state).toBe('arena');
  });

  it('winning shows a result with flames and walking on saves the stop', () => {
    const p = Progress.load(null), r = new CampaignRunner(p, make);
    until(r, 'arena'); r.interact(); until(r, 'practice');
    winFight(r);
    expect(r.state).toBe('result');
    expect(r.result!.flames).toBeGreaterThanOrEqual(1);
    r.walkOn();
    expect(p.isDone(STOPS[0].id)).toBe(true);
    expect(r.state).toBe('walk');
  });

  it('a scroll on the path stops the walk; picking it up unlocks its move with a notification', () => {
    const r = fresh();
    until(r, 'arena'); r.interact(); until(r, 'practice'); winFight(r); r.walkOn();
    expect(until(r, 'scroll')).toBe('scroll');
    expect(r.allowed.has('shield')).toBe(false);
    r.interact();
    expect(r.allowed.has('shield')).toBe(true);
    expect(r.notes.some(n => n.kind === 'scroll' && n.text.includes('Flame Shield'))).toBe(true);
    expect(r.state).toBe('walk');
  });

  it('losing a fight offers a retry of the fight only', () => {
    const r = fresh();
    until(r, 'arena'); r.interact(); until(r, 'practice');
    r.practice!.finished = true;
    r.update(1 / 30, true, []);
    expect(r.state).toBe('fight');
    r.game!.hp = 0; r.game!.state = 'over';
    r.update(1 / 30, true, []);
    expect(r.state).toBe('lost');
    r.retry();
    expect(r.state).toBe('fight');
    expect(r.game!.hp).toBeGreaterThan(0);
  });

  it('skip jumps ahead to the next pause', () => {
    const r = fresh();
    r.skip();
    r.update(1, true, []);
    expect(r.state).toBe('arena');
  });

  it('beating Daro gives the Final Flame, a finisher practice, then the end of the chapter', () => {
    const p = Progress.load(null);
    for (const s of STOPS.slice(0, -1)) p.completeStop(s.id, 1);
    for (const s of STOPS) if (s.scroll) p.addScroll(s.scroll);
    const r = new CampaignRunner(p, make);
    r.replay(STOPS.length - 1);
    expect(r.state).toBe('arena');
    r.interact(); until(r, 'fight');
    expect(r.game!.boss).not.toBeNull();
    winFight(r);
    expect(r.state).toBe('result');
    r.walkOn();
    expect(r.state).toBe('scroll');
    r.interact();
    expect(r.allowed.has('finisher')).toBe(true);
    expect(r.state).toBe('practice');
    expect(r.ghostMove).toBe('ultimate');
    r.practice!.finished = true;
    r.update(1 / 30, true, []);
    expect(r.state).toBe('end');
    expect(p.data.chapterDone).toBe(true);
  });
});
```

- [ ] **Step 2: Run to see it fail**

Run: `npx vitest run src/campaign/runner.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `src/campaign/runner.ts`:

```ts
import type { Game, GameEvent, MoveName } from '../game/game';
import { LESSONS, Tutorial } from '../game/tutorial';
import { EPILOGUE, movesFor, SCROLLS, STOPS } from './chapter1';
import type { Progress, ScrollId } from './progress';
import { FightRunner } from './scripts';
import { pauses, Rail, type Pause } from '../explore/path';

export type CampaignState = 'walk' | 'scroll' | 'arena' | 'handoff' | 'countdown' | 'practice' | 'fight' | 'result' | 'lost' | 'end';
export interface Note { kind: 'scroll' | 'stop' | 'info'; text: string }
export interface Result { stop: number; flames: number; reasons: string[] }

/** Seconds the camera must see you before a fight starts, and the countdown after. */
export const HANDOFF_S = 1;
export const COUNTDOWN_S = 3;

/**
 * Chapter 1 as a state machine: walk the path ⇄ fight. Owns the fight's Game while there is one;
 * main.ts feeds it time, whether the camera sees you ready, and the game's events.
 */
export class CampaignRunner {
  state: CampaignState = 'walk';
  stop = 0;
  rail: Rail;
  game: Game | null = null;
  practice: Tutorial | null = null;
  fight: FightRunner | null = null;
  result: Result | null = null;
  countdown = 0;
  handoffFor = 0;
  /** Ren's current lines (shown as subtitles), or null. */
  ren: string[] | null = null;
  readonly notes: Note[] = [];
  private pauseList: Pause[] = pauses();
  private usedNew = false;
  /** After Daro: the Final Flame and a finisher practice. */
  private epilogue = false;

  constructor(private progress: Progress, private makeGame: () => Game) {
    // resume at the first stop not yet done
    const next = STOPS.findIndex(s => !progress.isDone(s.id));
    this.stop = next < 0 ? STOPS.length - 1 : next;
    const prev = this.stop > 0 ? STOPS[this.stop - 1].pathAt : 0;
    this.rail = new Rail(prev + (this.stop > 0 ? 0.02 : 0));
  }

  get allowed(): Set<MoveName> { return movesFor(this.progress.data.scrolls); }

  get ghostMove(): string | null {
    return this.state === 'practice' && this.practice && !this.practice.finished ? this.practice.lesson.id : null;
  }

  update(dt: number, cameraReady: boolean, events: GameEvent[]): void {
    switch (this.state) {
      case 'walk': {
        const hit = this.rail.advance(dt, this.pauseList);
        if (!hit) break;
        this.stop = hit.stop;
        this.ren = STOPS[hit.stop].ren;
        // a scroll you already have (replaying) doesn't stop you
        const sc = STOPS[hit.stop].scroll;
        if (hit.kind === 'scroll' && sc && this.progress.hasScroll(sc)) { this.rail.leave(); break; }
        this.state = hit.kind;
        break;
      }
      case 'handoff':
        this.handoffFor = cameraReady ? this.handoffFor + dt : 0;
        if (this.handoffFor >= HANDOFF_S) { this.state = 'countdown'; this.countdown = COUNTDOWN_S; }
        break;
      case 'countdown':
        this.countdown -= dt;
        if (this.countdown <= 0) this.startStop();
        break;
      case 'practice':
        this.practice!.update(dt, events);
        if (this.practice!.finished) {
          if (this.epilogue) this.finishChapter();
          else this.startFight();
        }
        break;
      case 'fight': {
        this.noteUse(events);
        const out = this.fightOutcome(dt);
        if (out === 'won') this.win();
        else if (out === 'lost') this.state = 'lost';
        break;
      }
    }
  }

  interact(): void {
    if (this.state === 'scroll') {
      const s = STOPS[this.stop], id = (this.epilogue ? s.reward : s.scroll) as ScrollId;
      if (this.progress.addScroll(id)) {
        this.notes.push({ kind: 'scroll', text: `New move learned: ${SCROLLS[id].name} — scroll added to your Scrolls (Tab)` });
      }
      this.progress.save();
      if (this.epilogue) {
        this.ren = EPILOGUE.ren;
        this.beginPractice([EPILOGUE.lessonId]);
        return;
      }
      this.rail.leave();
      this.state = 'walk';
    } else if (this.state === 'arena') {
      this.state = 'handoff';
      this.handoffFor = 0;
    }
  }

  skip(): void {
    if (this.state === 'walk') this.rail.skip(this.pauseList);
  }

  back(): void {
    if (this.state === 'handoff' || this.state === 'countdown') this.state = 'arena';
  }

  retry(): void {
    if (this.state === 'lost') this.startFight();
  }

  walkOn(): void {
    if (this.state !== 'result') return;
    const s = STOPS[this.stop];
    this.progress.completeStop(s.id, this.result!.flames);
    this.progress.save();
    this.notes.push({ kind: 'stop', text: `${s.place} — ${'🔥'.repeat(this.result!.flames)}` });
    this.game = null;
    this.fight = null;
    if (s.reward && !this.progress.hasScroll(s.reward)) {
      this.epilogue = true;
      this.state = 'scroll';
      return;
    }
    this.rail.leave();
    this.state = this.stop === STOPS.length - 1 ? 'end' : 'walk';
  }

  replay(stop: number): void {
    this.stop = stop;
    this.rail = new Rail(STOPS[stop].pathAt);
    this.ren = STOPS[stop].ren;
    this.state = 'arena';
    this.epilogue = false;
  }

  private startStop(): void {
    const s = STOPS[this.stop];
    if (s.practice.length) this.beginPractice(s.practice);
    else this.startFight();
  }

  private beginPractice(ids: string[]): void {
    this.game = this.makeGame();
    this.practice = new Tutorial(this.game, 0, ids.map(id => LESSONS.find(l => l.id === id)!));
    this.game.allowed = this.allowed;
    this.state = 'practice';
  }

  private startFight(): void {
    const s = STOPS[this.stop];
    this.game = this.makeGame();
    this.game.scripted();
    this.game.noDamage = false;
    this.game.allowed = this.allowed;
    this.game.label = s.place;
    this.fight = new FightRunner(this.game, s.fight);
    if (s.fight.boss) this.game.addBoss(0, 9);
    this.practice = null;
    this.usedNew = false;
    this.state = 'fight';
  }

  private fightOutcome(dt: number): 'fighting' | 'won' | 'lost' {
    const g = this.game!, out = this.fight!.update(dt);
    if (STOPS[this.stop].fight.boss) {
      if (g.state === 'over') return 'lost';
      return g.enemies.some(e => e.boss && e.hp > 0) ? 'fighting' : 'won';
    }
    return out;
  }

  /** Did you use this stop's new move? (for the third flame) */
  private noteUse(events: GameEvent[]): void {
    const m = STOPS[this.stop].newMove, g = this.game!;
    for (const e of events) {
      if ((m === 'flurry' && e.type === 'combo' && e.name === 'flurry')
        || (m === 'charge' && e.type === 'combo' && e.name === 'charged')
        || (m === 'palm' && e.type === 'pillar')
        || (m === 'wall' && e.type === 'wall')
        || (m === 'shield' && e.type === 'blocked' && g.shield.on)) this.usedNew = true;
    }
  }

  private win(): void {
    const s = STOPS[this.stop], g = this.game!, reasons = ['Finished'];
    if (g.hp >= 70) reasons.push('Took little damage');
    if (this.usedNew || s.newMove === null) reasons.push(s.newMove ? 'Used your new move' : 'Beat the boss');
    this.result = { stop: this.stop, flames: reasons.length, reasons };
    this.state = 'result';
  }

  private finishChapter(): void {
    this.progress.finishChapter();
    this.progress.save();
    this.epilogue = false;
    this.practice = null;
    this.game = null;
    this.notes.push({ kind: 'info', text: 'Chapter 1 complete' });
    this.state = 'end';
  }
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run && npx tsc --noEmit -p .`
Expected: PASS. If the "skip" test fails because the first pause is an arena 30 m ahead, check `Rail.skip` puts `d` 2 m short of it and `WALK_SPEED × 1 s` reaches it.

- [ ] **Step 5: Commit**

```bash
git add src/campaign/runner.ts src/campaign/runner.test.ts
git commit -m "feat(web): campaign runner — walk, scrolls, camera handoff, practice, fight, results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Stop scripts are winnable (a scripted player)

**Files:**
- Test: `src/campaign/winnable.test.ts`

**Interfaces:**
- Consumes: `STOPS`, `movesFor` (T5), `FightRunner` (T3), `Game`, `Game.incoming()`, `Game.boss` (T4).

- [ ] **Step 1: Write the test** `src/campaign/winnable.test.ts` — a bot with only the moves you have by each stop: punches the nearest enemy twice a second, leans away from pillar lanes and ducks sweeps using `g.incoming()`, blocks with the shield (when it has it) when an orb is about to land, throws charged punches at 3-hit enemies (when it has the charge), raises a wall at the gate every 5 s:

```ts
import { describe, expect, it } from 'vitest';
import { Game, type Enemy } from '../game/game';
import { mulberry32 } from '../math';
import { movesFor, STOPS } from './chapter1';
import type { ScrollId } from './progress';
import { FightRunner } from './scripts';
import type { Intent } from '../intent/interpret';

const base: Intent = { present: true, head: { x: 0, y: 0 }, hands: { l: null, r: null }, shoulders: { l: { x: -20, y: 20 }, r: { x: 20, y: 20 } }, punches: [], palms: [], shield: false, xBlock: false, casts: [], face: null, bodyTilt: 0 };

/** Where an enemy appears on screen (view units), to aim at it. */
const onScreen = (g: Game, e: Enemy) => { const s = 3 / (3 + e.z); return { x: (e.x - g.cam.x) * s, y: (e.y - g.cam.y) * s }; };

function play(stopIndex: number, smart: boolean): 'won' | 'lost' {
  const s = STOPS[stopIndex];
  const scrolls = STOPS.slice(0, stopIndex + 1).flatMap(x => (x.scroll ? [x.scroll] : [])) as ScrollId[];
  const moves = movesFor(scrolls);
  const g = new Game(mulberry32(stopIndex + 3), 70, true);
  g.scripted(); g.noDamage = false; g.allowed = moves;
  const f = new FightRunner(g, s.fight);
  if (s.fight.boss) g.addBoss(0, 9);
  let head = { x: 0, y: 0 };
  for (let t = 0; t < 240; t += 1 / 60) {
    const i: Intent = { ...base, head, punches: [], palms: [], casts: [] };
    if (smart) {
      const next = g.incoming()[0];
      head = !next || next.safe ? head : next.kind === 'slab' ? { x: head.x, y: 20 } : { x: next.away * 30, y: 0 };
      if (!next) head = { x: head.x * 0.98, y: head.y * 0.9 };
      const target = g.enemies.find(e => e.hp > 0 && (e.boss ? true : !e.dummy || true));
      const frame = Math.round(t * 60);
      if (target && frame % 30 === 0) {
        const at = onScreen(g, target);
        const charged = moves.has('charge') && (target.hp >= 3 || !!target.boss) && frame % 60 === 0;
        i.punches = [{ hand: frame % 60 ? 'l' : 'r', at, shoulder: { x: 20, y: 20 }, dir: null, charged }];
      }
      if (moves.has('wall') && frame % 300 === 0) i.casts = [{ kind: 'wall', at: { x: 0, y: 10 } }];
      if (moves.has('palm') && target && frame % 90 === 45) i.palms = [{ kind: 'push', hand: 'r', at: onScreen(g, target), shoulder: { x: 20, y: 20 }, dir: null }];
    }
    g.step(1 / 60, i);
    g.drainEvents();
    const out = s.fight.boss ? (g.state === 'over' ? 'lost' : g.boss ? 'fighting' : 'won') : f.update(1 / 60);
    if (out !== 'fighting') return out;
  }
  return 'lost';
}

describe('Chapter 1 fights', () => {
  STOPS.forEach((s, i) => {
    it(`${s.place} can be won with only the moves you have by then`, () => {
      expect(play(i, true)).toBe('won');
    });
  });

  it('standing still loses where the fight is meant to make you move', () => {
    for (const id of ['bridge', 'gate', 'daro']) {
      expect(play(STOPS.findIndex(s => s.id === id), false), id).toBe('lost');
    }
  });
});
```

- [ ] **Step 2: Run it**

Run: `npx vitest run src/campaign/winnable.test.ts`
Expected: PASS. If a stop fails, tune that stop's `pace`, `hp` or group timing in `src/campaign/chapter1.ts` (not the bot) until it passes — the fights must be winnable by a competent player with the moves they have — and note the change in the commit message.

- [ ] **Step 3: Commit**

```bash
git add src/campaign/winnable.test.ts src/campaign/chapter1.ts
git commit -m "test(web): every chapter 1 fight is winnable with the moves you have by then

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Ghost hands, boss and place tint in the fight view

**Files:**
- Create: `src/render/ghost.ts`
- Test: `src/render/ghost.test.ts`
- Modify: `src/render/renderer.ts`

**Interfaces:**
- Consumes: `Game.boss`, `BossState` (T4), lesson ids (existing).
- Produces:
```ts
// ghost.ts
export interface GhostHand { pos: { x: number; y: number }; open: boolean; scale: number }
export interface GhostPose { l: GhostHand; r: GhostHand }
export const GHOST_LOOP_S: number;
export function ghostPose(lessonId: string, t: number): GhostPose | null;   // null = no ghost for this lesson
// renderer.ts
Renderer.ghost: { lessonId: string; alpha: number } | null;   // set by main each frame
Renderer.tint: { sky: string; light: string } | null;         // set by main for campaign fights
```

- [ ] **Step 1: Write the failing test** `src/render/ghost.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { GHOST_LOOP_S, ghostPose } from './ghost';

describe('ghost hands', () => {
  it('loop smoothly for each move taught in the campaign', () => {
    for (const id of ['move', 'punch', 'flurry', 'shield', 'pillar', 'palm', 'charge', 'wall', 'ultimate']) {
      const a = ghostPose(id, 0), b = ghostPose(id, GHOST_LOOP_S);
      expect(a, id).not.toBeNull();
      expect(b!.r.pos.x).toBeCloseTo(a!.r.pos.x, 5);
      expect(b!.r.pos.y).toBeCloseTo(a!.r.pos.y, 5);
    }
  });

  it('show the move: a palm push opens the right hand and brings it forward', () => {
    const rest = ghostPose('palm', 0)!, out = ghostPose('palm', GHOST_LOOP_S * 0.45)!;
    expect(out.r.open).toBe(true);
    expect(out.r.scale).toBeGreaterThan(rest.r.scale);
  });

  it('a charge drops a fist to the hip and holds it', () => {
    const held = ghostPose('charge', GHOST_LOOP_S * 0.4)!;
    expect(held.r.open).toBe(false);
    expect(held.r.pos.y).toBeGreaterThan(40);
  });

  it('no ghost for moves not in the campaign', () => {
    expect(ghostPose('xblock', 0)).toBeNull();
  });
});
```

- [ ] **Step 2: Run to see it fail**

Run: `npx vitest run src/render/ghost.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `src/render/ghost.ts`:

```ts
/**
 * Ghost hands: translucent hands that show a move in a slow loop, over where your own hands are.
 * Poses are keyframes in view units (as HandState.pos: x right, y down, guard near y 22), eased.
 */
export interface GhostHand { pos: { x: number; y: number }; open: boolean; scale: number }
export interface GhostPose { l: GhostHand; r: GhostHand }

export const GHOST_LOOP_S = 2.4;

type Key = [t: number, lx: number, ly: number, lOpen: boolean, lScale: number, rx: number, ry: number, rOpen: boolean, rScale: number];
const G = { l: [-12, 22], r: [12, 22] } as const;
/** Keyframes per lesson: t in 0..1 of the loop; the last key must equal the first (it loops). */
const KEYS: Record<string, Key[]> = {
  move: [[0, -12, 22, false, 1, 12, 22, false, 1], [0.3, -30, 22, false, 1, -6, 22, false, 1], [0.6, 6, 22, false, 1, 30, 22, false, 1], [0.8, -12, 36, false, 1, 12, 36, false, 1], [1, -12, 22, false, 1, 12, 22, false, 1]],
  punch: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.15, ...G.l, false, 1, 4, 8, false, 1.6], [0.35, ...G.l, false, 1, ...G.r, false, 1], [0.5, -4, 8, false, 1.6, ...G.r, false, 1], [0.7, ...G.l, false, 1, ...G.r, false, 1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  flurry: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.1, ...G.l, false, 1, 4, 8, false, 1.6], [0.2, -4, 8, false, 1.6, ...G.r, false, 1], [0.3, ...G.l, false, 1, 4, 8, false, 1.6], [0.45, ...G.l, false, 1, ...G.r, false, 1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  shield: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.25, -16, 14, true, 1.1, 16, 14, true, 1.1], [0.85, -16, 14, true, 1.1, 16, 14, true, 1.1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  pillar: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.3, -40, 22, false, 1, -16, 22, false, 1], [0.7, -40, 22, false, 1, -16, 22, false, 1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  palm: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.2, ...G.l, false, 1, ...G.r, true, 1], [0.45, ...G.l, false, 1, 6, 10, true, 1.6], [0.7, ...G.l, false, 1, ...G.r, true, 1], [0.85, ...G.l, false, 1, ...G.r, false, 1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  charge: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.2, ...G.l, false, 1, 20, 58, false, 0.9], [0.6, ...G.l, false, 1, 20, 58, false, 0.9], [0.75, ...G.l, false, 1, 4, 8, false, 1.7], [0.9, ...G.l, false, 1, ...G.r, false, 1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  wall: [[0, -14, 44, true, 1, 14, 44, true, 1], [0.35, -14, 4, true, 1.1, 14, 4, true, 1.1], [0.7, -14, 4, true, 1.1, 14, 4, true, 1.1], [1, -14, 44, true, 1, 14, 44, true, 1]],
  ultimate: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.1, ...G.l, false, 1, 4, 8, false, 1.6], [0.2, -4, 8, false, 1.6, ...G.r, false, 1], [0.35, -4, 16, true, 1, 4, 16, true, 1], [0.55, -4, 16, true, 1, 4, 16, true, 1], [0.7, -36, 16, true, 1.1, 36, 16, true, 1.1], [0.85, -36, 16, true, 1.1, 36, 16, true, 1.1], [1, ...G.l, false, 1, ...G.r, false, 1]],
};

const ease = (k: number) => k * k * (3 - 2 * k);

export function ghostPose(lessonId: string, t: number): GhostPose | null {
  const keys = KEYS[lessonId];
  if (!keys) return null;
  const u = ((t % GHOST_LOOP_S) + GHOST_LOOP_S) % GHOST_LOOP_S / GHOST_LOOP_S;
  let i = 1;
  while (i < keys.length - 1 && keys[i][0] < u) i++;
  const a = keys[i - 1], b = keys[i], k = ease(Math.min(1, Math.max(0, (u - a[0]) / (b[0] - a[0] || 1))));
  const mix = (p: number, q: number) => p + (q - p) * k;
  return {
    l: { pos: { x: mix(a[1], b[1]), y: mix(a[2], b[2]) }, open: k < 0.5 ? a[3] : b[3], scale: mix(a[4], b[4]) },
    r: { pos: { x: mix(a[5], b[5]), y: mix(a[6], b[6]) }, open: k < 0.5 ? a[7] : b[7], scale: mix(a[8], b[8]) },
  };
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run src/render/ghost.test.ts`
Expected: PASS.

- [ ] **Step 5: Draw them in `src/render/renderer.ts`:**

1. Import: `import { ghostPose } from './ghost';` and `import { BOSS } from '../game/boss';`
2. Public fields on `Renderer`:
```ts
  /** Ghost hands to show (campaign practice), and how visible. */
  ghost: { lessonId: string; alpha: number } | null = null;
  /** Campaign place colours for the fight view (sky wash and light), or null. */
  tint: { sky: string; light: string } | null = null;
```
3. A method, called in `render()` right after `this.drawHands(g);`:
```ts
  /** Translucent hands demonstrating the move being learned, over your own. */
  private drawGhost(): void {
    if (!this.ghost) return;
    const pose = ghostPose(this.ghost.lessonId, this.t);
    if (!pose) return;
    const c = this.ctx, a = this.ghost.alpha * (0.55 + 0.1 * Math.sin(this.t * 3));
    c.save();
    c.globalAlpha = a;
    c.globalCompositeOperation = 'lighter';
    c.strokeStyle = c.fillStyle = 'rgba(160,210,255,0.55)';
    for (const [side, sign] of [['l', -1], ['r', 1]] as const) {
      const h = pose[side];
      this.handShape(c, this.viewToScreen(h.pos), sign, 0.6 * this.u, h.open, null, h.scale);
    }
    c.restore();
  }
```
4. Tint: at the start of `render()` after drawing the sky layer (`c.drawImage(this.sky, …)`), add:
```ts
    if (this.tint) {
      c.globalCompositeOperation = 'color';
      c.fillStyle = this.tint.sky;
      c.globalAlpha = 0.35;
      c.fillRect(-M, -M, W + 2 * M, H + 2 * M);
      c.globalAlpha = 1;
      c.globalCompositeOperation = 'source-over';
    }
```
5. Boss: in `drawEnemy`, earthbenders with `e.boss` draw 1.6× larger (multiply `k` by 1.6 in `drawEarthbender` when `e.boss`), a sagging glow when `e.boss.winded > 0` (`rgba(255,220,120,.35)` radial behind him), and a stone slab in front when `e.boss.wall > 0`:
```ts
    if (e.boss?.wall) {
      const a = this.project(e.x - 16, FLOOR_Y, e.z - 0.8), b = this.project(e.x + 16, FLOOR_Y - 45, e.z - 0.8);
      c.fillStyle = '#6f5a44'; c.strokeStyle = '#2d2014'; c.lineWidth = 2;
      c.fillRect(a.x, b.y, b.x - a.x, a.y - b.y); c.strokeRect(a.x, b.y, b.x - a.x, a.y - b.y);
    }
```
6. Pillars with `h.halfW` use it: in `drawHazard`, `const hw = h.halfW ?? TUNE.stonePillarHalfW;`. Boulders: in the slab branch, when `h.look === 'boulder'`, draw a rotating stone disc at `this.project(g.cam.x, h.y, h.z)` of radius `18 * u * s` (reuse the rock polygon style from `drawEarthbender`'s colours) instead of the water wave.

- [ ] **Step 6: Run all tests and typecheck**

Run: `npx vitest run && npx tsc --noEmit -p .`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/render/ghost.ts src/render/ghost.test.ts src/render/renderer.ts
git commit -m "feat(web): ghost hands, Daro's look (wall, winded, boulders), campaign place tint

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: The 3D world (three.js)

**Files:**
- Modify: `package.json` (add `three`, `@types/three`)
- Create: `src/explore/world3d.ts`
- Modify: `index.html` (add `<canvas id="world" class="hidden"></canvas>` before `<canvas id="game">`), `src/style.css` (`canvas#world { position: fixed; inset: 0; width: 100vw; height: 100vh; display: block; }`)

**Interfaces:**
- Consumes: `WAYPOINTS`, `pointAt`, `Rail`, `Look`, `pauses` (T6), `STOPS` (T5).
- Produces:
```ts
export class World3D {
  constructor(canvas: HTMLCanvasElement);
  resize(): void;
  /** Mark which scrolls are already taken (their stands go dark). */
  setTaken(stopIndices: number[]): void;
  render(rail: Rail, look: Look, dt: number, highlight: 'scroll' | 'arena' | null): void;
  dispose(): void;
}
```

- [ ] **Step 1: Add the dependency**

Run: `npm install three && npm install -D @types/three`
Expected: `package.json` lists `"three"` under dependencies and `"@types/three"` under devDependencies.

- [ ] **Step 2: Implement** `src/explore/world3d.ts` (not unit-tested: WebGL). Stylised night mountain; every place built from primitives:

```ts
import * as THREE from 'three';
import { STOPS } from '../campaign/chapter1';
import { Look, pauses, pointAt, Rail, WAYPOINTS } from './path';

/** The campaign path in 3D: a night mountain under the Spirit Moon, lantern-lit, foggy, with embers. */
export class World3D {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(70, 1, 0.1, 600);
  private embers: THREE.Points;
  private scrolls = new Map<number, THREE.Group>();
  private arenas = new Map<number, THREE.Mesh>();
  private t = 0;

  constructor(private canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.scene.background = new THREE.Color('#120c1f');
    this.scene.fog = new THREE.FogExp2('#1a1230', 0.018);
    this.scene.add(new THREE.HemisphereLight('#6a6fb0', '#1b0f12', 0.55));
    const moon = new THREE.DirectionalLight('#b8c4ff', 0.9);
    moon.position.set(-80, 120, -60);
    this.scene.add(moon);
    this.buildSky();
    this.buildTerrain();
    this.buildPath();
    this.buildPlaces();
    this.embers = this.buildEmbers();
    this.resize();
  }

  resize(): void {
    const w = innerWidth, h = innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setTaken(stops: number[]): void {
    for (const [i, g] of this.scrolls) g.visible = !stops.includes(i);
  }

  render(rail: Rail, look: Look, dt: number, highlight: 'scroll' | 'arena' | null): void {
    this.t += dt;
    const { pos, heading } = rail.pose();
    const bob = Math.sin(this.t * 7) * 0.04;
    this.camera.position.set(pos.x, pos.y + 1.7 + bob, pos.z);
    this.camera.rotation.set(look.pitch, heading + look.yaw, 0, 'YXZ');
    // embers drift up and wrap around the camera
    const p = this.embers.geometry.getAttribute('position') as THREE.BufferAttribute;
    for (let i = 0; i < p.count; i++) {
      let y = p.getY(i) + dt * (0.4 + (i % 7) * 0.08);
      if (y > pos.y + 12) y = pos.y - 2;
      p.setY(i, y);
    }
    p.needsUpdate = true;
    this.embers.position.set(pos.x, 0, pos.z);
    // scrolls turn and pulse; arena rings flicker, brighter when you're at one
    for (const g of this.scrolls.values()) { g.rotation.y += dt * 0.8; g.position.y = g.userData.baseY + Math.sin(this.t * 2) * 0.1; }
    for (const m of this.arenas.values()) (m.material as THREE.MeshBasicMaterial).opacity = (highlight === 'arena' ? 0.9 : 0.55) + 0.15 * Math.sin(this.t * 9);
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void { this.renderer.dispose(); }

  private buildSky(): void {
    const moon = new THREE.Mesh(new THREE.SphereGeometry(14, 32, 16), new THREE.MeshBasicMaterial({ color: '#dfe6ff', fog: false }));
    moon.position.set(-160, 150, -380);
    this.scene.add(moon);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(24, 32, 16), new THREE.MeshBasicMaterial({ color: '#8a90ff', transparent: true, opacity: 0.18, fog: false }));
    halo.position.copy(moon.position);
    this.scene.add(halo);
    const stars = new THREE.BufferGeometry(), n = 900, a = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const th = Math.random() * Math.PI * 2, ph = Math.random() * 0.45 * Math.PI, r = 450;
      a.set([Math.cos(th) * Math.cos(ph) * r, Math.sin(ph) * r + 40, Math.sin(th) * Math.cos(ph) * r], i * 3);
    }
    stars.setAttribute('position', new THREE.BufferAttribute(a, 3));
    this.scene.add(new THREE.Points(stars, new THREE.PointsMaterial({ color: '#ffffff', size: 1.2, fog: false })));
  }

  /** Mountain slopes: a displaced plane falling away from the path, with far peaks. */
  private buildTerrain(): void {
    const geo = new THREE.PlaneGeometry(700, 700, 140, 140);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.getAttribute('position') as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), z = pos.getZ(i);
      // follow the path's height near it, rising into ridges away from it
      const along = Math.max(0, Math.min(300, -z));
      const onPath = pointAt(along);
      const side = Math.abs(x - onPath.x);
      const ridge = Math.max(0, side - 18) * 0.45 + Math.sin(x * 0.05) * 4 + Math.cos(z * 0.04) * 5;
      pos.setY(i, onPath.y - 1.5 + ridge);
    }
    geo.computeVertexNormals();
    this.scene.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: '#2b2436', roughness: 0.95, flatShading: true })));
    for (let i = 0; i < 9; i++) {
      const peak = new THREE.Mesh(new THREE.ConeGeometry(60 + i * 8, 140 + (i % 3) * 40, 5), new THREE.MeshStandardMaterial({ color: '#1f1a2c', flatShading: true }));
      peak.position.set(-260 + i * 65, 30, -420 - (i % 2) * 60);
      this.scene.add(peak);
    }
  }

  /** The stone path: flagstones along the waypoints, lanterns every few metres. */
  private buildPath(): void {
    const stone = new THREE.MeshStandardMaterial({ color: '#5a5360', roughness: 0.9 });
    for (let d = 0; d < 300; d += 1.2) {
      const a = pointAt(d), b = pointAt(d + 1.2);
      const slab = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.25, 1.1), stone);
      slab.position.set(a.x, a.y, a.z);
      slab.lookAt(b.x, a.y, b.z);
      this.scene.add(slab);
    }
    for (let d = 6; d < 300; d += 14) {
      const a = pointAt(d), side = (Math.floor(d / 14) % 2) * 2 - 1;
      this.addLantern(a.x + side * 2.4, a.y, a.z);
    }
  }

  private addLantern(x: number, y: number, z: number): void {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 1.6), new THREE.MeshStandardMaterial({ color: '#3a2618' }));
    post.position.set(x, y + 0.8, z);
    const glow = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.45, 0.35), new THREE.MeshBasicMaterial({ color: '#ffb35c' }));
    glow.position.set(x, y + 1.75, z);
    const light = new THREE.PointLight('#ff9a4a', 6, 14, 2);
    light.position.copy(glow.position);
    this.scene.add(post, glow, light);
  }

  /** Each stop's place, its scroll stand (if any) and its arena ring of fire. */
  private buildPlaces(): void {
    const at = (d: number) => pointAt(d);
    // temple courtyard: red pillars, a brazier, dummies
    const c = at(STOPS[0].pathAt);
    const red = new THREE.MeshStandardMaterial({ color: '#8c1f1a', roughness: 0.6 });
    for (const [dx, dz] of [[-6, 4], [6, 4], [-6, -8], [6, -8]]) {
      const p = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.5, 6), red);
      p.position.set(c.x + dx, c.y + 3, c.z + dz);
      this.scene.add(p);
    }
    const roof = new THREE.Mesh(new THREE.ConeGeometry(11, 3, 4), new THREE.MeshStandardMaterial({ color: '#2a1a1a' }));
    roof.position.set(c.x, c.y + 7.5, c.z - 2);
    roof.rotation.y = Math.PI / 4;
    this.scene.add(roof);
    this.addBrazier(c.x, c.y, c.z - 6);
    // bamboo along the bridge
    const bamboo = new THREE.MeshStandardMaterial({ color: '#4d6b3a' });
    for (let d = 95; d < 140; d += 1.6) {
      for (const side of [-1, 1]) {
        const a = at(d), stalk = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 5 + (d % 3)), bamboo);
        stalk.position.set(a.x + side * (3 + (d % 2)), a.y + 2.5, a.z);
        this.scene.add(stalk);
      }
    }
    // stone garden: boulders on raked sand
    const g = at(STOPS[3].pathAt);
    const sand = new THREE.Mesh(new THREE.CircleGeometry(12, 32), new THREE.MeshStandardMaterial({ color: '#6d6456' }));
    sand.rotation.x = -Math.PI / 2;
    sand.position.set(g.x + 8, g.y + 0.05, g.z);
    this.scene.add(sand);
    for (let i = 0; i < 6; i++) {
      const b = new THREE.Mesh(new THREE.DodecahedronGeometry(0.8 + (i % 3) * 0.5), new THREE.MeshStandardMaterial({ color: '#4c4852', flatShading: true }));
      b.position.set(g.x + 4 + (i * 2.3) % 9, g.y + 0.5, g.z - 5 + (i * 3.1) % 10);
      this.scene.add(b);
    }
    // the village gate: palisade, torches, banners
    const gate = at(STOPS[4].pathAt + 8);
    const wood = new THREE.MeshStandardMaterial({ color: '#4a3220' });
    for (let i = -8; i <= 8; i++) {
      if (Math.abs(i) < 2) continue;
      const log = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 4.5), wood);
      log.position.set(gate.x + i * 0.55, gate.y + 2.2, gate.z);
      this.scene.add(log);
    }
    this.addBrazier(gate.x - 2.5, gate.y, gate.z + 1);
    this.addBrazier(gate.x + 2.5, gate.y, gate.z + 1);
    // scroll stands and arenas
    STOPS.forEach((s, i) => {
      if (s.scrollAt !== undefined) this.addScroll(i, at(s.scrollAt));
      this.addArena(i, at(s.pathAt));
    });
  }

  private addBrazier(x: number, y: number, z: number): void {
    const bowl = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.35, 0.6, 12), new THREE.MeshStandardMaterial({ color: '#3b3036', metalness: 0.4 }));
    bowl.position.set(x, y + 1, z);
    const fire = new THREE.Mesh(new THREE.ConeGeometry(0.45, 1, 8), new THREE.MeshBasicMaterial({ color: '#ff8a2a' }));
    fire.position.set(x, y + 1.8, z);
    const light = new THREE.PointLight('#ff7a30', 12, 20, 2);
    light.position.set(x, y + 2, z);
    this.scene.add(bowl, fire, light);
  }

  private addScroll(stop: number, p: { x: number; y: number; z: number }): void {
    const g = new THREE.Group();
    const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.4, 1), new THREE.MeshStandardMaterial({ color: '#3d2a1c' }));
    stand.position.y = -0.6;
    const roll = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.7, 12), new THREE.MeshBasicMaterial({ color: '#ffd27a' }));
    roll.rotation.z = Math.PI / 2;
    const light = new THREE.PointLight('#ffcc66', 8, 10, 2);
    g.add(stand, roll, light);
    g.position.set(p.x + 1.8, p.y + 1.2, p.z);
    g.userData.baseY = g.position.y;
    this.scene.add(g);
    this.scrolls.set(stop, g);
  }

  private addArena(stop: number, p: { x: number; y: number; z: number }): void {
    const ring = new THREE.Mesh(new THREE.RingGeometry(2.6, 3, 48), new THREE.MeshBasicMaterial({ color: '#ff6a2a', transparent: true, opacity: 0.6, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(p.x, p.y + 0.15, p.z - 3);
    this.scene.add(ring);
    this.arenas.set(stop, ring);
  }

  private buildEmbers(): THREE.Points {
    const n = 400, a = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) a.set([(Math.random() - 0.5) * 40, Math.random() * 60, (Math.random() - 0.5) * 40], i * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(a, 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ color: '#ffa04a', size: 0.12, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.scene.add(pts);
    return pts;
  }
}

/** For the map: the path's outline from above, as 2D points in 0..1. */
export function pathOutline(): { x: number; y: number }[] {
  const xs = WAYPOINTS.map(p => p.x), zs = WAYPOINTS.map(p => p.z);
  const [x0, x1, z0, z1] = [Math.min(...xs) - 10, Math.max(...xs) + 10, Math.min(...zs), Math.max(...zs)];
  return Array.from({ length: 61 }, (_, i) => { const p = pointAt((i / 60) * 300); return { x: (p.x - x0) / (x1 - x0), y: (p.z - z1) / (z0 - z1) }; });
}
export { pauses };
```

- [ ] **Step 3: Typecheck and build**

Run: `npx tsc --noEmit -p . && npm run build`
Expected: no errors (three is bundled).

- [ ] **Step 4: Visual check** — a temporary `?world` route isn't needed: Task 12 wires it in. Commit now.

```bash
git add package.json package-lock.json src/explore/world3d.ts index.html src/style.css
git commit -m "feat(web): three.js world for the campaign path (mountain, places, lanterns, embers, scrolls, arenas)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Campaign screens (DOM)

**Files:**
- Modify: `index.html`, `src/style.css`
- Create: `src/campaign/ui.ts`

**Interfaces:**
- Consumes: `CampaignRunner`, `CampaignState`, `Note`, `Result` (T7), `STOPS`, `SCROLLS` (T5), `Progress` (T2), `LessonDemo` (existing `src/render/lessonDemo.ts`), `pathOutline` (T10), `Game.boss`, `BOSS` (T4).
- Produces:
```ts
export interface HandoffCheck { seen: boolean; handsUp: boolean; distance: 'ok' | 'close' | 'far' | 'unknown' }
export class CampaignUI {
  constructor(progress: Progress, onMapPick: (stop: number) => void);
  update(r: CampaignRunner, check: HandoffCheck, now: number): void;   // every frame
  toggleMap(on?: boolean): void;
  toggleScrolls(on?: boolean): void;
  get overlayOpen(): boolean;                                          // map or scrolls showing
  hideAll(): void;
}
```

- [ ] **Step 1: Markup** — add to `index.html` inside `<body>` after the `.hud` div:

```html
  <div id="camp" class="camp hidden">
    <div id="campControls" class="controls"></div>
    <div id="campNotes" class="notes"></div>
    <div id="campPrompt" class="prompt hidden"></div>
    <div id="campRen" class="ren hidden"></div>
    <div id="campBoss" class="bossbar hidden"><span>Daro Stonefist</span><div class="bar wide"><i id="campBossFill"></i></div></div>
    <div class="card hidden" id="campHandoff">
      <h1>Step back and raise your fists</h1>
      <ul class="checks"><li id="chkSeen">Head and shoulders seen</li><li id="chkHands">Fists up</li><li id="chkDist">Good distance</li></ul>
      <div class="bar wide"><i id="campHandoffFill"></i></div>
      <p class="muted"><kbd>Esc</kbd> back</p>
    </div>
    <div id="campCount" class="count hidden"></div>
    <div class="card hidden" id="campResult">
      <h1 id="campResultTitle"></h1>
      <div id="campFlames" class="flames"></div>
      <ul id="campReasons" class="checks"></ul>
      <p class="muted">Take a seat — walk on when you're ready.</p>
      <div class="actions"><button id="campWalk">Walk on</button><button id="campAgain" class="secondary">Try again</button></div>
    </div>
    <div class="card hidden" id="campLost">
      <h1>The flame gutters</h1>
      <p>Catch your breath, then try the fight again.</p>
      <div class="actions"><button id="campRetry">Try again</button></div>
    </div>
    <div class="card hidden" id="campMap"><h1>The Ember Path</h1><canvas id="campMapCanvas"></canvas><p class="muted">Click a lit lantern to replay it · <kbd>M</kbd> close</p></div>
    <div class="card hidden" id="campScrolls"><h1>Scrolls</h1><div id="campScrollList" class="scrolls"></div><p class="muted"><kbd>Tab</kbd> close</p></div>
    <div class="card hidden" id="campEnd"><h1>Chapter 1 complete</h1><p>Daro falls back down the valley. Kuzan will come himself now.</p><div class="actions"><button id="campMenu">Menu</button></div></div>
  </div>
```

- [ ] **Step 2: Styles** — append to `src/style.css`:

```css
.camp .controls { position: fixed; left: 16px; bottom: 16px; font-size: 12px; color: var(--muted); background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; display: flex; gap: 10px; flex-wrap: wrap; max-width: calc(100vw - 32px); }
.camp .notes { position: fixed; right: 16px; top: 70px; display: flex; flex-direction: column; gap: 8px; width: min(340px, calc(100vw - 32px)); }
.camp .note { background: rgba(20,12,24,.85); border: 1px solid rgba(255,200,120,.45); border-radius: 12px; padding: 10px 14px; font-size: 14px; animation: slidein .35s ease-out; }
.camp .note.scroll { border-color: #ffd27a; box-shadow: 0 0 24px rgba(255,200,100,.25); }
@keyframes slidein { from { transform: translateX(40px); opacity: 0; } to { transform: none; opacity: 1; } }
.camp .prompt { position: fixed; left: 50%; bottom: 22%; transform: translateX(-50%); font-family: Cinzel, serif; font-size: 20px; color: #ffe2b8; background: rgba(14,9,18,.7); padding: 10px 18px; border-radius: 12px; border: 1px solid var(--line); }
.camp .ren { position: fixed; left: 50%; bottom: 9%; transform: translateX(-50%); width: min(720px, calc(100vw - 32px)); text-align: center; font-size: 18px; line-height: 1.5; color: #fff4e4; text-shadow: 0 2px 8px #000; }
.camp .ren b { color: var(--accent-2); font-family: Cinzel, serif; }
.camp .bossbar { position: fixed; left: 50%; top: 14px; transform: translateX(-50%); width: min(520px, calc(100vw - 32px)); text-align: center; font-family: Cinzel, serif; color: #ffd0a0; }
.camp .bossbar i { background: linear-gradient(90deg, #7a4a2a, #c08040); }
.camp .checks { list-style: none; padding: 0; margin: 10px 0; }
.camp .checks li { padding: 4px 0; color: var(--muted); }
.camp .checks li.ok { color: #9ef0b4; }
.camp .checks li.ok::before { content: '✓ '; }
.camp .count { position: fixed; inset: 0; display: grid; place-items: center; font-family: Cinzel, serif; font-size: 120px; color: #ffe2b8; text-shadow: 0 0 40px rgba(255,140,60,.8); pointer-events: none; }
.camp .flames { font-size: 44px; letter-spacing: 8px; }
.camp .flames .off { filter: grayscale(1); opacity: .25; }
#campMapCanvas { width: 100%; height: 360px; display: block; cursor: pointer; }
.camp .scrolls { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
.camp .scroll { border: 1px solid var(--line); border-radius: 10px; padding: 8px; text-align: center; }
.camp .scroll canvas { width: 100%; height: 90px; }
.camp .scroll.locked { opacity: .35; }
body.campaign #pill, body.campaign .tr .wave { display: none; }
```

- [ ] **Step 3: Implement** `src/campaign/ui.ts`:

```ts
import { BOSS } from '../game/boss';
import { LessonDemo } from '../render/lessonDemo';
import { pathOutline } from '../explore/world3d';
import { SCROLLS, STOPS } from './chapter1';
import type { Progress, ScrollId } from './progress';
import type { CampaignRunner } from './runner';

export interface HandoffCheck { seen: boolean; handsUp: boolean; distance: 'ok' | 'close' | 'far' | 'unknown' }

const $ = (id: string) => document.getElementById(id)!;
const show = (id: string, on = true) => $(id).classList.toggle('hidden', !on);

/** Everything the campaign shows on top of the world and the fights. */
export class CampaignUI {
  private demos = new Map<ScrollId, LessonDemo>();
  private noteTimes: { el: HTMLElement; until: number }[] = [];

  constructor(private progress: Progress, private onMapPick: (stop: number) => void) {
    for (const sc of Object.values(SCROLLS)) {
      const box = document.createElement('div');
      box.className = 'scroll';
      box.id = `scroll-${sc.id}`;
      const cv = document.createElement('canvas');
      const name = document.createElement('b');
      name.textContent = sc.name;
      box.append(cv, name);
      $('campScrollList').append(box);
      this.demos.set(sc.id, new LessonDemo(cv));
    }
    ($('campMapCanvas') as HTMLCanvasElement).addEventListener('click', e => this.mapClick(e));
  }

  get overlayOpen(): boolean { return !$('campMap').classList.contains('hidden') || !$('campScrolls').classList.contains('hidden'); }

  toggleMap(on = $('campMap').classList.contains('hidden')): void { show('campScrolls', false); show('campMap', on); if (on) this.drawMap(); }
  toggleScrolls(on = $('campScrolls').classList.contains('hidden')): void { show('campMap', false); show('campScrolls', on); }

  hideAll(): void { show('camp', false); document.body.classList.remove('campaign'); }

  update(r: CampaignRunner, check: HandoffCheck, now: number): void {
    show('camp');
    document.body.classList.add('campaign');
    const s = r.state, exploring = s === 'walk' || s === 'scroll' || s === 'arena';
    $('campControls').innerHTML = exploring
      ? '<span><kbd>Mouse</kbd> look</span><span><kbd>E</kbd> interact</span><span><kbd>Space</kbd> skip ahead</span><span><kbd>M</kbd> map</span><span><kbd>Tab</kbd> scrolls</span><span><kbd>Esc</kbd> menu</span>'
      : '<span><kbd>Esc</kbd> pause</span>';
    // Ren's lines while exploring or at the start of a fight
    const ren = r.ren && (exploring || s === 'handoff');
    show('campRen', !!ren);
    if (ren) $('campRen').innerHTML = r.ren!.map(l => `<b>Ren:</b> ${l}`).join('<br>');
    // prompt
    const stop = STOPS[r.stop];
    const prompt = s === 'scroll' ? `<kbd>E</kbd> Pick up the scroll` : s === 'arena' ? `<kbd>E</kbd> Enter ${stop.place}` : '';
    show('campPrompt', !!prompt);
    $('campPrompt').innerHTML = prompt;
    // camera handoff
    show('campHandoff', s === 'handoff');
    if (s === 'handoff') {
      $('chkSeen').classList.toggle('ok', check.seen);
      $('chkHands').classList.toggle('ok', check.handsUp);
      $('chkDist').classList.toggle('ok', check.distance === 'ok');
      $('chkDist').textContent = check.distance === 'close' ? 'Step back a little' : check.distance === 'far' ? 'Come a little closer' : 'Good distance';
      $('campHandoffFill').style.width = `${Math.round(Math.min(1, r.handoffFor) * 100)}%`;
    }
    show('campCount', s === 'countdown');
    if (s === 'countdown') $('campCount').textContent = String(Math.max(1, Math.ceil(r.countdown)));
    // result / lost / end
    show('campResult', s === 'result');
    if (s === 'result' && r.result) {
      $('campResultTitle').textContent = STOPS[r.result.stop].place;
      $('campFlames').innerHTML = [0, 1, 2].map(i => `<span class="${i < r.result!.flames ? '' : 'off'}">🔥</span>`).join('');
      $('campReasons').innerHTML = r.result.reasons.map(x => `<li class="ok">${x}</li>`).join('');
    }
    show('campLost', s === 'lost');
    show('campEnd', s === 'end');
    // boss bar
    const boss = r.game?.boss ?? null;
    show('campBoss', s === 'fight' && !!boss);
    if (boss) $('campBossFill').style.width = `${Math.round((boss.hp / (boss.maxHp ?? BOSS.hp)) * 100)}%`;
    // notifications
    for (const n of r.notes.splice(0)) {
      const el = document.createElement('div');
      el.className = `note ${n.kind}`;
      el.textContent = n.text;
      $('campNotes').append(el);
      this.noteTimes.push({ el, until: now + 5000 });
    }
    this.noteTimes = this.noteTimes.filter(n => (now < n.until ? true : (n.el.remove(), false)));
    // scrolls inventory
    if (!$('campScrolls').classList.contains('hidden')) {
      for (const [id, demo] of this.demos) {
        const have = this.progress.hasScroll(id);
        $(`scroll-${id}`).classList.toggle('locked', !have);
        $(`scroll-${id}`).title = have ? '' : 'Found later on the path';
        if (have) demo.draw(SCROLLS[id].lessonId, now / 1000);
      }
    }
  }

  private mapLanterns: { x: number; y: number; stop: number }[] = [];

  private drawMap(): void {
    const cv = $('campMapCanvas') as HTMLCanvasElement, dpr = Math.min(devicePixelRatio || 1, 2);
    cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
    const c = cv.getContext('2d')!, W = cv.clientWidth, H = cv.clientHeight;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = '#140d1c'; c.fillRect(0, 0, W, H);
    const pts = pathOutline().map(p => ({ x: 30 + p.x * (W - 60), y: 20 + p.y * (H - 40) }));
    c.strokeStyle = '#6a5a70'; c.lineWidth = 4; c.beginPath();
    pts.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y))); c.stroke();
    this.mapLanterns = STOPS.map((s, i) => { const k = Math.round((s.pathAt / 300) * 60); return { ...pts[k], stop: i }; });
    const next = STOPS.findIndex(s => !this.progress.isDone(s.id));
    for (const l of this.mapLanterns) {
      const done = this.progress.isDone(STOPS[l.stop].id), isNext = l.stop === next;
      c.fillStyle = done ? '#ffb35c' : isNext ? '#ffe08a' : '#3a3040';
      c.beginPath(); c.arc(l.x, l.y, isNext ? 10 : 8, 0, 7); c.fill();
      c.fillStyle = '#fff4e4'; c.font = '12px Inter';
      c.fillText(`${STOPS[l.stop].place}${done ? ' ' + '🔥'.repeat(this.progress.flames(STOPS[l.stop].id)) : ''}`, l.x + 14, l.y + 4);
    }
  }

  private mapClick(e: MouseEvent): void {
    const r = (e.target as HTMLCanvasElement).getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const hit = this.mapLanterns.find(l => Math.hypot(l.x - x, l.y - y) < 14);
    if (hit && this.progress.isDone(STOPS[hit.stop].id)) { this.toggleMap(false); this.onMapPick(hit.stop); }
  }
}
```

- [ ] **Step 4: Typecheck**

Run: `npx tsc --noEmit -p . && npx vitest run`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add index.html src/style.css src/campaign/ui.ts
git commit -m "feat(web): campaign screens — controls, notes, Ren, handoff, countdown, results, map, scrolls, boss bar

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Wire the campaign into the game

**Files:**
- Modify: `src/main.ts`, `index.html` (Campaign button in `#modes`)
- Modify: `web/PLAYTEST.md`, `docs/superpowers/specs/2026-09-27-campaign-chapter-1-design.md` (a "Built" note)

**Interfaces:**
- Consumes: everything above. `CameraTracker`, `MockTracker`, `Renderer.ghost`, `Renderer.tint`, `World3D`, `Look`, `CampaignRunner`, `CampaignUI`, `Progress`.

- [ ] **Step 1: Menu entry** — in `index.html`, first in `.modes`:

```html
      <button class="mode" data-mode="campaign"><b id="campaignLabel">Campaign</b><span>Chapter 1: The Ember Path. Walk down the mountain, find Ren's scrolls, and face Daro Stonefist.</span></button>
```

- [ ] **Step 2: main.ts** — add the campaign phase. Concretely:

1. Imports:
```ts
import { CampaignRunner } from './campaign/runner';
import { CampaignUI, type HandoffCheck } from './campaign/ui';
import { Progress } from './campaign/progress';
import { STOPS } from './campaign/chapter1';
import { World3D } from './explore/world3d';
import { Look } from './explore/path';
```
2. `type Mode = 'tutorial' | 'waves' | 'training' | 'campaign';` and the phase union gains nothing new — campaign play runs in phase `'play'` with `mode === 'campaign'`.
3. State:
```ts
const storage = (() => { try { return localStorage; } catch { return null; } })();
const progress = Progress.load(storage);
let campaign: CampaignRunner | null = null;
let campUI: CampaignUI | null = null;
let world: World3D | null = null;
const look = new Look();
```
4. `beginPlay(m)` for `'campaign'`: create `world ??= new World3D($('world') as HTMLCanvasElement)`, `campUI ??= new CampaignUI(progress, stop => campaign?.replay(stop))`, `campaign = new CampaignRunner(progress, () => new Game(Math.random, renderer.viewHalfW, true))`, `game = null`, and set the menu label to "Continue" when `Object.keys(progress.data.stops).length > 0`.
5. Each frame when `mode === 'campaign'` (in `loop`, instead of `stepGame`):
```ts
function stepCampaign(dt: number, now: number): void {
  const r = campaign!;
  const exploring = r.state === 'walk' || r.state === 'scroll' || r.state === 'arena' || r.state === 'end';
  const check = handoffCheck();
  // the fight's game is the runner's
  game = r.game;
  let events: GameEvent[] = [];
  if (game && intent && (r.state === 'practice' || r.state === 'fight')) {
    acc += dt;
    while (acc >= STEP) { game.step(STEP, { ...intent, punches: pendingPunches, casts: pendingCasts, palms: pendingPalms }); pendingPunches = []; pendingCasts = []; pendingPalms = []; acc -= STEP; }
    events = game.drainEvents();
    for (const e of events) { renderer.onEvent(e); hud.onEvent(e); }
    hud.update(game, intent.hands);
  }
  if (!campUI!.overlayOpen) r.update(dt, check.seen && check.handsUp && check.distance === 'ok', events);
  renderer.ghost = r.ghostMove ? { lessonId: r.ghostMove, alpha: 1 } : null;
  renderer.tint = STOPS[r.stop].tint;
  show('world', exploring || r.state === 'handoff' || r.state === 'countdown');
  show('game', !(exploring || r.state === 'handoff' || r.state === 'countdown'));
  if (exploring || r.state === 'handoff' || r.state === 'countdown') {
    look.relax(dt);
    world!.setTaken(STOPS.flatMap((s, i) => (s.scroll && progress.hasScroll(s.scroll) ? [i] : [])));
    world!.render(r.rail, look, dt, r.state === 'scroll' ? 'scroll' : r.state === 'arena' ? 'arena' : null);
  }
  campUI!.update(r, check, now);
}

/** Is the camera ready for a fight: you're seen, fists up, at a good distance? (Mouse & keys: always.) */
function handoffCheck(): HandoffCheck {
  if (tracker instanceof MockTracker) return { seen: true, handsUp: true, distance: 'ok' };
  const f = lastFrame, i = intent;
  if (!f || !i?.present) return { seen: false, handsUp: false, distance: 'unknown' };
  const up = (h: typeof i.hands.l) => !!h && h.inView && h.pos.y < 40;
  const d = f.body ? (1.05 * f.body.span3) / f.body.span2 : null;
  return { seen: true, handsUp: up(i.hands.l) && up(i.hands.r), distance: d === null ? 'unknown' : d < 0.9 ? 'close' : d > 2.2 ? 'far' : 'ok' };
}
```
   Interpretation keeps running during the campaign (so `intent` is fresh for the handoff check): in `onFrame`, treat `phase === 'play'` the same for `mode === 'campaign'` — it already calls `interpret` when `phase === 'play' && calibration`.
6. Input while exploring (only when `mode === 'campaign'` and the runner is in `walk`/`scroll`/`arena`/`end`):
   - `mousemove` with pointer lock: `look.move(e.movementX, e.movementY)`; clicking the world canvas requests pointer lock (`$('world').requestPointerLock()`), and a click while locked is `campaign.interact()`.
   - keys: `e` → `interact()`; ` ` (space) → `skip()` (preventDefault); `m` → `campUI.toggleMap()`; `tab` → `campUI.toggleScrolls()` (preventDefault); `escape` → if handoff/countdown `back()`, else if an overlay is open close it, else `showModes()`.
   - Result/lost/end buttons: `campWalk` → `walkOn()`, `campAgain` / `campRetry` → `retry()` (for `campAgain`, only when state is `result`: set `campaign.state = 'lost'` then `retry()`), `campMenu` → `showModes()`.
7. `showModes()` also calls `campUI?.hideAll()`, hides `#world`, sets `renderer.ghost = null; renderer.tint = null;` and `document.exitPointerLock?.()`.
8. Mouse & keys mode (`MockTracker`) keys that clash while exploring (`e`, `m`, space) are only routed to the campaign while exploring; during fights the mock keeps them.

- [ ] **Step 3: Typecheck, test, build**

Run: `npx tsc --noEmit -p . && npx vitest run && npm run build`
Expected: all pass.

- [ ] **Step 4: Play it in the browser** (dev server `web` on port 5173, `?input=mock&mode=campaign`): walk to the courtyard, E to enter, the countdown, practice with ghost hands (click to punch), the fight, the result, walk on, pick up the Flame Shield scroll (note appears), M map, Tab scrolls, Space skip, and the boss via the map replay after completing stops. Take screenshots of the 3D path, a scroll pickup, ghost hands, and the boss with its bar. Fix anything broken before committing.

- [ ] **Step 5: Docs** — append to `web/PLAYTEST.md`:

```md
- [ ] Campaign: the walk down the path looks good and runs smoothly; mouse look feels natural; Space skips ahead.
- [ ] Scrolls: picking one up shows the note; Tab lists it with its animation.
- [ ] The camera handoff is clear: the checklist tells you what's missing; the countdown starts once you're ready.
- [ ] Ghost hands make each new move obvious; they fade once you've got it.
- [ ] Every fight is winnable with the moves you have, and the boss's attacks are readable.
- [ ] M map shows your progress and replays finished stops.
```
and add a "Built" line under the spec's title: `**Status:** built (plan: docs/superpowers/plans/2026-09-27-campaign-chapter-1.md).`

- [ ] **Step 6: Commit**

```bash
git add src/main.ts index.html web/PLAYTEST.md ../docs/superpowers/specs/2026-09-27-campaign-chapter-1-design.md
git commit -m "feat(web): campaign chapter 1 playable — menu, explore/fight switching, pointer-lock look, map, scrolls

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
