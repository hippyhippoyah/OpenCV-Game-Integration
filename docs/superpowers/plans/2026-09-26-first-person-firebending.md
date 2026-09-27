# First-Person Firebending Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser game where your webcam-tracked head moves a first-person camera and your tracked hands make, throw and shield with fire against waves of spirits.

**Architecture:** Trackers (camera or mock) produce `TrackingFrame`s → a pure `interpret()` turns them into distance-invariant `Intent` → a pure fixed-timestep `Game` simulates → a Canvas 2D `Renderer` + DOM `Hud` draw it. The mock tracker emits the same frames as the camera, so the whole game runs and is tested without a webcam.

**Tech Stack:** Vite 8, TypeScript 5.9 (strict), Vitest 5, `@mediapipe/tasks-vision` 1.0.1 (HandLandmarker + PoseLandmarker lite, GPU), Canvas 2D.

**Spec:** `docs/superpowers/specs/2026-09-26-first-person-firebending-design.md` · visual reference `mockup/index.html`

## Global Constraints

- Everything lives in `web/`. The Python prototype and `mockup/` are not modified.
- v1 moves: fireball (summon + push-throw), flame shield (spread, drains, breaks), dodge (lean/duck). Nothing else.
- Geometry in `intent/` is measured in shoulder widths; game/world geometry is in world units (screen height ≈ 100 units, eyes at origin, y down, depth z ≥ 0 into the screen).
- `interpret/` and `game/` stay free of DOM access so they run in Vitest's node environment.
- Camera denied / model load failure must fall back to mouse & keys.
- Code blocks below start with a `// file:` (or `<!-- file: -->`, `/* file: */`) marker naming the file they belong to; the marker line itself is not part of the file.

## File Map

| File | Responsibility |
|---|---|
| `web/src/math.ts` | `Vec2`, clamp/lerp/dist/distToSeg, seeded PRNG |
| `web/src/input/types.ts` | `TrackingFrame`, `HandObs`, `Tracker` |
| `web/src/input/landmarks.ts` | MediaPipe landmarks → mirrored `TrackingFrame` (pure) |
| `web/src/input/camera.ts` | Webcam + MediaPipe models → `Tracker` |
| `web/src/input/mock.ts` | Mouse/keys → `Tracker` (+ browser bindings) |
| `web/src/intent/calibration.ts` | Neutral head + shoulder width from a still body |
| `web/src/intent/interpret.ts` | Frames → `Intent` (smoothing, push detection, dropout grace) |
| `web/src/game/game.ts` | Fire, shield, enemies, projectiles, waves, scoring, events |
| `web/src/render/renderer.ts` | Parallax world, enemies, particles, first-person hands |
| `web/src/render/hud.ts` | DOM HUD, toasts, banners |
| `web/src/render/debug.ts` | Corner camera/landmark view + live numbers |
| `web/src/main.ts` | Screens, tracker selection, calibration, game loop |
| `web/src/test/frames.ts` | Test helpers for building frames |

---

### Task 1: Scaffold `web/` with math utilities

**Files:**
- Create: `web/package.json`, `web/tsconfig.json`, `web/vite.config.ts`, `web/src/math.ts`
- Test: `web/src/math.test.ts`

**Interfaces:**
- Produces: `Vec2`, `clamp(v, lo, hi)`, `lerp(a, b, k)`, `dist(a, b)`, `distToSeg(p, a, b)`, `mulberry32(seed): () => number`

- [ ] **Step 1: Create the package and install tooling**

```bash
mkdir -p web/src && cd web
npm init -y >/dev/null
npm pkg set name=firebending-web private=true type=module version=0.1.0
npm pkg set scripts.dev="vite" scripts.build="tsc --noEmit && vite build" scripts.preview="vite preview" scripts.test="vitest run"
npm pkg delete main keywords author license description
npm i @mediapipe/tasks-vision@1.0.1
npm i -D vite@8 vitest@5 typescript@5.9
```

- [ ] **Step 2: Add TypeScript and Vite config**

`web/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "types": ["vite/client"]
  },
  "include": ["src"]
}
```

```ts
// file: web/vite.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
```

- [ ] **Step 3: Write the failing test**

```ts
// file: web/src/math.test.ts
import { describe, expect, it } from 'vitest';
import { clamp, distToSeg, mulberry32 } from './math';

describe('math', () => {
  it('clamps', () => {
    expect(clamp(5, 0, 3)).toBe(3);
    expect(clamp(-1, 0, 3)).toBe(0);
  });

  it('measures distance to a segment, including past its ends', () => {
    const a = { x: 0, y: 0 }, b = { x: 10, y: 0 };
    expect(distToSeg({ x: 5, y: 3 }, a, b)).toBeCloseTo(3);
    expect(distToSeg({ x: 13, y: 4 }, a, b)).toBeCloseTo(5);
  });

  it('seeded random is repeatable', () => {
    const r1 = mulberry32(7), r2 = mulberry32(7);
    expect([r1(), r1()]).toEqual([r2(), r2()]);
  });
});
```

- [ ] **Step 4: Run it to see it fail**

Run: `cd web && npm test`
Expected: FAIL — cannot find module `./math`

- [ ] **Step 5: Implement**

```ts
// file: web/src/math.ts
export interface Vec2 { x: number; y: number }

export const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));
export const lerp = (a: number, b: number, k: number): number => a + (b - a) * k;
export const dist = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);

/** Distance from p to the segment a–b. */
export function distToSeg(p: Vec2, a: Vec2, b: Vec2): number {
  const vx = b.x - a.x, vy = b.y - a.y, l2 = vx * vx + vy * vy || 1;
  const k = clamp(((p.x - a.x) * vx + (p.y - a.y) * vy) / l2, 0, 1);
  return Math.hypot(p.x - a.x - vx * k, p.y - a.y - vy * k);
}

/** Small seeded PRNG so game tests are deterministic. */
export function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let r = Math.imul(a ^ (a >>> 15), 1 | a);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
```

- [ ] **Step 6: Run tests and type-check**

Run: `cd web && npm test && npx tsc --noEmit`
Expected: 3 tests PASS, no type errors

- [ ] **Step 7: Commit**

```bash
git add web/package.json web/package-lock.json web/tsconfig.json web/vite.config.ts web/src/math.ts web/src/math.test.ts
git commit -m "feat(web): scaffold Vite/TS app with math utilities"
```

---

### Task 2: Tracking frames, landmark conversion and calibration

**Files:**
- Create: `web/src/input/types.ts`, `web/src/input/landmarks.ts`, `web/src/intent/calibration.ts`, `web/src/test/frames.ts`
- Test: `web/src/input/landmarks.test.ts`, `web/src/intent/calibration.test.ts`

**Interfaces:**
- Consumes: `Vec2`, `dist` from `math.ts`
- Produces:
  - `type Point = Vec2`; `interface HandObs { center: Point; size: number }`
  - `interface TrackingFrame { t: number; head: Point | null; shoulderL: Point | null; shoulderR: Point | null; hands: HandObs[] }`
  - `interface Tracker { poll(now: number): TrackingFrame | null; dispose(): void }`
  - `interface Landmark { x: number; y: number; visibility?: number }`; `toFrame(t, hands: Landmark[][], pose: Landmark[] | undefined): TrackingFrame`
  - `interface Calibration { head: Vec2; sw: number }`; `class Calibrator { add(f): number; progress(): number; result(): Calibration | null }`; `CALIBRATION_SECONDS = 1.5`
  - test helpers `bodyFrame(t, opts)`, `handsRel(mid, sw, left, right, size?)`

- [ ] **Step 1: Write types and test helpers**

```ts
// file: web/src/input/types.ts
import type { Vec2 } from '../math';

/** Mirrored, normalized video coordinates: (0,0) top-left, (1,1) bottom-right, as seen in a mirror. */
export type Point = Vec2;

export interface HandObs {
  /** Palm centre. */
  center: Point;
  /** Wrist → middle-knuckle distance; grows as the hand moves toward the camera. */
  size: number;
}

/** One tracked camera frame. Camera and mock trackers both produce these. */
export interface TrackingFrame {
  /** Seconds. */
  t: number;
  head: Point | null;
  shoulderL: Point | null;
  shoulderR: Point | null;
  hands: HandObs[];
}

export interface Tracker {
  /** Newest frame, or null if nothing new since the last call. `now` is in milliseconds. */
  poll(now: number): TrackingFrame | null;
  dispose(): void;
}
```

```ts
// file: web/src/test/frames.ts
import type { HandObs, TrackingFrame } from '../input/types';
import type { Vec2 } from '../math';

/** A standing body: shoulders centred on `mid`, head 0.75 shoulder-widths above them. */
export function bodyFrame(t: number, o: { mid?: Vec2; sw?: number; head?: Vec2; hands?: HandObs[] } = {}): TrackingFrame {
  const sw = o.sw ?? 0.2, mid = o.mid ?? { x: 0.5, y: 0.5 };
  return {
    t,
    head: o.head ?? { x: mid.x, y: mid.y - 0.75 * sw },
    shoulderL: { x: mid.x - sw / 2, y: mid.y },
    shoulderR: { x: mid.x + sw / 2, y: mid.y },
    hands: o.hands ?? [],
  };
}

/** Two hands placed relative to the shoulders, in shoulder widths (x right, y down). */
export function handsRel(mid: Vec2, sw: number, left: Vec2, right: Vec2, size = 0.4): HandObs[] {
  return [left, right].map(p => ({ center: { x: mid.x + p.x * sw, y: mid.y + p.y * sw }, size: size * sw }));
}
```

- [ ] **Step 2: Write the failing tests**

```ts
// file: web/src/input/landmarks.test.ts
import { describe, expect, it } from 'vitest';
import { toFrame, type Landmark } from './landmarks';

const pose = (): Landmark[] => Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 1 }));

describe('toFrame', () => {
  it('mirrors x, orders shoulders left-to-right on screen and summarises hands', () => {
    const p = pose();
    p[0] = { x: 0.4, y: 0.3, visibility: 1 };
    p[11] = { x: 0.6, y: 0.5, visibility: 1 };
    p[12] = { x: 0.4, y: 0.5, visibility: 1 };
    const hand: Landmark[] = Array.from({ length: 21 }, () => ({ x: 0.3, y: 0.6 }));
    hand[0] = { x: 0.3, y: 0.7 };
    const f = toFrame(1, [hand], p);
    expect(f.head!.x).toBeCloseTo(0.6);
    expect(f.shoulderL!.x).toBeCloseTo(0.4);
    expect(f.shoulderR!.x).toBeCloseTo(0.6);
    expect(f.hands[0].center.x).toBeCloseTo(0.7);
    expect(f.hands[0].center.y).toBeCloseTo(0.62);
    expect(f.hands[0].size).toBeCloseTo(0.1);
  });

  it('drops points the model is unsure about', () => {
    const p = pose();
    p[0].visibility = 0.2;
    p[11].visibility = 0.2;
    const f = toFrame(0, [], p);
    expect(f.head).toBeNull();
    expect(f.shoulderL).toBeNull();
  });

  it('handles no person', () => {
    expect(toFrame(0, [], undefined)).toEqual({ t: 0, head: null, shoulderL: null, shoulderR: null, hands: [] });
  });
});
```

```ts
// file: web/src/intent/calibration.test.ts
import { describe, expect, it } from 'vitest';
import { Calibrator } from './calibration';
import { bodyFrame } from '../test/frames';

describe('Calibrator', () => {
  it('needs 1.5 s of a still body, then returns the average pose', () => {
    const c = new Calibrator();
    for (let t = 0; t < 1.4; t += 0.1) c.add(bodyFrame(t));
    expect(c.result()).toBeNull();
    c.add(bodyFrame(1.5));
    const r = c.result()!;
    expect(r.sw).toBeCloseTo(0.2);
    expect(r.head.x).toBeCloseTo(0.5);
    expect(r.head.y).toBeCloseTo(0.35);
  });

  it('restarts when the head moves too much', () => {
    const c = new Calibrator();
    for (let t = 0; t <= 1.0; t += 0.1) c.add(bodyFrame(t));
    c.add(bodyFrame(1.1, { head: { x: 0.6, y: 0.35 } }));
    expect(c.progress()).toBe(0);
  });

  it('restarts when the body is lost', () => {
    const c = new Calibrator();
    for (let t = 0; t <= 1.0; t += 0.1) c.add(bodyFrame(t));
    c.add({ t: 1.1, head: null, shoulderL: null, shoulderR: null, hands: [] });
    expect(c.progress()).toBe(0);
  });
});
```

- [ ] **Step 3: Run to see them fail**

Run: `cd web && npm test`
Expected: FAIL — cannot find `./landmarks` and `./calibration`

- [ ] **Step 4: Implement**

```ts
// file: web/src/input/landmarks.ts
import type { HandObs, TrackingFrame } from './types';
import { dist, type Vec2 } from '../math';

/** The part of MediaPipe's NormalizedLandmark we use. */
export interface Landmark { x: number; y: number; visibility?: number }

const NOSE = 0, L_SHOULDER = 11, R_SHOULDER = 12;
const PALM = [0, 5, 9, 13, 17], WRIST = 0, MIDDLE_KNUCKLE = 9;
const MIN_VISIBILITY = 0.5;

const mirror = (p: Vec2): Vec2 => ({ x: 1 - p.x, y: p.y });
const visible = (p: Landmark | undefined): p is Landmark => !!p && (p.visibility ?? 1) >= MIN_VISIBILITY;

/** Raw MediaPipe results → mirrored TrackingFrame (moving right moves right on screen). */
export function toFrame(t: number, hands: Landmark[][], pose: Landmark[] | undefined): TrackingFrame {
  const head = pose && visible(pose[NOSE]) ? mirror(pose[NOSE]) : null;
  let shoulderL: Vec2 | null = null, shoulderR: Vec2 | null = null;
  if (pose && visible(pose[L_SHOULDER]) && visible(pose[R_SHOULDER])) {
    [shoulderL, shoulderR] = [mirror(pose[L_SHOULDER]), mirror(pose[R_SHOULDER])].sort((a, b) => a.x - b.x);
  }
  return { t, head, shoulderL, shoulderR, hands: hands.map(handObs) };
}

function handObs(lm: Landmark[]): HandObs {
  const c = { x: 0, y: 0 };
  for (const i of PALM) { c.x += lm[i].x / PALM.length; c.y += lm[i].y / PALM.length; }
  return { center: mirror(c), size: dist(lm[WRIST], lm[MIDDLE_KNUCKLE]) };
}
```

```ts
// file: web/src/intent/calibration.ts
import type { TrackingFrame } from '../input/types';
import { dist, type Vec2 } from '../math';

export interface Calibration {
  /** Neutral head position, normalized video coords. */
  head: Vec2;
  /** Neutral shoulder width, normalized. */
  sw: number;
}

export const CALIBRATION_SECONDS = 1.5;
/** Head drift, in shoulder widths, that restarts calibration. */
const MAX_DRIFT_SW = 0.15;

/** Averages the pose of someone standing still for CALIBRATION_SECONDS. */
export class Calibrator {
  private samples: { head: Vec2; sw: number; t: number }[] = [];

  /** Feed a frame; returns progress in [0, 1]. */
  add(f: TrackingFrame): number {
    if (!f.head || !f.shoulderL || !f.shoulderR) {
      this.samples = [];
      return 0;
    }
    const sw = dist(f.shoulderL, f.shoulderR);
    const first = this.samples[0];
    if (first && dist(first.head, f.head) / sw > MAX_DRIFT_SW) this.samples = [];
    this.samples.push({ head: { ...f.head }, sw, t: f.t });
    return this.progress();
  }

  progress(): number {
    const n = this.samples.length;
    if (n < 2) return 0;
    return Math.min(1, (this.samples[n - 1].t - this.samples[0].t) / CALIBRATION_SECONDS);
  }

  result(): Calibration | null {
    if (this.progress() < 1) return null;
    const n = this.samples.length, head = { x: 0, y: 0 };
    let sw = 0;
    for (const s of this.samples) {
      head.x += s.head.x / n;
      head.y += s.head.y / n;
      sw += s.sw / n;
    }
    return { head, sw };
  }
}
```

- [ ] **Step 5: Run tests**

Run: `cd web && npm test && npx tsc --noEmit`
Expected: all PASS, no type errors

- [ ] **Step 6: Commit**

```bash
git add web/src/input web/src/intent web/src/test
git commit -m "feat(web): tracking frame types, landmark conversion and calibration"
```

---

### Task 3: Intent interpreter

**Files:**
- Create: `web/src/intent/interpret.ts`
- Test: `web/src/intent/interpret.test.ts`

**Interfaces:**
- Consumes: `TrackingFrame`, `HandObs`, `Calibration`, `clamp`, `dist`, `Vec2`
- Produces:
  - `interface HandsIntent { l: Vec2; r: Vec2; center: Vec2; spread: number; vel: Vec2 }` (view units: world units relative to the eyes)
  - `interface Intent { present: boolean; head: Vec2; hands: HandsIntent | null; raised: boolean; throwNow: boolean }`
  - `TUNING` constants (`handScaleX`, `handScaleY`, `handOffsetY`, `leanUnitsPerSw`, `duckUnitsPerSw`, …)
  - `interface InterpretState`, `initialState(): InterpretState`, `interpret(f, cal, state): Intent` (mutates `state`)

- [ ] **Step 1: Write the failing tests**

```ts
// file: web/src/intent/interpret.test.ts
import { describe, expect, it } from 'vitest';
import { initialState, interpret, TUNING } from './interpret';
import type { Calibration } from './calibration';
import { bodyFrame, handsRel } from '../test/frames';

const cal: Calibration = { head: { x: 0.5, y: 0.35 }, sw: 0.2 };
const MID = { x: 0.5, y: 0.5 };

function pushSequence(growthPerFrame: number, spreadSw = 0.1) {
  const s = initialState();
  const out = [];
  for (let i = 0; i < 12; i++) {
    const size = 0.4 * (1 + growthPerFrame * i);
    const hands = handsRel(MID, 0.2, { x: -spreadSw / 2, y: -0.3 }, { x: spreadSw / 2, y: -0.3 }, size);
    out.push(interpret(bodyFrame(i / 30, { hands }), cal, s));
  }
  return out;
}

describe('interpret', () => {
  it('maps hands relative to the shoulders, independent of distance to the camera', () => {
    const near = interpret(
      bodyFrame(0, { sw: 0.2, hands: handsRel(MID, 0.2, { x: -0.5, y: -0.5 }, { x: 0.5, y: -0.5 }) }), cal, initialState());
    const farMid = { x: 0.5, y: 0.45 };
    const far = interpret(
      bodyFrame(0, { mid: farMid, sw: 0.1, hands: handsRel(farMid, 0.1, { x: -0.5, y: -0.5 }, { x: 0.5, y: -0.5 }) }), cal, initialState());
    for (const r of [near, far]) {
      expect(r.hands!.l.x).toBeCloseTo(-0.5 * TUNING.handScaleX);
      expect(r.hands!.r.x).toBeCloseTo(0.5 * TUNING.handScaleX);
      expect(r.hands!.center.y).toBeCloseTo(TUNING.handOffsetY - 0.5 * TUNING.handScaleY);
    }
  });

  it('labels the left-most hand as l regardless of detection order', () => {
    const hands = handsRel(MID, 0.2, { x: 0.6, y: 0 }, { x: -0.6, y: 0 });
    const r = interpret(bodyFrame(0, { hands }), cal, initialState());
    expect(r.hands!.l.x).toBeLessThan(r.hands!.r.x);
  });

  it('turns head offset into camera lean and duck', () => {
    const r = interpret(bodyFrame(0, { head: { x: 0.6, y: 0.4 } }), cal, initialState());
    expect(r.head.x).toBeCloseTo(0.5 * TUNING.leanUnitsPerSw);
    expect(r.head.y).toBeCloseTo(0.25 * TUNING.duckUnitsPerSw);
  });

  it('reports raised hands at chest height but not at the hips', () => {
    const up = interpret(bodyFrame(0, { hands: handsRel(MID, 0.2, { x: -0.1, y: 0 }, { x: 0.1, y: 0 }) }), cal, initialState());
    const down = interpret(bodyFrame(0, { hands: handsRel(MID, 0.2, { x: -0.1, y: 1.2 }, { x: 0.1, y: 1.2 }) }), cal, initialState());
    expect(up.raised).toBe(true);
    expect(down.raised).toBe(false);
  });

  it('fires exactly one throw when both hands push quickly toward the camera', () => {
    expect(pushSequence(0.06).filter(r => r.throwNow)).toHaveLength(1);
  });

  it('ignores slow drift in hand size', () => {
    expect(pushSequence(0.005).some(r => r.throwNow)).toBe(false);
  });

  it('does not throw when the hands are spread apart', () => {
    expect(pushSequence(0.06, 1.2).some(r => r.throwNow)).toBe(false);
  });

  it('keeps the hands through a short tracking dropout, then lets go', () => {
    const s = initialState();
    const hands = handsRel(MID, 0.2, { x: -0.1, y: 0 }, { x: 0.1, y: 0 });
    interpret(bodyFrame(0, { hands }), cal, s);
    expect(interpret(bodyFrame(0.3), cal, s).hands).not.toBeNull();
    expect(interpret(bodyFrame(0.6), cal, s).hands).toBeNull();
  });

  it('reports nobody present without a head and shoulders', () => {
    const r = interpret({ t: 0, head: null, shoulderL: null, shoulderR: null, hands: [] }, cal, initialState());
    expect(r.present).toBe(false);
    expect(r.hands).toBeNull();
  });
});
```

- [ ] **Step 2: Run to see it fail**

Run: `cd web && npx vitest run src/intent/interpret.test.ts`
Expected: FAIL — cannot find `./interpret`

- [ ] **Step 3: Implement**

```ts
// file: web/src/intent/interpret.ts
import type { HandObs, TrackingFrame } from '../input/types';
import type { Calibration } from './calibration';
import { clamp, dist, type Vec2 } from '../math';

/**
 * View space: world units relative to the eyes, x right, y down.
 * The screen shows roughly x ∈ ±80, y ∈ −45…55.
 */
export interface HandsIntent { l: Vec2; r: Vec2; center: Vec2; spread: number; vel: Vec2 }

export interface Intent {
  /** A head and shoulders are visible. */
  present: boolean;
  /** Camera offset in world units: lean → x, duck → y (down is +). */
  head: Vec2;
  hands: HandsIntent | null;
  /** Hands are up in front of the body (not resting at the hips). */
  raised: boolean;
  /** A push toward the camera happened this frame. */
  throwNow: boolean;
}

export const TUNING = {
  leanUnitsPerSw: 40, maxLean: 30,
  duckUnitsPerSw: 40, minDuck: -10, maxDuck: 25,
  /** Hand offset from the shoulder centre (in shoulder widths) × scale = view units. */
  handScaleX: 40, handScaleY: 32, handOffsetY: 20,
  raisedAboveY: 37,
  /** Exponential smoothing rate, 1/s. Higher = snappier but jittery. */
  smoothing: 18,
  lostGraceS: 0.5,
  /** A throw = average hand size grows by pushGrowth× within pushWindowS. */
  pushWindowS: 0.15, pushGrowth: 1.18, maxPushSpread: 20, throwCooldownS: 0.4,
};

export interface InterpretState {
  head: Vec2 | null;
  l: Vec2 | null;
  r: Vec2 | null;
  center: Vec2 | null;
  vel: Vec2;
  lastT: number | null;
  lastSeen: number;
  sizes: { t: number; size: number }[];
  lastThrow: number;
}

export const initialState = (): InterpretState => ({
  head: null, l: null, r: null, center: null, vel: { x: 0, y: 0 },
  lastT: null, lastSeen: -Infinity, sizes: [], lastThrow: -Infinity,
});

const smooth = (prev: Vec2 | null, next: Vec2, k: number): Vec2 =>
  prev ? { x: prev.x + (next.x - prev.x) * k, y: prev.y + (next.y - prev.y) * k } : { ...next };

export function interpret(f: TrackingFrame, cal: Calibration, s: InterpretState): Intent {
  const dt = s.lastT === null ? 0 : Math.max(1e-3, f.t - s.lastT);
  s.lastT = f.t;
  const k = dt === 0 ? 1 : 1 - Math.exp(-TUNING.smoothing * dt);

  if (!f.head || !f.shoulderL || !f.shoulderR) {
    return { present: false, head: s.head ? { ...s.head } : { x: 0, y: 0 }, hands: null, raised: false, throwNow: false };
  }
  const sw = dist(f.shoulderL, f.shoulderR) || cal.sw;
  const mid = { x: (f.shoulderL.x + f.shoulderR.x) / 2, y: (f.shoulderL.y + f.shoulderR.y) / 2 };

  s.head = smooth(s.head, {
    x: clamp(((f.head.x - cal.head.x) / sw) * TUNING.leanUnitsPerSw, -TUNING.maxLean, TUNING.maxLean),
    y: clamp(((f.head.y - cal.head.y) / sw) * TUNING.duckUnitsPerSw, TUNING.minDuck, TUNING.maxDuck),
  }, k);

  if (f.hands.length >= 2) {
    const [a, b] = [...f.hands].sort((p, q) => p.center.x - q.center.x);
    const toView = (h: HandObs): Vec2 => ({
      x: ((h.center.x - mid.x) / sw) * TUNING.handScaleX,
      y: TUNING.handOffsetY + ((h.center.y - mid.y) / sw) * TUNING.handScaleY,
    });
    s.l = smooth(s.l, toView(a), k);
    s.r = smooth(s.r, toView(b), k);
    s.lastSeen = f.t;
    s.sizes.push({ t: f.t, size: (a.size + b.size) / 2 / sw });
    while (s.sizes.length && f.t - s.sizes[0].t > TUNING.pushWindowS) s.sizes.shift();
  } else if (f.t - s.lastSeen > TUNING.lostGraceS) {
    s.l = s.r = s.center = null;
    s.sizes = [];
    s.vel = { x: 0, y: 0 };
  }

  let hands: HandsIntent | null = null, throwNow = false;
  if (s.l && s.r) {
    const center = { x: (s.l.x + s.r.x) / 2, y: (s.l.y + s.r.y) / 2 };
    if (s.center && dt > 0) {
      const kv = 1 - Math.exp(-12 * dt);
      s.vel = {
        x: s.vel.x + ((center.x - s.center.x) / dt - s.vel.x) * kv,
        y: s.vel.y + ((center.y - s.center.y) / dt - s.vel.y) * kv,
      };
    }
    s.center = center;
    hands = { l: { ...s.l }, r: { ...s.r }, center: { ...center }, spread: dist(s.l, s.r), vel: { ...s.vel } };

    const oldest = s.sizes[0], newest = s.sizes[s.sizes.length - 1];
    if (oldest && newest && newest.size / oldest.size >= TUNING.pushGrowth
      && hands.spread <= TUNING.maxPushSpread && center.y < TUNING.raisedAboveY
      && f.t - s.lastThrow >= TUNING.throwCooldownS) {
      throwNow = true;
      s.lastThrow = f.t;
      s.sizes = [];
    }
  }
  return { present: true, head: { ...s.head }, hands, raised: !!hands && hands.center.y < TUNING.raisedAboveY, throwNow };
}
```

- [ ] **Step 4: Run tests**

Run: `cd web && npm test && npx tsc --noEmit`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add web/src/intent/interpret.ts web/src/intent/interpret.test.ts
git commit -m "feat(web): interpret tracking into distance-invariant player intent"
```

---

### Task 4: Game simulation

**Files:**
- Create: `web/src/game/game.ts`
- Test: `web/src/game/game.test.ts`

**Interfaces:**
- Consumes: `Intent`, `HandsIntent`, `distToSeg`, `Vec2`
- Produces:
  - `FOCAL = 3`, `FLOOR_Y = 63`, `TUNE` constants
  - `interface Enemy { id; x; y; z; hp; t; appear; dying; flash; cd; winding; wind; side: 1 | -1; phase }`
  - `interface Proj { id; kind: 'player' | 'enemy'; x; y; z; vx; vy; vz; r; resolved }`
  - `type GameEvent` (positioned: `summon | extinguish | throw | blocked | playerHit | dodged | hitEnemy | killEnemy | clash`; plus `shieldBroken`, `gameOver`, `{ type: 'wave'; wave }`)
  - `bodyHit(v: Vec2, r: number): boolean`, `arrival(p: Proj): Vec2`
  - `class Game(rand?, viewHalfW?)` with `state, hp, score, wave, cam, hands, fire, shield, inv, enemies, projs, spawning, viewHalfW`, `step(dt, intent)`, `handWorld(p)`, `isThreat(p)`, `drainEvents()`

- [ ] **Step 1: Write the failing tests**

```ts
// file: web/src/game/game.test.ts
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

  it('bodyHit covers head and torso only', () => {
    expect(bodyHit({ x: 0, y: 0 }, 4)).toBe(true);
    expect(bodyHit({ x: 0, y: 30 }, 4)).toBe(true);
    expect(bodyHit({ x: 30, y: 0 }, 4)).toBe(false);
  });
});
```

- [ ] **Step 2: Run to see it fail**

Run: `cd web && npx vitest run src/game/game.test.ts`
Expected: FAIL — cannot find `./game`

- [ ] **Step 3: Implement**

```ts
// file: web/src/game/game.ts
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
```

- [ ] **Step 4: Run tests**

Run: `cd web && npm test && npx tsc --noEmit`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add web/src/game
git commit -m "feat(web): game simulation for fireball, shield, dodging and spirit waves"
```

---

### Task 5: Mock tracker

**Files:**
- Create: `web/src/input/mock.ts`
- Test: `web/src/input/mock.test.ts`

**Interfaces:**
- Consumes: `Tracker`, `TrackingFrame`, `HandObs`, `Calibration`, `TUNING`, `clamp`
- Produces: `MOCK_CALIBRATION: Calibration`; `interface ViewMapper { screenToView(x, y): Vec2 }`; `class MockTracker(view)` with `setMouse(x, y)`, `setKey(key, down)`, `wheel(deltaY)`, `setSpread(units)`, `push()`, `poll(now)`, `dispose()`; `bindMockControls(m, canvas): () => void`

- [ ] **Step 1: Write the failing tests**

```ts
// file: web/src/input/mock.test.ts
import { describe, expect, it } from 'vitest';
import { MOCK_CALIBRATION, MockTracker } from './mock';
import { initialState, interpret, type Intent } from '../intent/interpret';

const identity = { screenToView: (x: number, y: number) => ({ x, y }) };

describe('MockTracker', () => {
  it('produces frames that interpret back to the mouse position', () => {
    const m = new MockTracker(identity);
    m.setMouse(10, 15);
    const r = interpret(m.poll(0), MOCK_CALIBRATION, initialState());
    expect(r.hands!.center.x).toBeCloseTo(10);
    expect(r.hands!.center.y).toBeCloseTo(15);
    expect(r.hands!.spread).toBeCloseTo(6);
  });

  it('turns a click into a push that interpret reads as one throw', () => {
    const m = new MockTracker(identity);
    m.setMouse(0, 15);
    const s = initialState();
    let thrown = 0;
    for (let i = 0; i < 40; i++) {
      if (i === 5) m.push();
      if (interpret(m.poll((i * 1000) / 60), MOCK_CALIBRATION, s).throwNow) thrown++;
    }
    expect(thrown).toBe(1);
  });

  it('leaning with D moves the camera right', () => {
    const m = new MockTracker(identity);
    m.setKey('d', true);
    const s = initialState();
    let last: Intent | null = null;
    for (let i = 0; i < 90; i++) last = interpret(m.poll((i * 1000) / 60), MOCK_CALIBRATION, s);
    expect(last!.head.x).toBeGreaterThan(15);
  });
});
```

- [ ] **Step 2: Run to see it fail**

Run: `cd web && npx vitest run src/input/mock.test.ts`
Expected: FAIL — cannot find `./mock`

- [ ] **Step 3: Implement**

```ts
// file: web/src/input/mock.ts
import type { HandObs, Tracker, TrackingFrame } from './types';
import type { Calibration } from '../intent/calibration';
import { TUNING } from '../intent/interpret';
import { clamp, type Vec2 } from '../math';

/** The body the mock pretends to see, which is also its calibration. */
export const MOCK_CALIBRATION: Calibration = { head: { x: 0.5, y: 0.35 }, sw: 0.2 };
const MID = { x: 0.5, y: 0.5 }, SW = 0.2, HAND_SIZE = 0.08;
const PUSH_RAMP_S = 0.12, PUSH_HOLD_S = 0.3;

export interface ViewMapper { screenToView(x: number, y: number): Vec2 }

/** Pretends to be the camera: mouse = both hands, scroll = spread, A/D/S = lean/duck, click = push. */
export class MockTracker implements Tracker {
  private mouse = { x: 0, y: 0 };
  private keys = new Set<string>();
  private spreadT = 6;
  private spread = 6;
  private lean = 0;
  private duck = 0;
  private pushAt: number | null = null;
  private pushRequested = false;
  private lastT: number | null = null;

  constructor(private view: ViewMapper) {}

  setMouse(x: number, y: number): void { this.mouse = { x, y }; }
  setKey(key: string, down: boolean): void { if (down) this.keys.add(key); else this.keys.delete(key); }
  wheel(deltaY: number): void { this.spreadT = clamp(this.spreadT - deltaY * 0.03, 4, 40); }
  setSpread(units: number): void { this.spreadT = units; }
  push(): void { this.pushRequested = true; }

  poll(now: number): TrackingFrame {
    const t = now / 1000, dt = this.lastT === null ? 0 : clamp(t - this.lastT, 0, 0.05);
    this.lastT = t;
    const approach = (v: number, target: number, rate: number) => v + (target - v) * Math.min(1, dt * rate);
    this.lean = approach(this.lean, (this.keys.has('a') ? -1 : 0) + (this.keys.has('d') ? 1 : 0), 7);
    this.duck = approach(this.duck, this.keys.has('s') ? 1 : 0, 8);
    this.spread = approach(this.spread, this.spreadT, 12);

    if (this.pushRequested) { this.pushAt = t; this.pushRequested = false; }
    if (this.pushAt !== null && t - this.pushAt > PUSH_HOLD_S) this.pushAt = null;
    const push = this.pushAt === null ? 0 : clamp((t - this.pushAt) / PUSH_RAMP_S, 0, 1);

    // Leaning/ducking moves the whole upper body, like it does in front of a real camera.
    const mid = { x: MID.x + this.lean * 0.55 * SW, y: MID.y + this.duck * 0.6 * SW };
    const head = { x: MOCK_CALIBRATION.head.x + mid.x - MID.x, y: MOCK_CALIBRATION.head.y + mid.y - MID.y };
    const v = this.view.screenToView(this.mouse.x, this.mouse.y);
    const hand = (vx: number): HandObs => ({
      center: {
        x: mid.x + (vx / TUNING.handScaleX) * SW,
        y: mid.y + ((v.y - TUNING.handOffsetY) / TUNING.handScaleY) * SW,
      },
      size: HAND_SIZE * (1 + 0.35 * push),
    });
    return {
      t, head,
      shoulderL: { x: mid.x - SW / 2, y: mid.y },
      shoulderR: { x: mid.x + SW / 2, y: mid.y },
      hands: [hand(v.x - this.spread / 2), hand(v.x + this.spread / 2)],
    };
  }

  dispose(): void {}
}

/** Wire browser mouse and keyboard to a MockTracker. Returns an unbind function. */
export function bindMockControls(m: MockTracker, canvas: HTMLElement): () => void {
  const onMove = (e: MouseEvent) => m.setMouse(e.clientX, e.clientY);
  const onDown = (e: MouseEvent) => { if (e.button === 0) m.push(); };
  const onWheel = (e: WheelEvent) => m.wheel(e.deltaY);
  const onKeyDown = (e: KeyboardEvent) => {
    const k = e.key.toLowerCase();
    if (k === '1') m.setSpread(6);
    if (k === '2') m.setSpread(34);
    m.setKey(k, true);
  };
  const onKeyUp = (e: KeyboardEvent) => m.setKey(e.key.toLowerCase(), false);
  addEventListener('mousemove', onMove);
  canvas.addEventListener('mousedown', onDown);
  addEventListener('wheel', onWheel, { passive: true });
  addEventListener('keydown', onKeyDown);
  addEventListener('keyup', onKeyUp);
  return () => {
    removeEventListener('mousemove', onMove);
    canvas.removeEventListener('mousedown', onDown);
    removeEventListener('wheel', onWheel);
    removeEventListener('keydown', onKeyDown);
    removeEventListener('keyup', onKeyUp);
  };
}
```

- [ ] **Step 4: Run tests**

Run: `cd web && npm test && npx tsc --noEmit`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add web/src/input/mock.ts web/src/input/mock.test.ts
git commit -m "feat(web): mouse-and-keys mock tracker producing real tracking frames"
```

---

### Task 6: Browser app — renderer, HUD, debug view, camera tracker, screens

**Files:**
- Create: `web/index.html`, `web/src/style.css`, `web/src/render/renderer.ts`, `web/src/render/hud.ts`, `web/src/render/debug.ts`, `web/src/input/camera.ts`, `web/src/main.ts`

**Interfaces:**
- Consumes: everything above.
- Produces: `Renderer(canvas)` with `W, H, u, VP, viewHalfW, screenToView(x, y), resize(), onEvent(e), render(game | null, dt)`; `Hud` with `update(game), onEvent(e)`; `DebugView(canvas, textEl)` with `resize(), toggle(), draw(frame, intent, video)`; `CameraTracker.create(onStatus): Promise<CameraTracker>` (throws `CameraError`), `.video`.

This task is verified manually in the browser (drawing code), plus `tsc` and `vite build`.

- [ ] **Step 1: Page shell and styles**

```html
<!-- file: web/index.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Firebending</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <script type="module" src="/src/main.ts"></script>
</head>
<body>
  <canvas id="game"></canvas>

  <div class="hud">
    <div class="tl"><div class="label">Vitality</div><div class="bar"><i id="hpFill"></i></div></div>
    <div class="tr"><div class="label">Score</div><div class="score" id="score">0</div><div class="wave" id="wave"></div></div>
    <div class="pill off" id="pill"><b id="modeName">NO FIRE</b><span id="modeHint"></span></div>
    <div class="bl">
      <div class="label">Shield <span class="state" id="shieldState">ready</span></div>
      <div class="bar" id="shieldBar"><i id="shieldFill"></i></div>
    </div>
    <div class="br">
      <canvas id="pip"></canvas>
      <div class="pipinfo"><span>HEAD <em id="headTag">—</em></span><span>HANDS <em id="handsN">0</em></span><span id="fps"></span></div>
      <pre id="debugText" class="hidden"></pre>
    </div>
    <div id="toasts"></div>
    <div id="banner"></div>
    <div id="away" class="notice hidden">Step into frame — we can't see you</div>
  </div>

  <div class="card" id="start">
    <h1>Firebending</h1>
    <p>Bend fire with your hands in front of your webcam. You look through your own eyes: lean and duck to dodge, bring your palms together to make fire, push toward the screen to throw it, and spread your hands wide for a flame shield.</p>
    <p class="muted">Stand about 1.5–2 m back so your head, shoulders and hands are all in view.</p>
    <div class="actions">
      <button id="camBtn">Play with camera</button>
      <button id="mockBtn" class="secondary">Mouse &amp; keys</button>
    </div>
  </div>

  <div class="card hidden" id="status">
    <p id="statusText">Loading…</p>
    <div class="actions"><button id="statusFallback" class="hidden">Play with mouse &amp; keys</button></div>
  </div>

  <div class="card hidden" id="calib">
    <h1>Calibrating</h1>
    <p>Stand where you'll play, face the screen with your arms relaxed, and hold still.</p>
    <div class="bar wide"><i id="calibFill"></i></div>
  </div>

  <div class="card hidden" id="mockHelp">
    <h1>Mouse &amp; keys</h1>
    <div class="grid">
      <div><kbd>Mouse</kbd> move both hands</div>
      <div><kbd>Scroll</kbd> hands together / apart</div>
      <div><kbd>1</kbd> palms together → fire</div>
      <div><kbd>Click</kbd> push → throw</div>
      <div><kbd>2</kbd> hands wide → shield</div>
      <div><kbd>A</kbd><kbd>D</kbd> lean · <kbd>S</kbd> duck</div>
    </div>
    <div class="actions"><button id="mockHelpClose">Got it</button><span class="muted"><kbd>?</kbd> reopens · <kbd>`</kbd> debug numbers</span></div>
  </div>

  <div class="card hidden" id="over">
    <h1>Extinguished</h1>
    <p>The spirits got through. Final score: <b id="overScore">0</b></p>
    <div class="actions"><button id="againBtn">Rekindle</button><span class="muted"><kbd>R</kbd> restart · <kbd>C</kbd> recalibrate</span></div>
  </div>
</body>
</html>
```

```css
/* file: web/src/style.css */
:root {
  --ui: #f4e9dc;
  --muted: #b3a08c;
  --accent: #ff8a3d;
  --accent-2: #ffc36b;
  --spirit: #7fe3ff;
  --panel: rgba(14, 9, 18, 0.62);
  --line: rgba(255, 190, 140, 0.18);
}
html, body { margin: 0; height: 100%; background: #07060b; overflow: hidden; color: var(--ui);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
canvas#game { display: block; width: 100vw; height: 100vh; cursor: none; }
.hidden { display: none !important; }

.hud { position: fixed; inset: 0; pointer-events: none; }
.label { font-size: 10px; letter-spacing: .2em; text-transform: uppercase; color: var(--muted); }
.tl { position: absolute; top: 16px; left: 16px; }
.tr { position: absolute; top: 12px; right: 16px; text-align: right; }
.bar { width: min(260px, 38vw); height: 10px; border-radius: 6px; margin-top: 6px;
  background: rgba(255,255,255,.07); border: 1px solid var(--line); overflow: hidden; }
.bar > i { display: block; height: 100%; width: 100%; transition: width .12s;
  background: linear-gradient(90deg, #ff4b2b, #ff9a3d 70%, #ffd27a); }
.bar.wide, .bl .bar { width: 100%; }
.score { font-family: Cinzel, serif; font-size: 34px; font-weight: 700; line-height: 1.1;
  text-shadow: 0 0 18px rgba(255,140,60,.45); }
.wave { margin-top: 10px; font-family: Cinzel, serif; font-size: 15px; color: var(--accent-2); }

.pill { position: absolute; top: 14px; left: 50%; transform: translateX(-50%); text-align: center;
  background: var(--panel); border: 1px solid var(--line); border-radius: 999px; padding: 8px 18px;
  backdrop-filter: blur(6px); white-space: nowrap; }
.pill b { font-family: Cinzel, serif; font-size: 15px; letter-spacing: .08em; color: var(--accent-2); }
.pill span { display: block; font-size: 11px; color: var(--muted); margin-top: 2px; }
.pill.off b { color: var(--muted); }
.pill.shield b { color: #ffe08a; }

.bl { position: absolute; left: 16px; bottom: 16px; width: min(230px, 42vw); padding: 12px 14px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px; backdrop-filter: blur(6px); }
#shieldBar > i { background: linear-gradient(90deg, #ff8a3d, #ffe08a); }
#shieldBar.broken > i { background: #6b4a3a; }
.state { float: right; letter-spacing: .1em; }

.br { position: absolute; right: 16px; bottom: 16px; width: min(220px, 40vw); padding: 8px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px; backdrop-filter: blur(6px); }
.br canvas { display: block; width: 100%; aspect-ratio: 4 / 3; border-radius: 6px; }
.pipinfo { display: flex; justify-content: space-between; gap: 6px; font-size: 10px; color: var(--muted);
  margin-top: 6px; font-variant-numeric: tabular-nums; }
.pipinfo em { font-style: normal; color: #9dffcf; }
#debugText { margin: 6px 0 0; font: 10px/1.5 ui-monospace, Menlo, monospace; color: #cfe; white-space: pre; overflow: hidden; }

.notice { position: absolute; top: 84px; left: 50%; transform: translateX(-50%); padding: 10px 18px; border-radius: 10px;
  background: rgba(120,20,20,.75); border: 1px solid rgba(255,120,120,.3); font-weight: 600; }

#toasts { position: absolute; left: 50%; top: 26%; transform: translateX(-50%); display: flex;
  flex-direction: column; align-items: center; gap: 4px; }
.toast { font-family: Cinzel, serif; font-weight: 700; font-size: 22px; letter-spacing: .1em;
  color: var(--accent-2); text-shadow: 0 0 16px rgba(255,120,40,.7); animation: pop 1s ease-out forwards; }
.toast.bad { color: #ff8080; text-shadow: 0 0 16px rgba(255,40,40,.7); }
.toast.cool { color: var(--spirit); text-shadow: 0 0 16px rgba(80,200,255,.6); }
@keyframes pop { 0% { opacity: 0; transform: translateY(8px) scale(.9); } 15% { opacity: 1; transform: none; }
  75% { opacity: 1; } 100% { opacity: 0; transform: translateY(-16px); } }

#banner { position: absolute; left: 0; right: 0; top: 36%; text-align: center; font-family: Cinzel, serif;
  font-size: clamp(34px, 7vw, 72px); font-weight: 700; letter-spacing: .2em; color: #ffe2b8; opacity: 0;
  text-shadow: 0 0 40px rgba(255,120,40,.6); }
#banner.show { animation: banner 2s ease-out forwards; }
@keyframes banner { 0% { opacity: 0; letter-spacing: .5em; } 20% { opacity: 1; letter-spacing: .2em; }
  75% { opacity: 1; } 100% { opacity: 0; } }

.card { position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%);
  width: min(560px, calc(100vw - 32px)); max-height: calc(100vh - 32px); overflow: auto; box-sizing: border-box;
  padding: 24px 26px; background: rgba(14, 9, 18, .9); border: 1px solid var(--line); border-radius: 16px;
  backdrop-filter: blur(10px); box-shadow: 0 30px 80px rgba(0,0,0,.6); cursor: default; }
.card h1 { font-family: Cinzel, serif; margin: 0 0 8px; font-size: 28px; color: #ffe2b8; }
.card p { margin: 0 0 14px; color: var(--muted); font-size: 14px; line-height: 1.5; }
.card b { color: var(--accent-2); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px 18px; font-size: 13px; }
.grid div { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
kbd { font-family: inherit; font-size: 11px; min-width: 18px; text-align: center; padding: 2px 6px; border-radius: 5px;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); color: var(--ui); }
.actions { display: flex; align-items: center; gap: 12px; margin-top: 18px; flex-wrap: wrap; }
button { font: 600 14px Inter, sans-serif; color: #1a0c05; cursor: pointer; border: 0; padding: 10px 20px; border-radius: 10px;
  background: linear-gradient(180deg, #ffc36b, #ff8a3d); box-shadow: 0 6px 24px rgba(255,120,40,.35); }
button.secondary { background: rgba(255,255,255,.08); color: var(--ui); box-shadow: none; border: 1px solid var(--line); }
.muted { color: var(--muted); font-size: 12px; }

@media (max-width: 760px) {
  .pill { top: 92px; }
  .score { font-size: 26px; }
}
```

- [ ] **Step 2: Renderer**

```ts
// file: web/src/render/renderer.ts
import { arrival, FLOOR_Y, FOCAL, type Enemy, type Game, type GameEvent } from '../game/game';
import { lerp, mulberry32, type Vec2 } from '../math';

type Pal = 'fire' | 'spirit';
interface Particle {
  x: number; y: number; z: number; vx: number; vy: number; vz: number;
  life: number; max: number; size: number; pal: Pal; rise: number;
}

/** How far each background layer shifts when your head moves (1 = as much as the floor at your feet). */
const PAR_SKY = 0.03, PAR_MID = 0.16;
const MAX_PARTICLES = 2600;
const rnd = (a: number, b: number) => a + Math.random() * (b - a);
const nOf = (rate: number, dt: number) => { const x = rate * dt; return Math.floor(x) + (Math.random() < x % 1 ? 1 : 0); };

function sprite(stops: [number, string][]): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d')!, gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  for (const [o, col] of stops) gr.addColorStop(o, col);
  g.fillStyle = gr;
  g.fillRect(0, 0, 64, 64);
  return c;
}

const SPR: Record<Pal, HTMLCanvasElement[]> = {
  fire: [
    sprite([[0, 'rgba(255,255,235,1)'], [0.3, 'rgba(255,220,130,.85)'], [1, 'rgba(255,130,30,0)']]),
    sprite([[0, 'rgba(255,190,90,.95)'], [0.4, 'rgba(255,110,30,.55)'], [1, 'rgba(200,40,10,0)']]),
    sprite([[0, 'rgba(220,60,20,.6)'], [0.5, 'rgba(120,20,10,.25)'], [1, 'rgba(60,0,0,0)']]),
  ],
  spirit: [
    sprite([[0, 'rgba(240,255,255,1)'], [0.3, 'rgba(150,235,255,.8)'], [1, 'rgba(60,160,255,0)']]),
    sprite([[0, 'rgba(120,220,255,.9)'], [0.4, 'rgba(60,140,255,.5)'], [1, 'rgba(40,40,200,0)']]),
    sprite([[0, 'rgba(120,80,255,.5)'], [0.5, 'rgba(70,40,180,.2)'], [1, 'rgba(30,0,80,0)']]),
  ],
};

/** Draws the first-person world. World units: eyes at the origin, x right, y down, z into the screen. */
export class Renderer {
  W = 0;
  H = 0;
  /** Pixels per world unit at the player's plane. */
  u = 8;
  VP: Vec2 = { x: 0, y: 0 };

  private ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private M = 0;
  private t = 0;
  private shake = 0;
  private flash = 0;
  private cam: Vec2 = { x: 0, y: 0 };
  private sky = document.createElement('canvas');
  private mid = document.createElement('canvas');
  private vig = document.createElement('canvas');
  private handLayer = document.createElement('canvas');
  private parts: Particle[] = [];
  private lanterns: Vec2[] = [];

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    this.resize();
  }

  /** Half the screen width in world units at the player's plane. */
  get viewHalfW(): number { return this.W / 2 / this.u; }

  screenToView(x: number, y: number): Vec2 { return { x: (x - this.VP.x) / this.u, y: (y - this.VP.y) / this.u }; }

  private viewToScreen(p: Vec2): Vec2 { return { x: this.VP.x + p.x * this.u, y: this.VP.y + p.y * this.u }; }

  private project(x: number, y: number, z: number): { x: number; y: number; s: number } {
    const s = FOCAL / (FOCAL + Math.max(z, -1.8));
    return { x: this.VP.x + (x - this.cam.x) * s * this.u, y: this.VP.y + (y - this.cam.y) * s * this.u, s };
  }

  resize(): void {
    this.dpr = Math.min(devicePixelRatio || 1, 2);
    this.W = innerWidth;
    this.H = innerHeight;
    this.u = Math.min(this.H, this.W * 1.25) / 100;
    this.M = 12 * this.u;
    this.VP = { x: this.W / 2, y: this.H * 0.45 };
    for (const c of [this.canvas, this.vig, this.handLayer]) {
      c.width = Math.round(this.W * this.dpr);
      c.height = Math.round(this.H * this.dpr);
    }
    for (const c of [this.sky, this.mid]) {
      c.width = Math.round((this.W + 2 * this.M) * this.dpr);
      c.height = Math.round((this.H + 2 * this.M) * this.dpr);
    }
    this.drawSky();
    this.drawMid();
    this.drawVignette();
  }

  onEvent(e: GameEvent): void {
    switch (e.type) {
      case 'summon': this.burst(e.x, e.y, e.z, 'fire', 30, 30); break;
      case 'extinguish':
        for (let i = 0; i < 14; i++) this.emit(e.x + rnd(-3, 3), e.y, 0, rnd(-5, 5), rnd(-12, -4), 0, rnd(0.3, 0.6), rnd(1.5, 3), 'fire', 0.3);
        break;
      case 'throw': this.burst(e.x, e.y, e.z, 'fire', 28, 26); this.shake = Math.max(this.shake, 0.15); break;
      case 'hitEnemy':
      case 'killEnemy': this.burst(e.x, e.y, e.z, 'fire', 30, 40); break;
      case 'clash': this.burst(e.x, e.y, e.z, 'spirit', 24, 34); break;
      case 'blocked':
        this.burst(e.x, e.y, 0, 'spirit', 20, 30);
        this.burst(e.x, e.y, 0, 'fire', 14, 24);
        this.shake = Math.max(this.shake, 0.25);
        break;
      case 'playerHit': this.burst(e.x, e.y, 0, 'spirit', 26, 36); this.shake = 1; this.flash = 1; break;
    }
  }

  render(g: Game | null, dt: number): void {
    this.t += dt;
    this.cam = g ? g.cam : { x: 0, y: 0 };
    if (g) this.emitFromState(g, dt);
    this.updateParticles(dt);
    this.shake = Math.max(0, this.shake - dt * 2.5);
    this.flash = Math.max(0, this.flash - dt * 2);

    const c = this.ctx, { W, H, M, u } = this;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    if (this.shake > 0) c.translate(rnd(-1, 1) * this.shake * 1.2 * u, rnd(-1, 1) * this.shake * 1.2 * u);

    c.drawImage(this.sky, -M - this.cam.x * u * PAR_SKY, -M - this.cam.y * u * PAR_SKY, W + 2 * M, H + 2 * M);
    this.drawFloor();
    const mx = -M - this.cam.x * u * PAR_MID, my = -M - this.cam.y * u * PAR_MID;
    c.drawImage(this.mid, mx, my, W + 2 * M, H + 2 * M);
    this.drawLanterns(mx + M, my + M);

    if (g) [...g.enemies].sort((a, b) => b.z - a.z).forEach(e => this.drawEnemy(e));
    this.drawParticles(true);
    if (g) {
      this.drawProjectiles(g, true);
      this.drawLandingMarkers(g);
      this.drawHeldLight(g);
      this.drawProjectiles(g, false);
      if (g.hands) this.drawHands(g);
    }
    this.drawParticles(false);
    if (g?.fire.held && g.hands) {
      const p = this.viewToScreen(g.hands.center), r = 4.4 * u * (0.95 + 0.08 * Math.sin(this.t * 25));
      c.globalCompositeOperation = 'lighter';
      c.drawImage(SPR.fire[0], p.x - r, p.y - r, r * 2, r * 2);
      c.globalCompositeOperation = 'source-over';
    }

    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.drawImage(this.vig, 0, 0, W, H);
    if (this.flash > 0) {
      const gr = c.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.2, W / 2, H / 2, Math.max(W, H) * 0.7);
      gr.addColorStop(0, 'rgba(255,0,0,0)');
      gr.addColorStop(1, `rgba(255,30,30,${0.5 * this.flash})`);
      c.fillStyle = gr;
      c.fillRect(0, 0, W, H);
    }
  }

  // ---------- static layers ----------

  private drawSky(): void {
    const { W, H, M, u } = this, g = this.sky.getContext('2d')!, hz = this.VP.y, R = mulberry32(11);
    g.setTransform(this.dpr, 0, 0, this.dpr, this.dpr * M, this.dpr * M);
    g.clearRect(-M, -M, W + 2 * M, H + 2 * M);
    let gr = g.createLinearGradient(0, -M, 0, hz);
    gr.addColorStop(0, '#06061a'); gr.addColorStop(0.55, '#151131'); gr.addColorStop(1, '#3c1d33');
    g.fillStyle = gr;
    g.fillRect(-M, -M, W + 2 * M, hz + M + 1);
    for (let i = 0; i < 180; i++) {
      const x = R() * (W + 2 * M) - M, y = R() * hz * 0.8 - M, r = R() * 1.3 + 0.3;
      g.globalAlpha = 0.25 + R() * 0.75;
      g.fillStyle = '#fff';
      g.fillRect(x, y, r, r);
    }
    g.globalAlpha = 1;
    const mx = W * 0.8, my = H * 0.13, mr = 4.2 * u;
    gr = g.createRadialGradient(mx, my, 0, mx, my, mr * 6);
    gr.addColorStop(0, 'rgba(255,238,215,.28)'); gr.addColorStop(1, 'rgba(255,238,215,0)');
    g.fillStyle = gr;
    g.fillRect(mx - mr * 6, my - mr * 6, mr * 12, mr * 12);
    g.fillStyle = '#f3ead8';
    g.beginPath(); g.arc(mx, my, mr, 0, 7); g.fill();
    g.fillStyle = 'rgba(120,100,90,.12)';
    for (const [a, b, r] of [[-0.3, -0.2, 0.28], [0.25, 0.2, 0.22], [0.1, -0.4, 0.12]]) {
      g.beginPath(); g.arc(mx + a * mr, my + b * mr, r * mr, 0, 7); g.fill();
    }
    const ridge = (base: number, amp: number, col: string, seed: number, f1: number, f2: number) => {
      g.fillStyle = col;
      g.beginPath();
      g.moveTo(-M, hz + 2 * u);
      for (let x = -M; x <= W + M + 8; x += 6) {
        g.lineTo(x, base - amp * (0.55 + 0.3 * Math.sin(x * f1 + seed) + 0.15 * Math.sin(x * f2 + seed * 3)));
      }
      g.lineTo(W + M, hz + 2 * u);
      g.closePath();
      g.fill();
    };
    ridge(hz - 2 * u, 14 * u, '#231838', 1.3, 0.006, 0.021);
    ridge(hz, 9 * u, '#170f27', 4.1, 0.009, 0.03);
    gr = g.createLinearGradient(0, hz - 8 * u, 0, hz + 3 * u);
    gr.addColorStop(0, 'rgba(255,110,80,0)'); gr.addColorStop(0.7, 'rgba(255,110,80,.12)'); gr.addColorStop(1, 'rgba(255,110,80,0)');
    g.fillStyle = gr;
    g.fillRect(-M, hz - 8 * u, W + 2 * M, 11 * u);
  }

  private drawMid(): void {
    const { W, M, u } = this, g = this.mid.getContext('2d')!, hz = this.VP.y, k = u * 0.8;
    g.setTransform(this.dpr, 0, 0, this.dpr, this.dpr * M, this.dpr * M);
    g.clearRect(-M, -M, W + 2 * M, this.H + 2 * M);
    g.fillStyle = '#110c19';
    g.fillRect(-M, hz - 2.2 * u, W + 2 * M, 2.6 * u);
    for (let x = -M; x < W + M; x += 9 * u) g.fillRect(x, hz - 3.4 * u, 1.6 * u, 1.4 * u);
    this.lanterns = [];
    const roof = (l: number, r: number, y: number, h: number) => {
      const w = r - l;
      g.beginPath();
      g.moveTo(l - 1.5 * k, y - 2 * k);
      g.quadraticCurveTo(l + w * 0.06, y + 0.6 * k, l + w * 0.16, y + 0.6 * k);
      g.lineTo(r - w * 0.16, y + 0.6 * k);
      g.quadraticCurveTo(r - w * 0.06, y + 0.6 * k, r + 1.5 * k, y - 2 * k);
      g.lineTo(r - w * 0.24, y - h);
      g.lineTo(l + w * 0.24, y - h);
      g.closePath();
      g.fill();
    };
    const temple = (x: number, w: number) => {
      const base = hz - u, bodyH = 12 * k, b1 = base - 3 * k - bodyH, b2 = b1 - 12 * k;
      g.fillStyle = '#0e0a16';
      g.fillRect(x - w * 0.04, base - 3 * k, w * 1.08, 3 * k);
      g.fillRect(x + w * 0.1, b1, w * 0.8, bodyH);
      roof(x - w * 0.02, x + w * 1.02, b1, 5 * k);
      g.fillRect(x + w * 0.3, b2, w * 0.4, 7 * k);
      roof(x + w * 0.18, x + w * 0.82, b2, 4.5 * k);
      g.fillStyle = 'rgba(255,165,90,.22)';
      for (let i = 0; i < 3; i++) g.fillRect(x + w * (0.22 + i * 0.22), b1 + bodyH * 0.35, w * 0.1, bodyH * 0.4);
      g.fillRect(x + w * 0.44, b2 + 2 * k, w * 0.12, 3.5 * k);
      this.lanterns.push({ x: x + w * 0.12, y: b1 + 2.4 * k }, { x: x + w * 0.88, y: b1 + 2.4 * k });
    };
    temple(W * 0.02, W * 0.22);
    temple(W * 0.76, W * 0.22);
  }

  private drawVignette(): void {
    const g = this.vig.getContext('2d')!, { W, H } = this;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const gr = g.createRadialGradient(W / 2, H * 0.5, Math.min(W, H) * 0.35, W / 2, H * 0.5, Math.max(W, H) * 0.75);
    gr.addColorStop(0, 'rgba(0,0,0,0)'); gr.addColorStop(1, 'rgba(0,0,0,.7)');
    g.fillStyle = gr;
    g.fillRect(0, 0, W, H);
  }

  // ---------- live world ----------

  /** The floor is drawn every frame so its perspective follows your head. */
  private drawFloor(): void {
    const c = this.ctx, hz = this.VP.y;
    const gr = c.createLinearGradient(0, hz, 0, this.H);
    gr.addColorStop(0, '#24182b'); gr.addColorStop(1, '#0a080f');
    c.fillStyle = gr;
    c.fillRect(0, hz, this.W, this.H - hz);
    c.strokeStyle = 'rgba(255,200,160,.06)';
    c.lineWidth = 1;
    const step = (this.W * 0.09) / this.u;
    for (let i = -18; i <= 18; i++) {
      const a = this.project(i * step, FLOOR_Y, 16), b = this.project(i * step, FLOOR_Y, -0.4);
      c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
    }
    for (const z of [0, 0.5, 1.1, 1.9, 2.9, 4.2, 5.9, 8.1, 11, 15]) {
      const y = this.project(0, FLOOR_Y, z).y;
      c.beginPath(); c.moveTo(0, y); c.lineTo(this.W, y); c.stroke();
    }
  }

  private drawLanterns(ox: number, oy: number): void {
    const c = this.ctx, u = this.u;
    c.globalCompositeOperation = 'lighter';
    this.lanterns.forEach((l, i) => {
      const x = l.x + ox, y = l.y + oy, r = 7 * u * (0.75 + 0.25 * Math.sin(this.t * 9 + i * 2) * Math.sin(this.t * 5.3 + i));
      const gr = c.createRadialGradient(x, y, 0, x, y, r);
      gr.addColorStop(0, 'rgba(255,170,80,.55)'); gr.addColorStop(1, 'rgba(255,120,40,0)');
      c.fillStyle = gr;
      c.fillRect(x - r, y - r, r * 2, r * 2);
      c.fillStyle = '#ffb45e';
      c.fillRect(x - 0.6 * u, y - 0.9 * u, 1.2 * u, 1.8 * u);
    });
    c.globalCompositeOperation = 'source-over';
  }

  private drawEnemy(e: Enemy): void {
    const c = this.ctx, u = this.u, p = this.project(e.x, e.y, e.z), s = p.s, sx = p.x;
    const bob = Math.sin(e.t * 2 + e.phase) * 1.5 * u * s;
    const cy = p.y + bob, hgt = 52 * u * s, feet = this.project(e.x, FLOOR_Y, e.z).y;
    const alpha = e.appear * (1 - Math.min(1, e.dying));
    if (alpha <= 0) return;
    c.globalAlpha = alpha * 0.5;
    c.fillStyle = '#000';
    c.beginPath(); c.ellipse(sx, feet, 9 * u * s, 2 * u * s, 0, 0, 7); c.fill();
    c.globalCompositeOperation = 'lighter';
    c.globalAlpha = alpha;
    let gr = c.createRadialGradient(sx, cy, 0, sx, cy, hgt * 0.75);
    gr.addColorStop(0, `rgba(80,190,255,${0.22 + e.flash * 0.4})`); gr.addColorStop(1, 'rgba(80,190,255,0)');
    c.fillStyle = gr;
    c.fillRect(sx - hgt, cy - hgt, hgt * 2, hgt * 2);
    c.globalCompositeOperation = 'source-over';
    // robe
    const top = cy - hgt * 0.42, hem = cy + hgt * 0.45, w = 11 * u * s;
    gr = c.createLinearGradient(0, top, 0, hem);
    gr.addColorStop(0, e.flash > 0 ? 'rgba(255,230,200,.95)' : 'rgba(175,235,255,.88)');
    gr.addColorStop(0.6, 'rgba(70,130,210,.55)'); gr.addColorStop(1, 'rgba(40,70,160,0)');
    c.fillStyle = gr;
    c.beginPath();
    c.moveTo(sx - w * 0.45, top + hgt * 0.12);
    c.quadraticCurveTo(sx, top - hgt * 0.08, sx + w * 0.45, top + hgt * 0.12);
    c.lineTo(sx + w * 0.8, top + hgt * 0.28);
    c.lineTo(sx + w * 0.55, top + hgt * 0.4);
    for (let i = 0; i <= 8; i++) {
      const k = i / 8;
      c.lineTo(sx + w * (0.6 - 1.2 * k), hem + Math.sin(k * 12 + e.t * 5) * 1.6 * u * s);
    }
    c.lineTo(sx - w * 0.55, top + hgt * 0.4);
    c.lineTo(sx - w * 0.8, top + hgt * 0.28);
    c.closePath();
    c.fill();
    // mask
    const hy = top + hgt * 0.03, hr = 4.6 * u * s;
    c.fillStyle = '#eaf4f6';
    c.beginPath(); c.ellipse(sx, hy, hr * 0.85, hr, 0, 0, 7); c.fill();
    c.fillStyle = '#0a1a28';
    c.beginPath(); c.ellipse(sx - hr * 0.35, hy - hr * 0.1, hr * 0.22, hr * 0.1, 0.35, 0, 7); c.fill();
    c.beginPath(); c.ellipse(sx + hr * 0.35, hy - hr * 0.1, hr * 0.22, hr * 0.1, -0.35, 0, 7); c.fill();
    c.fillStyle = '#c0392b';
    c.fillRect(sx - hr * 0.08, hy + hr * 0.3, hr * 0.16, hr * 0.35);
    // wind-up telegraph
    if (e.winding) {
      const o = this.project(e.x + e.side * 11, e.y - 14, e.z), r = (1.2 + e.wind * 3.5) * u * s;
      const ox = o.x, oy = o.y + bob;
      c.globalCompositeOperation = 'lighter';
      c.drawImage(SPR.spirit[1], ox - r * 2.2, oy - r * 2.2, r * 4.4, r * 4.4);
      c.drawImage(SPR.spirit[0], ox - r, oy - r, r * 2, r * 2);
      c.strokeStyle = `rgba(140,230,255,${0.6 * (1 - e.wind)})`;
      c.lineWidth = 2;
      c.beginPath(); c.arc(ox, oy, r * (4 - e.wind * 2.5), 0, 7); c.stroke();
      c.globalCompositeOperation = 'source-over';
    }
    c.globalAlpha = 1;
  }

  private drawProjectiles(g: Game, far: boolean): void {
    const c = this.ctx;
    c.globalCompositeOperation = 'lighter';
    for (const p of g.projs) {
      if ((p.z > 1) !== far) continue;
      const q = this.project(p.x, p.y, p.z), r = p.r * q.s * this.u, set = p.kind === 'player' ? SPR.fire : SPR.spirit;
      c.drawImage(set[1], q.x - r * 2.4, q.y - r * 2.4, r * 4.8, r * 4.8);
      c.drawImage(set[0], q.x - r * 1.2, q.y - r * 1.2, r * 2.4, r * 2.4);
    }
    c.globalCompositeOperation = 'source-over';
  }

  /** Where each incoming attack will land if you stay still; red = it would hit you. */
  private drawLandingMarkers(g: Game): void {
    const c = this.ctx;
    for (const p of g.projs) {
      if (p.kind !== 'enemy' || p.z > 5 || p.z < 0.2) continue;
      const a = arrival(p), q = this.project(a.x, a.y, 0), near = 1 - p.z / 5, danger = g.isThreat(p);
      c.strokeStyle = danger ? `rgba(255,80,70,${0.25 + near * 0.65})` : `rgba(130,220,255,${near * 0.6})`;
      c.lineWidth = danger ? 3 : 2;
      const r = p.r * this.u * (1.3 + (1 - near) * 2.5);
      c.beginPath(); c.arc(q.x, q.y, r, 0, 7); c.stroke();
      if (danger) { c.beginPath(); c.arc(q.x, q.y, r * 0.35, 0, 7); c.stroke(); }
    }
  }

  /** Warm light around held fire, and the shield's flame sheet. */
  private drawHeldLight(g: Game): void {
    const h = g.hands;
    if (!h || (!g.fire.held && !g.shield.on)) return;
    const c = this.ctx, u = this.u, C = this.viewToScreen(h.center);
    const glow = 0.9 + 0.1 * Math.sin(this.t * 20), r = (g.shield.on ? 55 : 40) * u;
    c.globalCompositeOperation = 'lighter';
    const gr = c.createRadialGradient(C.x, C.y, 0, C.x, C.y, r);
    gr.addColorStop(0, `rgba(255,140,60,${0.28 * glow})`); gr.addColorStop(1, 'rgba(255,120,40,0)');
    c.fillStyle = gr;
    c.fillRect(C.x - r, C.y - r, r * 2, r * 2);
    if (g.shield.on) {
      const a = this.viewToScreen(h.l), b = this.viewToScreen(h.r), e = g.shield.energy, hgt = (14 + e * 10) * u;
      const sg = c.createLinearGradient(0, a.y, 0, a.y - hgt);
      sg.addColorStop(0, `rgba(255,150,60,${0.35 * e + 0.1})`); sg.addColorStop(1, 'rgba(255,90,30,0)');
      c.fillStyle = sg;
      c.beginPath();
      c.moveTo(a.x, a.y + 2 * u);
      c.lineTo(b.x, b.y + 2 * u);
      for (let i = 0; i <= 10; i++) {
        const k = 1 - i / 10;
        c.lineTo(lerp(a.x, b.x, k), lerp(a.y, b.y, k) - hgt * (0.75 + 0.25 * Math.sin(k * 14 + this.t * 12)));
      }
      c.closePath();
      c.fill();
    }
    c.globalCompositeOperation = 'source-over';
  }

  /** First-person forearm and hand in screen space. side: −1 left, +1 right. */
  private handShape(c: CanvasRenderingContext2D, h: Vec2, side: number, grow: number, open: boolean): void {
    const k = 1.5 * this.u, g = grow;
    c.lineCap = 'round';
    c.lineJoin = 'round';
    const line = (x1: number, y1: number, x2: number, y2: number, w: number) => {
      c.lineWidth = w + 2 * g;
      c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
    };
    const elbow = { x: h.x + side * 16 * k, y: this.H + 14 * k };
    const mid = { x: lerp(elbow.x, h.x, 0.55), y: lerp(elbow.y, h.y + 4 * k, 0.55) };
    line(elbow.x, elbow.y, mid.x, mid.y, 9 * k);
    line(mid.x, mid.y, h.x, h.y + 3.5 * k, 6.4 * k);
    const pw = open ? 5.4 * k : 3.4 * k;
    c.beginPath(); c.ellipse(h.x, h.y, pw / 2 + g, 3.6 * k + g, 0, 0, 7); c.fill();
    const fan = open ? 0.2 : 0.06, curl = open ? 0 : -side * 0.28;
    for (let i = 0; i < 4; i++) {
      const a = -Math.PI / 2 + (i - 1.5) * fan + curl, len = (i === 1 || i === 2 ? 5.2 : 4.4) * k;
      const bx = h.x + (i - 1.5) * pw * 0.26, by = h.y - 2.6 * k;
      line(bx, by, bx + Math.cos(a) * len, by + Math.sin(a) * len, 1.6 * k);
    }
    const ta = -Math.PI / 2 - side * (open ? 1.05 : 0.7), tx = h.x - side * pw * 0.42, ty = h.y + 0.4 * k;
    line(tx, ty, tx + Math.cos(ta) * 3.6 * k, ty + Math.sin(ta) * 3.6 * k, 1.9 * k);
  }

  private drawHands(g: Game): void {
    const h = g.hands!, c = this.ctx;
    const L = this.viewToScreen(h.l), R = this.viewToScreen(h.r), C = this.viewToScreen(h.center);
    const open = g.shield.on, glow = g.fire.held || g.shield.on ? 0.9 + 0.1 * Math.sin(this.t * 20) : 0.15;
    // rim glow, then the dark hands lit by the fire on top
    c.strokeStyle = c.fillStyle = `rgba(255,130,60,${0.12 + 0.28 * glow})`;
    this.handShape(c, L, -1, 0.9 * this.u, open);
    this.handShape(c, R, 1, 0.9 * this.u, open);
    const hl = this.handLayer.getContext('2d')!;
    hl.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    hl.globalCompositeOperation = 'source-over';
    hl.clearRect(0, 0, this.W, this.H);
    hl.strokeStyle = hl.fillStyle = g.inv > 0 && Math.sin(this.t * 40) > 0 ? '#3a1216' : '#150f19';
    this.handShape(hl, L, -1, 0, open);
    this.handShape(hl, R, 1, 0, open);
    hl.globalCompositeOperation = 'source-atop';
    const gr = hl.createRadialGradient(C.x, C.y - 4 * this.u, 0, C.x, C.y, 50 * this.u);
    gr.addColorStop(0, `rgba(255,160,80,${0.75 * glow})`); gr.addColorStop(1, 'rgba(255,90,30,0)');
    hl.fillStyle = gr;
    hl.fillRect(0, 0, this.W, this.H);
    c.drawImage(this.handLayer, 0, 0, this.W, this.H);
  }

  // ---------- particles ----------

  private emit(x: number, y: number, z: number, vx: number, vy: number, vz: number, life: number, size: number, pal: Pal = 'fire', rise = 1): void {
    if (this.parts.length < MAX_PARTICLES) this.parts.push({ x, y, z, vx, vy, vz, life, max: life, size, pal, rise });
  }

  private burst(x: number, y: number, z: number, pal: Pal, n: number, spd: number): void {
    for (let i = 0; i < n; i++) {
      const a = Math.random() * 6.283, sp = rnd(0.3, 1) * spd;
      this.emit(x, y, z, Math.cos(a) * sp, Math.sin(a) * sp, rnd(-1, 1), rnd(0.25, 0.65), rnd(2, 4.5), pal, 0.5);
    }
  }

  private emitFromState(g: Game, dt: number): void {
    const h = g.hands;
    if (h) {
      const vx0 = h.vel.x * 0.3, vy0 = h.vel.y * 0.3;
      if (g.fire.held) {
        const w = g.handWorld(h.center), r = 4.2;
        for (let i = nOf(140, dt); i > 0; i--) {
          const a = Math.random() * 6.283, d = Math.sqrt(Math.random()) * r * 0.55;
          this.emit(w.x + Math.cos(a) * d, w.y + Math.sin(a) * d, 0, rnd(-6, 6) + vx0, rnd(-16, -5) + vy0, 0, rnd(0.3, 0.6), r * rnd(0.55, 1));
        }
      }
      if (g.shield.on) {
        const a = g.handWorld(h.l), b = g.handWorld(h.r), e = g.shield.energy;
        for (let i = nOf(260, dt); i > 0; i--) {
          const k = Math.random();
          this.emit(lerp(a.x, b.x, k) + rnd(-1, 1), lerp(a.y, b.y, k) + rnd(-2, 3), 0,
            rnd(-3, 3) + vx0, rnd(-34, -14) * (0.6 + e * 0.6) + vy0, 0, rnd(0.25, 0.5), (3 + e * 2) * rnd(0.7, 1.1));
        }
      }
    }
    for (const p of g.projs) {
      const pal: Pal = p.kind === 'player' ? 'fire' : 'spirit';
      for (let j = nOf(p.kind === 'player' ? 120 : 90, dt); j > 0; j--) {
        this.emit(p.x + rnd(-0.4, 0.4) * p.r, p.y + rnd(-0.4, 0.4) * p.r, p.z, rnd(-3, 3), rnd(-6, 2), p.vz * 0.25, rnd(0.2, 0.45), p.r * rnd(0.6, 1), pal, 0.6);
      }
    }
    for (const e of g.enemies) {
      if (e.hp > 0) continue;
      const s = FOCAL / (FOCAL + e.z);
      for (let j = nOf(120, dt); j > 0; j--) {
        this.emit(e.x + rnd(-8, 8), e.y + rnd(-20, 20), e.z, rnd(-10, 10) / s, rnd(-30, -5) / s, 0, rnd(0.4, 0.8), rnd(3, 6) / s, 'spirit', 0.6);
      }
    }
  }

  private updateParticles(dt: number): void {
    const drag = Math.exp(-2.2 * dt), parts = this.parts;
    for (let i = parts.length - 1; i >= 0; i--) {
      const p = parts[i];
      p.life -= dt;
      if (p.life <= 0) { parts[i] = parts[parts.length - 1]; parts.pop(); continue; }
      p.vy -= p.rise * 38 * dt;
      p.vx *= drag;
      p.vy *= drag;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z += p.vz * dt;
    }
  }

  private drawParticles(far: boolean): void {
    const c = this.ctx;
    c.globalCompositeOperation = 'lighter';
    for (const p of this.parts) {
      if ((p.z > 1) !== far) continue;
      const q = this.project(p.x, p.y, p.z), k = 1 - p.life / p.max, idx = k < 0.28 ? 0 : k < 0.62 ? 1 : 2;
      const sz = p.size * (1 - k * 0.55) * q.s * this.u;
      c.globalAlpha = k < 0.12 ? k / 0.12 : 1 - (k - 0.12) / 0.88;
      c.drawImage(SPR[p.pal][idx], q.x - sz, q.y - sz, sz * 2, sz * 2);
    }
    c.globalAlpha = 1;
    c.globalCompositeOperation = 'source-over';
  }
}
```

- [ ] **Step 3: HUD and debug view**

```ts
// file: web/src/render/hud.ts
import type { Game, GameEvent } from '../game/game';

const $ = (id: string) => document.getElementById(id)!;

/** DOM overlay: health, score, current move, shield energy, toasts and wave banners. */
export class Hud {
  update(g: Game): void {
    $('hpFill').style.width = `${g.hp}%`;
    $('score').textContent = String(g.score);
    $('wave').textContent = `Wave ${g.wave}`;
    const pill = $('pill');
    pill.classList.toggle('off', !g.fire.held && !g.shield.on);
    pill.classList.toggle('shield', g.shield.on);
    const [name, hint] = g.shield.on ? ['FLAME SHIELD', 'cover the red rings · drains while held']
      : g.fire.held ? ['FIREBALL', 'push toward the screen to throw']
        : ['NO FIRE', 'raise hands, palms together · or spread wide to shield'];
    $('modeName').textContent = name;
    $('modeHint').textContent = hint;
    $('shieldFill').style.width = `${Math.round(g.shield.energy * 100)}%`;
    $('shieldBar').classList.toggle('broken', g.shield.broken > 0);
    $('shieldState').textContent = g.shield.broken > 0 ? 'broken' : g.shield.on ? 'holding' : g.shield.energy < 1 ? 'recharging' : 'ready';
  }

  onEvent(e: GameEvent): void {
    switch (e.type) {
      case 'blocked': this.toast('BLOCKED', 'cool'); break;
      case 'dodged': this.toast('DODGED', 'cool'); break;
      case 'clash': this.toast('CLASH', 'cool'); break;
      case 'playerHit': this.toast('HIT', 'bad'); break;
      case 'shieldBroken': this.toast('SHIELD BROKEN', 'bad'); break;
      case 'killEnemy': this.toast('+100'); break;
      case 'wave': this.banner(`WAVE ${e.wave}`); break;
    }
  }

  private toast(text: string, cls = ''): void {
    const box = $('toasts');
    while (box.children.length > 3) box.firstChild!.remove();
    const d = document.createElement('div');
    d.className = `toast ${cls}`;
    d.textContent = text;
    box.appendChild(d);
    setTimeout(() => d.remove(), 1000);
  }

  private banner(text: string): void {
    const b = $('banner');
    b.textContent = text;
    b.classList.remove('show');
    void b.offsetWidth; // restart the CSS animation
    b.classList.add('show');
  }
}
```

```ts
// file: web/src/render/debug.ts
import type { TrackingFrame } from '../input/types';
import type { Intent } from '../intent/interpret';
import type { Vec2 } from '../math';

/** Corner panel: what the camera sees plus the tracked points; backtick adds live numbers. */
export class DebugView {
  private ctx: CanvasRenderingContext2D;
  private detailed = false;

  constructor(private canvas: HTMLCanvasElement, private text: HTMLElement) {
    this.ctx = canvas.getContext('2d')!;
    this.resize();
  }

  resize(): void {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.canvas.clientWidth * dpr);
    this.canvas.height = Math.round(this.canvas.clientHeight * dpr);
  }

  toggle(): void {
    this.detailed = !this.detailed;
    this.text.classList.toggle('hidden', !this.detailed);
  }

  draw(f: TrackingFrame | null, intent: Intent | null, video: HTMLVideoElement | null): void {
    const c = this.ctx, w = this.canvas.width, h = this.canvas.height;
    c.fillStyle = '#05070a';
    c.fillRect(0, 0, w, h);
    if (video && video.readyState >= 2) {
      c.save();
      c.globalAlpha = 0.6;
      c.translate(w, 0);
      c.scale(-1, 1); // mirror, to match the tracking frame
      c.drawImage(video, 0, 0, w, h);
      c.restore();
    }
    if (f) {
      const P = (p: Vec2) => ({ x: p.x * w, y: p.y * h });
      c.lineWidth = 2;
      if (f.shoulderL && f.shoulderR) {
        const a = P(f.shoulderL), b = P(f.shoulderR);
        c.strokeStyle = '#9dffcf';
        c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
      }
      if (f.head) {
        const p = P(f.head);
        c.fillStyle = '#fff';
        c.beginPath(); c.arc(p.x, p.y, 5, 0, 7); c.fill();
      }
      c.strokeStyle = '#ffc36b';
      for (const hand of f.hands) {
        const p = P(hand.center);
        c.beginPath(); c.arc(p.x, p.y, Math.max(3, hand.size * w * 0.5), 0, 7); c.stroke();
      }
    }
    if (this.detailed && intent) {
      const hd = intent.hands;
      this.text.textContent = [
        `present ${intent.present}  raised ${intent.raised}`,
        `head    x ${intent.head.x.toFixed(1)}  y ${intent.head.y.toFixed(1)}`,
        hd ? `hands   ${hd.center.x.toFixed(1)}, ${hd.center.y.toFixed(1)}  spread ${hd.spread.toFixed(1)}` : 'hands   —',
        hd ? `vel     ${hd.vel.x.toFixed(0)}, ${hd.vel.y.toFixed(0)}` : '',
        f?.hands.length ? `size    ${f.hands.map(x => x.size.toFixed(3)).join('  ')}` : '',
      ].join('\n');
    }
  }
}
```

- [ ] **Step 4: Camera tracker**

```ts
// file: web/src/input/camera.ts
import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import { toFrame } from './landmarks';
import type { Tracker, TrackingFrame } from './types';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

/** A failure we can explain to the player. */
export class CameraError extends Error {}

/** Webcam + MediaPipe hand and pose models, polled once per new video frame. */
export class CameraTracker implements Tracker {
  private lastVideoTime = -1;

  private constructor(
    readonly video: HTMLVideoElement,
    private hands: HandLandmarker,
    private pose: PoseLandmarker,
    private stream: MediaStream,
  ) {}

  static async create(onStatus: (message: string) => void): Promise<CameraTracker> {
    onStatus('Asking for camera access…');
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
    } catch (e) {
      const blocked = e instanceof DOMException && e.name === 'NotAllowedError';
      throw new CameraError(blocked ? 'Camera access was blocked.' : 'No camera was found.');
    }
    const video = document.createElement('video');
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();

    onStatus('Loading hand and pose tracking…');
    try {
      const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
      const [hands, pose] = await Promise.all([
        HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        }),
        PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numPoses: 1,
        }),
      ]);
      return new CameraTracker(video, hands, pose, stream);
    } catch {
      stream.getTracks().forEach(t => t.stop());
      throw new CameraError('Could not load the tracking models. Check your internet connection.');
    }
  }

  poll(now: number): TrackingFrame | null {
    if (this.video.readyState < 2 || this.video.currentTime === this.lastVideoTime) return null;
    this.lastVideoTime = this.video.currentTime;
    const hands = this.hands.detectForVideo(this.video, now);
    const pose = this.pose.detectForVideo(this.video, now);
    return toFrame(now / 1000, hands.landmarks, pose.landmarks[0]);
  }

  dispose(): void {
    this.stream.getTracks().forEach(t => t.stop());
    this.hands.close();
    this.pose.close();
  }
}
```

- [ ] **Step 5: Main — screens and loop**

```ts
// file: web/src/main.ts
import './style.css';
import { Game } from './game/game';
import { CameraError, CameraTracker } from './input/camera';
import { bindMockControls, MOCK_CALIBRATION, MockTracker } from './input/mock';
import type { Tracker, TrackingFrame } from './input/types';
import { Calibrator, type Calibration } from './intent/calibration';
import { initialState, interpret, type Intent, type InterpretState } from './intent/interpret';
import { DebugView } from './render/debug';
import { Hud } from './render/hud';
import { Renderer } from './render/renderer';

const STEP = 1 / 60;
const $ = (id: string) => document.getElementById(id)!;
const show = (id: string, on = true) => $(id).classList.toggle('hidden', !on);

const renderer = new Renderer($('game') as HTMLCanvasElement);
const hud = new Hud();
const debug = new DebugView($('pip') as HTMLCanvasElement, $('debugText'));

let phase: 'menu' | 'loading' | 'calibrating' | 'play' = 'menu';
let tracker: Tracker | null = null;
let camera: CameraTracker | null = null;
let calibrator = new Calibrator();
let calibration: Calibration | null = null;
let istate: InterpretState = initialState();
let intent: Intent | null = null;
let pendingThrow = false;
let lastFrame: TrackingFrame | null = null;
let game: Game | null = null;
let acc = 0, last = performance.now(), fpsTime = 0, fpsFrames = 0;

function startMock(): void {
  const mock = new MockTracker(renderer);
  mock.setMouse(innerWidth / 2, innerHeight * 0.7);
  bindMockControls(mock, $('game'));
  tracker = mock;
  calibration = MOCK_CALIBRATION;
  show('start', false);
  show('status', false);
  show('mockHelp');
  beginPlay();
}

async function startCamera(): Promise<void> {
  phase = 'loading';
  show('start', false);
  show('status');
  show('statusFallback', false);
  try {
    camera = await CameraTracker.create(message => { $('statusText').textContent = message; });
    tracker = camera;
    show('status', false);
    beginCalibration();
  } catch (e) {
    const why = e instanceof CameraError ? e.message : 'Something went wrong starting the camera.';
    $('statusText').textContent = `${why} You can still play with mouse and keys.`;
    show('statusFallback');
    phase = 'menu';
  }
}

function beginCalibration(): void {
  calibrator = new Calibrator();
  phase = 'calibrating';
  game = null;
  show('over', false);
  show('away', false);
  show('calib');
  $('calibFill').style.width = '0%';
}

function beginPlay(): void {
  istate = initialState();
  intent = null;
  pendingThrow = false;
  acc = 0;
  game = new Game(Math.random, renderer.viewHalfW);
  phase = 'play';
  show('calib', false);
  show('over', false);
}

function onFrame(f: TrackingFrame): void {
  lastFrame = f;
  if (phase === 'calibrating') {
    $('calibFill').style.width = `${Math.round(calibrator.add(f) * 100)}%`;
    const result = calibrator.result();
    if (result) {
      calibration = result;
      beginPlay();
    }
  } else if (phase === 'play' && calibration) {
    intent = interpret(f, calibration, istate);
    if (intent.throwNow) pendingThrow = true; // held until the next fixed step consumes it
  }
}

function stepGame(dt: number): void {
  if (!game || !intent) return;
  show('away', !intent.present);
  const paused = !$('mockHelp').classList.contains('hidden');
  if (!intent.present || paused) { acc = 0; return; }
  acc += dt;
  while (acc >= STEP) {
    game.step(STEP, { ...intent, throwNow: pendingThrow });
    pendingThrow = false;
    acc -= STEP;
  }
  for (const e of game.drainEvents()) {
    renderer.onEvent(e);
    hud.onEvent(e);
    if (e.type === 'gameOver') {
      $('overScore').textContent = String(game.score);
      show('over');
    }
  }
  hud.update(game);
}

function headLabel(i: Intent | null): string {
  if (!i) return '—';
  if (i.head.y > 10) return 'duck';
  if (i.head.x < -10) return 'left';
  if (i.head.x > 10) return 'right';
  return 'center';
}

function loop(now: number): void {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  fpsTime += dt;
  fpsFrames++;
  if (fpsTime > 0.5) {
    $('fps').textContent = `${Math.round(fpsFrames / fpsTime)} fps`;
    fpsTime = 0;
    fpsFrames = 0;
  }
  const f = tracker?.poll(now);
  if (f) onFrame(f);
  if (phase === 'play') stepGame(dt);
  renderer.render(phase === 'play' ? game : null, dt);
  debug.draw(lastFrame, intent, camera?.video ?? null);
  $('handsN').textContent = String(lastFrame?.hands.length ?? 0);
  $('headTag').textContent = headLabel(intent);
  requestAnimationFrame(loop);
}

$('camBtn').addEventListener('click', () => void startCamera());
$('mockBtn').addEventListener('click', startMock);
$('statusFallback').addEventListener('click', startMock);
$('againBtn').addEventListener('click', beginPlay);
$('mockHelpClose').addEventListener('click', () => show('mockHelp', false));
addEventListener('resize', () => {
  renderer.resize();
  debug.resize();
  if (game) game.viewHalfW = renderer.viewHalfW;
});
addEventListener('keydown', e => {
  if (e.repeat) return;
  const k = e.key.toLowerCase();
  if (k === '`') debug.toggle();
  if (k === 'r' && game?.state === 'over') beginPlay();
  if (k === 'c' && camera && phase === 'play') beginCalibration();
  if ((k === '?' || k === '/') && tracker instanceof MockTracker) $('mockHelp').classList.toggle('hidden');
});

if (new URLSearchParams(location.search).get('input') === 'mock') startMock();
requestAnimationFrame(loop);
```

- [ ] **Step 6: Type-check, test and build**

Run: `cd web && npx tsc --noEmit && npm test && npm run build`
Expected: no type errors, all tests PASS, `dist/` built

- [ ] **Step 7: Verify in the browser (mock mode)**

Run: `cd web && npm run dev`, open `http://localhost:5173/?input=mock`.
Expected: courtyard renders; moving the mouse moves the fire-hands; `1` makes fire; click throws it at a spirit; `2` raises a shield that drains; `A`/`D`/`S` shift the view with parallax; red rings appear for incoming attacks; getting hit flashes red and lowers vitality; waves advance; game over + `R` restarts. No console errors.

- [ ] **Step 8: Verify with the camera**

Open `http://localhost:5173/`, click **Play with camera**, allow access.
Expected: status messages while loading; calibration bar fills while standing still; corner panel shows the mirrored video with head, shoulder and hand markers; leaning moves the view; palms together makes fire; a quick push throws; hands wide shields; stepping out of frame shows "Step into frame". Denying camera access shows the message and the mouse & keys button.

- [ ] **Step 9: Commit**

```bash
git add web/index.html web/src
git commit -m "feat(web): first-person renderer, HUD, camera tracking and game screens"
```

---

### Task 7: Docs and playtest checklist

**Files:**
- Modify: `README.md` (append a section)
- Create: `web/PLAYTEST.md`

- [ ] **Step 1: Append to README.md**

~~~markdown
## Web game: first-person firebending (`web/`)

A browser version played with just your webcam: your head moves the camera, your hands make, throw and shield with fire.

```bash
cd web
npm install
npm run dev        # http://localhost:5173 (add ?input=mock to play with mouse & keys)
npm test
```

Design: `docs/superpowers/specs/2026-09-26-first-person-firebending-design.md` · visual mockup: `mockup/index.html`
~~~

- [ ] **Step 2: Create the playtest checklist**

```markdown
<!-- file: web/PLAYTEST.md -->
# Playtest checklist

Run `npm run dev`, play with the camera in Chrome. Note the result of each item.

- [ ] Frame rate (corner panel) stays ≥ 30 fps while playing.
- [ ] Fire follows your hands without noticeable lag.
- [ ] Palms together makes fire at 1 m, 1.5 m and 2.5 m from the camera.
- [ ] A quick push throws; slowly moving hands toward the camera does not.
- [ ] Spreading hands raises the shield; it blocks an attack you cover.
- [ ] Leaning and ducking dodge attacks aimed at your head.
- [ ] Walking out of frame pauses with "Step into frame".
- [ ] A 3-minute run is fun and doesn't wear out your arms.

Tuning knobs: `TUNING` in `src/intent/interpret.ts` (tracking feel), `TUNE` in `src/game/game.ts` (gameplay). Press `` ` `` in game for live numbers.
```

- [ ] **Step 3: Commit**

```bash
git add README.md web/PLAYTEST.md
git commit -m "docs: how to run the web game and playtest checklist"
```

---

## Self-Review Notes

- Spec coverage: fireball (Tasks 3–4, 6), shield (4, 6), dodge + landing rings (4, 6), drop (4), waves/score/game over (4, 6), calibration (2, 6), mock mode (5, 6), debug overlay (6), camera errors + fallback (6), "step into frame" (6), unit tests for interpret and game (3, 4), playtest checklist (7).
- Types used across tasks: `TrackingFrame`, `HandObs`, `Calibration`, `Intent`, `HandsIntent`, `TUNING`, `Game`, `GameEvent`, `Proj`, `Enemy`, `FOCAL`, `FLOOR_Y`, `arrival`, `bodyHit` — defined once, names consistent.
