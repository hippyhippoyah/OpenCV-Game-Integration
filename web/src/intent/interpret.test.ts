import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { initialState, interpret, TUNING, type Intent } from './interpret';
import type { Calibration } from './calibration';
import { arm, bodyFrame, hand, type HandSpec } from '../test/frames';
import type { Vec2 } from '../math';

const cal: Calibration = { head: { x: 0.5, y: 0.35 }, sw: 0.2 };
const MID = { x: 0.5, y: 0.5 }, SW = 0.2, FPS = 30;
const GUARD_L: HandSpec = { x: -0.3, y: 0.1 }, GUARD_R: HandSpec = { x: 0.3, y: 0.1 };

type Pair = HandSpec[];
const repeat = <T>(n: number, f: (i: number) => T): T[] => Array.from({ length: n }, (_, i) => f(i));
/** Shoulder-width position → view units, as interpret maps it. */
const view = (p: HandSpec): Vec2 => ({ x: p.x * TUNING.handScaleX, y: TUNING.handOffsetY + p.y * TUNING.handScaleY });

/** Feed frames of visible hands at 30 fps; returns every intent. */
function play(frames: Pair[], s = initialState()): Intent[] {
  return frames.map((hs, i) => interpret(bodyFrame(i / FPS, { hands: hs.map(h => hand(MID, SW, h)) }), cal, s));
}
const punchesIn = (out: Intent[]) => out.flatMap(o => o.punches);

/** Right fist at guard, snaps to `to` over 4 frames, then opens and holds. */
function rightPunch(to: HandSpec): Pair[] {
  return [
    ...repeat(6, () => [GUARD_L, GUARD_R]),
    ...repeat(4, i => [GUARD_L, { x: GUARD_R.x + ((to.x - GUARD_R.x) * (i + 1)) / 4, y: GUARD_R.y + ((to.y - GUARD_R.y) * (i + 1)) / 4 }]),
    ...repeat(10, () => [GUARD_L, { ...to, open: 1 }]),
  ];
}

describe('interpret', () => {
  afterEach(() => { TUNING.punchTrigger = 'extend'; });

  it('maps hands relative to the shoulders, independent of distance to the camera', () => {
    const at = (mid: Vec2, sw: number) =>
      interpret(bodyFrame(0, { mid, sw, hands: [hand(mid, sw, { x: -0.5, y: -0.5 }), hand(mid, sw, { x: 0.5, y: -0.5 })] }), cal, initialState());
    for (const r of [at(MID, 0.2), at({ x: 0.5, y: 0.45 }, 0.1)]) {
      expect(r.hands.l!.pos.x).toBeCloseTo(-0.5 * TUNING.handScaleX);
      expect(r.hands.r!.pos.x).toBeCloseTo(0.5 * TUNING.handScaleX);
      expect(r.hands.l!.pos.y).toBeCloseTo(TUNING.handOffsetY - 0.5 * TUNING.handScaleY);
    }
  });

  it('turns head offset into camera lean and duck', () => {
    const r = interpret(bodyFrame(0, { head: { x: 0.6, y: 0.4 } }), cal, initialState());
    expect(r.head.x).toBeCloseTo(0.5 * TUNING.leanUnitsPerSw);
    expect(r.head.y).toBeCloseTo(0.25 * TUNING.duckUnitsPerSw);
  });

  it('reads fist vs open with hysteresis and passes palm facing through', () => {
    const out = play([
      ...repeat(5, () => [GUARD_L, GUARD_R]),
      ...repeat(5, () => [{ ...GUARD_L, open: 0.5 }, GUARD_R]), // half-open still counts as a fist
      ...repeat(8, () => [{ ...GUARD_L, open: 1, facing: 0.2 }, GUARD_R]),
    ]);
    expect(out[4].hands.l!.open).toBe(false);
    expect(out[9].hands.l!.open).toBe(false);
    expect(out[17].hands.l!.open).toBe(true);
    expect(out[17].hands.l!.facing).toBeCloseTo(0.2, 1);
  });

  it('fires one punch from the hand that opens at the end of a fast move, where it opened', () => {
    TUNING.punchTrigger = 'open';
    const to = { x: -0.1, y: -0.6 };
    const p = punchesIn(play(rightPunch(to)));
    expect(p).toHaveLength(1);
    expect(p[0].hand).toBe('r');
    expect(Math.abs(p[0].at.x - view(to).x)).toBeLessThan(2);
    expect(Math.abs(p[0].at.y - view(to).y)).toBeLessThan(2);
    expect(p[0].shoulder.x).toBeCloseTo(0.5 * TUNING.handScaleX);
  });

  it('does not punch when a still fist simply opens', () => {
    TUNING.punchTrigger = 'open';
    const out = play([...repeat(6, () => [GUARD_L, GUARD_R]), ...repeat(10, () => [GUARD_L, { ...GUARD_R, open: 1 }])]);
    expect(punchesIn(out)).toHaveLength(0);
  });

  it('does not punch with hands down at rest', () => {
    TUNING.punchTrigger = 'open';
    const low = { x: 0.3, y: 1.2 };
    const out = play([...repeat(6, () => [GUARD_L, GUARD_R]), ...rightPunch(low).slice(6)]);
    expect(punchesIn(out)).toHaveLength(0);
  });

  it('opening both hands raises a shield instead of punching', () => {
    const out = play([
      ...repeat(6, () => [GUARD_L, GUARD_R]),
      ...repeat(3, i => [{ x: -0.3 - 0.15 * (i + 1), y: 0.1 - 0.2 * (i + 1) }, { x: 0.3 + 0.15 * (i + 1), y: 0.1 - 0.2 * (i + 1) }]),
      [{ x: -0.75, y: -0.5 }, { x: 0.75, y: -0.5, open: 1 }],
      ...repeat(16, () => [{ x: -0.75, y: -0.5, open: 1 }, { x: 0.75, y: -0.5, open: 1 }]),
    ]);
    expect(punchesIn(out)).toHaveLength(0);
    expect(out.flatMap(o => o.casts)).toHaveLength(0); // spreading before opening is not an ultimate
    expect(out[11].shield).toBe(false);
    expect(out[out.length - 1].shield).toBe(true);
  });

  it("keeps each hand's identity when a punch crosses the body", () => {
    const frames: Pair[] = [
      ...repeat(5, () => [GUARD_L, GUARD_R]),
      ...repeat(20, i => [GUARD_L, { x: 0.3 - (0.9 * (i + 1)) / 20, y: -0.2 }]),
    ].map((hs, i) => (i % 2 ? [...hs].reverse() : hs)); // detection order must not matter
    const last = play(frames).at(-1)!;
    expect(last.hands.r!.pos.x).toBeLessThan(last.hands.l!.pos.x);
  });

  it('keeps hands through a short tracking dropout, then lets go', () => {
    const s = initialState();
    interpret(bodyFrame(0, { hands: [hand(MID, SW, GUARD_L), hand(MID, SW, GUARD_R)] }), cal, s);
    expect(interpret(bodyFrame(0.3), cal, s).hands.l).not.toBeNull();
    expect(interpret(bodyFrame(0.6), cal, s).hands.l).toBeNull();
  });

  it('reports nobody present without a head and shoulders', () => {
    const r = interpret({ t: 0, head: null, shoulderL: null, shoulderR: null, hands: [], arms: { l: null, r: null }, face: null }, cal, initialState());
    expect(r.present).toBe(false);
    expect(r.hands.l).toBeNull();
  });

  describe('fist punches (arm extension)', () => {
    beforeEach(() => { TUNING.punchTrigger = 'extend'; });
    const armFrame = (i: number, extR: number, o: { openR?: number; openL?: number; reachR?: { x: number; y: number; z: number }; at?: HandSpec } = {}) => {
      const at = o.at ?? { x: 0.2, y: -0.2 };
      return bodyFrame(i / FPS, {
        hands: [hand(MID, SW, { ...GUARD_L, open: o.openL ?? 0, side: 'l' }), hand(MID, SW, { ...at, open: o.openR ?? 0, side: 'r' })],
        arms: { l: arm(MID, SW, 'l', GUARD_L, 0.25), r: { ...arm(MID, SW, 'r', at, extR), reach: o.reachR ?? null } },
      });
    };
    const runExt = (exts: number[], o: Parameters<typeof armFrame>[2] = {}) => {
      const s = initialState();
      return exts.map((e, i) => interpret(armFrame(i, e, o), cal, s));
    };
    const punchExt = [...repeat(6, () => 0.25), 0.4, 0.55, 0.7, 0.85, 0.95, ...repeat(10, () => 0.95)];

    it('fires when a fist is driven out by a fast-straightening arm, without opening', () => {
      const out = runExt(punchExt);
      const p = punchesIn(out);
      expect(p).toHaveLength(1);
      expect(p[0].hand).toBe('r');
      expect(out.at(-1)!.hands.r!.open).toBe(false);
    });

    it('ignores an arm that straightens slowly', () => {
      expect(punchesIn(runExt([...repeat(6, () => 0.25), ...repeat(60, i => 0.25 + (0.7 * (i + 1)) / 60)]))).toHaveLength(0);
    });

    it('needs the arm pulled back before it can punch again', () => {
      const exts = [...punchExt, 0.8, 0.95, 0.8, 0.95, ...repeat(8, () => 0.25), 0.4, 0.55, 0.7, 0.85, 0.95, ...repeat(6, () => 0.95)];
      expect(punchesIn(runExt(exts))).toHaveLength(2);
    });

    it('does not fire with open hands (shield)', () => {
      expect(punchesIn(runExt(punchExt, { openR: 1, openL: 1 }))).toHaveLength(0);
    });

    it('does not fire with the hand resting low', () => {
      expect(punchesIn(runExt(punchExt, { at: { x: 0.3, y: 1.2 } }))).toHaveLength(0);
    });

    it('passes the 3D reach direction on as aim', () => {
      const p = punchesIn(runExt(punchExt, { reachR: { x: -0.3, y: 0, z: -0.5 } }));
      expect(p[0].dir!.x).toBeCloseTo(-0.6);
      expect(p[0].dir!.y).toBeCloseTo(0);
    });

    it('reports when each arm is ready to punch', () => {
      const out = runExt(punchExt);
      expect(out[3].hands.r!.punchReady).toBe(true);
      expect(out.at(-1)!.hands.r!.punchReady).toBe(false);
    });
  });

  describe('two-hand casts', () => {
    const OPEN = 1;
    /** Fists at the start of `path`, then hands open as they move along it over `frames` frames (k: 0 → 1), then held open. */
    function cast(path: (k: number) => [HandSpec, HandSpec], frames: number): Intent[] {
      const [l0, r0] = path(0);
      return play([
        ...repeat(8, () => [l0, r0]),
        ...repeat(frames, i => path((i + 1) / frames).map(h => ({ ...h, open: OPEN }))),
        ...repeat(10, () => path(1).map(h => ({ ...h, open: OPEN }))),
      ]);
    }
    const kinds = (out: Intent[]) => out.flatMap(o => o.casts.map(c => c.kind));
    const sweepUp = (k: number): [HandSpec, HandSpec] => [{ x: -0.35, y: 0.6 - 1.2 * k }, { x: 0.35, y: 0.6 - 1.2 * k }];
    const spread = (k: number): [HandSpec, HandSpec] => [{ x: -0.1 - 0.9 * k, y: -0.3 }, { x: 0.1 + 0.9 * k, y: -0.3 }];

    it('open hands sweeping up quickly make a fire wall, where the hands are', () => {
      const out = cast(sweepUp, 6);
      expect(kinds(out)).toEqual(['wall']);
      const wall = out.flatMap(o => o.casts)[0];
      expect(Math.abs(wall.at.x)).toBeLessThan(2);
    });

    it('open hands spreading apart quickly make the ultimate', () => {
      expect(kinds(cast(spread, 6))).toEqual(['ultimate']);
    });

    it('slow movements cast nothing', () => {
      expect(kinds(cast(sweepUp, 60))).toEqual([]);
      expect(kinds(cast(spread, 60))).toEqual([]);
    });

    it('casting neither punches nor raises the shield mid-gesture', () => {
      const out = cast(sweepUp, 6);
      expect(punchesIn(out)).toHaveLength(0);
      expect(out.slice(8, 14).some(o => o.shield)).toBe(false);
    });

    it('the shield needs both open hands held still', () => {
      const still = cast(() => [{ x: -0.5, y: -0.2 }, { x: 0.5, y: -0.2 }], 1);
      expect(still.at(-1)!.shield).toBe(true);
    });
  });

  describe('with the body tracked', () => {
    it('uses the arms to tell the hands apart, even when they start crossed', () => {
      const r = interpret(bodyFrame(0, { hands: [hand(MID, SW, { ...GUARD_R, side: 'l' }), hand(MID, SW, { ...GUARD_L, side: 'r' })] }), cal, initialState());
      expect(r.hands.l!.pos.x).toBeGreaterThan(r.hands.r!.pos.x);
    });

    it('ignores a label flip that would make both hands teleport', () => {
      const s = initialState();
      interpret(bodyFrame(0, { hands: [hand(MID, SW, { ...GUARD_L, side: 'l' }), hand(MID, SW, { ...GUARD_R, side: 'r' })] }), cal, s);
      let r!: Intent;
      for (let i = 1; i <= 5; i++) r = interpret(bodyFrame(i / FPS, { hands: [hand(MID, SW, { ...GUARD_R, side: 'l' }), hand(MID, SW, { ...GUARD_L, side: 'r' })] }), cal, s);
      expect(r.hands.l!.pos.x).toBeLessThan(r.hands.r!.pos.x);
    });

    it('keeps following a hand the hand tracker lost, from its arm', () => {
      const s = initialState();
      interpret(bodyFrame(0, { hands: [hand(MID, SW, { ...GUARD_R, side: 'r' })], arms: { r: arm(MID, SW, 'r', GUARD_R) } }), cal, s);
      let r!: Intent;
      for (let i = 1; i <= 30; i++) r = interpret(bodyFrame(i / FPS, { arms: { r: arm(MID, SW, 'r', { x: 0.6, y: -0.2 }) } }), cal, s);
      expect(r.hands.r!.source).toBe('arm');
      expect(r.hands.r!.inView).toBe(true);
      expect(r.hands.r!.pos.x).toBeGreaterThan(view({ x: 0.6, y: 0 }).x); // palm sits just past the wrist
      expect(r.hands.r!.elbow).not.toBeNull();
    });

    it('still knows where a hand is when it leaves the picture', () => {
      const s = initialState();
      let r!: Intent;
      for (let i = 0; i <= 30; i++) r = interpret(bodyFrame(i / FPS, { arms: { r: arm(MID, SW, 'r', { x: 3.5, y: 0, vis: 0.1 }) } }), cal, s);
      expect(r.hands.r).not.toBeNull();
      expect(r.hands.r!.source).toBe('estimate');
      expect(r.hands.r!.inView).toBe(false);
      expect(r.hands.r!.pos.x).toBeGreaterThan(80);
    });

    it('counts a fast-straightening arm as a punch even when the hand barely moves on screen', () => {
    TUNING.punchTrigger = 'open';
      const at = { x: 0.2, y: -0.2 };
      const frame = (i: number, ext: number, open = 0) =>
        bodyFrame(i / FPS, { hands: [hand(MID, SW, { ...GUARD_L, side: 'l' }), hand(MID, SW, { ...at, open, side: 'r' })], arms: { l: arm(MID, SW, 'l', GUARD_L), r: arm(MID, SW, 'r', at, ext) } });
      const s = initialState();
      const out = [
        ...repeat(6, i => frame(i, 0.2)),
        ...repeat(4, i => frame(6 + i, 0.2 + 0.18 * (i + 1))),
        ...repeat(10, i => frame(10 + i, 0.92, 1)),
      ].map(f => interpret(f, cal, s));
      expect(punchesIn(out)).toHaveLength(1);
      expect(out.at(-1)!.hands.r!.extension).toBeGreaterThan(0.8);
    });

    it('passes head turn and shoulder tilt through', () => {
      const f = { ...bodyFrame(0), face: { yaw: 0.3, roll: 0.1 }, shoulderR: { x: 0.6, y: 0.52 } };
      const r = interpret(f, cal, initialState());
      expect(r.face).toEqual({ yaw: 0.3, roll: 0.1 });
      expect(r.bodyTilt).toBeGreaterThan(0);
    });
  });
});
