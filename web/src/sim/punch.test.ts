import { describe, expect, it } from 'vitest';
import { guardState, lerpReach, POSES, punchReach, simulate, type BodyState, type Reach, type SimOptions } from './synthetic';
import { initialState, interpret, TUNING, type Intent } from '../intent/interpret';
import type { Side } from '../input/types';
import { dist } from '../math';

/** Run a performance through the synthetic camera and the interpreter. */
function perform(script: (t: number) => BodyState, seconds: number, opts: SimOptions = {}): Intent[] {
  const frames = simulate(script, seconds, opts);
  const f0 = frames[0];
  const cal = { head: f0.head!, sw: dist(f0.shoulderL!, f0.shoulderR!) };
  const s = initialState();
  return frames.map(f => interpret(f, cal, s));
}
const punchesIn = (out: Intent[]) => out.flatMap((o, i) => o.punches.map(p => ({ ...p, t: i / 30 })));

/** Punches from guard at the given times: [time, side, target]. */
function combo(distance: number, moves: [number, Side, Reach][]) {
  return (t: number): BodyState => {
    const reachFor = (side: Side) => {
      const mine = moves.filter(([t0, s]) => s === side && t0 <= t).at(-1);
      return mine ? punchReach(t, mine[0], mine[2]) : POSES.guard;
    };
    return guardState(distance, { l: { reach: reachFor('l') }, r: { reach: reachFor('r') } });
  };
}

/**
 * Fist punches are read from how big each fist looks, which a webcam does cleanly up to about
 * 1.8 m: there every punch must be caught and nothing may misfire. Further away the reading gets
 * shaky; at 2.5 m most punches must still land and misfires must stay rare.
 */
const PUNCH_RANGE = [1.2, 1.5, 1.8];
const SEEDS = [1, 2, 3];
const eachCase = (fn: (distance: number, seed: number) => void, distances = PUNCH_RANGE) => {
  for (const distance of distances) for (const seed of SEEDS) fn(distance, seed);
};

describe('fist punches on a simulated webcam', () => {
  it('catches every jab, once, from the right hand and quickly', () => {
    eachCase((distance, seed) => {
      const at = [1.5, 2.5, 3.5];
      const p = punchesIn(perform(combo(distance, at.map(t => [t, 'r', POSES.jab])), 4.5, { seed }));
      expect(p.map(x => x.hand), `${distance} m seed ${seed}`).toEqual(['r', 'r', 'r']);
      // fires within 0.32 s of starting the punch (the fist is out by 0.12 s; nearer 1.8 m it waits closer to full extension)
      p.forEach((x, i) => expect(x.t - at[i], `latency ${distance} m seed ${seed}`).toBeLessThan(0.32));
    });
  });

  it('catches left jabs and left-right combos', () => {
    eachCase((distance, seed) => {
      const moves: [number, Side, Reach][] = [[1.5, 'l', POSES.jab], [2.1, 'r', POSES.jab], [2.7, 'l', POSES.jab], [3.3, 'r', POSES.cross]];
      const p = punchesIn(perform(combo(distance, moves), 4.2, { seed }));
      expect(p.map(x => x.hand), `${distance} m seed ${seed}`).toEqual(['l', 'r', 'l', 'r']);
    });
  });

  it('aims a jab roughly straight and a cross across the body', () => {
    eachCase((distance, seed) => {
      const [jab] = punchesIn(perform(combo(distance, [[1.5, 'r', POSES.jab]]), 2.5, { seed }));
      const [cross] = punchesIn(perform(combo(distance, [[1.5, 'r', POSES.cross]]), 2.5, { seed }));
      expect(Math.abs(jab.dir!.x), `jab ${distance} m seed ${seed}`).toBeLessThan(0.35);
      expect(cross.dir!.x, `cross ${distance} m seed ${seed}`).toBeLessThan(-0.3);
    });
  });

  it('still lands most punches at 2.5 m', () => {
    let hit = 0, total = 0;
    for (const seed of SEEDS) {
      const at = [1.5, 2.5, 3.5];
      hit += punchesIn(perform(combo(2.5, at.map(t => [t, 'r', POSES.jab])), 4.5, { seed })).length;
      total += at.length;
    }
    expect(hit / total).toBeGreaterThanOrEqual(0.75);
  });

  it('still fires when the hand tracker loses the blurred fist', () => {
    for (const distance of PUNCH_RANGE) {
      const p = punchesIn(perform(combo(distance, [[1.5, 'r', POSES.jab], [2.5, 'r', POSES.jab]]), 3.5, { blurDropChance: 1 }));
      expect(p.map(x => x.hand), `${distance} m`).toEqual(['r', 'r']);
    }
  });

  describe('does not fire on', () => {
    const quiet = (name: string, script: (distance: number) => (t: number) => BodyState, seconds = 5) =>
      it(name, () => {
        eachCase((distance, seed) => {
          expect(punchesIn(perform(script(distance), seconds, { seed })), `${distance} m seed ${seed}`).toHaveLength(0);
        });
        const far = SEEDS.reduce((n, seed) => n + punchesIn(perform(script(2.5), seconds, { seed })).length, 0);
        expect(far, `2.5 m: at most one stray punch over ${SEEDS.length} runs`).toBeLessThanOrEqual(1);
      });

    quiet('standing in guard', d => () => guardState(d));
    quiet('weaving in guard', d => t => {
      const w = (k: number) => ({ out: POSES.guard.out + 0.06 * Math.sin(t * 9 + k), up: POSES.guard.up + 0.05 * Math.sin(t * 7 + k), fwd: POSES.guard.fwd + 0.04 * Math.sin(t * 8 + k) });
      return guardState(d, { l: { reach: w(0) }, r: { reach: w(2) } });
    });
    quiet('leaning in and back', d => t => ({ ...guardState(d), distance: d - 0.35 * Math.max(0, Math.sin(Math.max(0, t - 1) * 2)) }));
    // a deliberate reach counts (punches are mostly about distance), but guard drifting forward doesn't
    quiet('the guard slowly drifting forward', d => t => guardState(d, { r: { reach: lerpReach(POSES.guard, { ...POSES.guard, fwd: POSES.guard.fwd + 0.1 }, (t - 1) / 3) } }));
    quiet('pushing both hands forward together', d => t => {
      const reach = lerpReach(POSES.guard, POSES.jab, (t - 1.5) / 0.12);
      return guardState(d, { l: { reach }, r: { reach } });
    }, 3);
    quiet('raising the shield', d => t => guardState(d, t < 1.5 ? {} : { l: { reach: POSES.shield, open: true }, r: { reach: POSES.shield, open: true } }), 3);
    quiet('crossing the arms (X block)', d => t => {
      const reach = lerpReach(POSES.guard, POSES.xblock, (t - 1.5) / 0.2);
      return guardState(d, { l: { reach }, r: { reach } });
    }, 3);
  });


  it('says when you are too far away for fist punches', () => {
    const noise = (distance: number) => perform(() => guardState(distance), 4).at(-1)!.hands.r!.reachNoise!;
    expect(noise(1.5)).toBeLessThan(TUNING.reachNoiseMax);
    expect(noise(2.5)).toBeGreaterThan(TUNING.reachNoiseMax);
  });
});

describe('X block on a simulated webcam', () => {
  it('holds while the forearms are crossed, and not during a cross punch', () => {
    eachCase((distance, seed) => {
      const out = perform(t => {
        const reach = lerpReach(POSES.guard, POSES.xblock, (t - 1) / 0.2);
        return guardState(distance, t < 2.5 ? { l: { reach }, r: { reach } } : {});
      }, 3.5, { seed });
      expect(out[Math.round(2.2 * 30)].xBlock, `crossed ${distance} m seed ${seed}`).toBe(true);
      expect(out.at(-1)!.xBlock, `released ${distance} m seed ${seed}`).toBe(false);
      const cross = perform(combo(distance, [[1.5, 'r', POSES.cross]]), 2.5, { seed });
      expect(cross.some(o => o.xBlock), `cross punch ${distance} m seed ${seed}`).toBe(false);
    });
  });
});
