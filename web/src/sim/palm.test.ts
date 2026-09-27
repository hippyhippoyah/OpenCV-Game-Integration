import { describe, expect, it } from 'vitest';
import { guardState, lerpReach, POSES, punchReach, simulate, type BodyState, type Reach, type SimOptions } from './synthetic';
import { initialState, interpret, type Intent } from '../intent/interpret';
import { dist } from '../math';

function perform(script: (t: number) => BodyState, seconds: number, opts: SimOptions = {}): Intent[] {
  const frames = simulate(script, seconds, opts);
  const f0 = frames[0];
  const cal = { head: f0.head!, sw: dist(f0.shoulderL!, f0.shoulderR!) };
  const s = initialState();
  return frames.map(f => interpret(f, cal, s));
}
const palmsIn = (out: Intent[]) => out.flatMap((o, i) => o.palms.map(p => ({ ...p, t: i / 30 })));
const punchesIn = (out: Intent[]) => out.flatMap(o => o.punches);

/** An open palm held up in front, pushed out toward the camera. */
const PALM: Reach = { out: -0.06, up: 0.14, fwd: 0.27 };
const PALM_OUT: Reach = { out: -0.05, up: 0.14, fwd: 0.52 };
/** An open palm low in front, swept up to above the shoulder. */
const PALM_LOW: Reach = { out: -0.04, up: -0.15, fwd: 0.28 };
const PALM_HIGH: Reach = { out: -0.06, up: 0.22, fwd: 0.32 };

const DISTANCES = [1.2, 1.5, 1.8], SEEDS = [1, 2, 3];
const eachCase = (fn: (distance: number, seed: number) => void) => {
  for (const distance of DISTANCES) for (const seed of SEEDS) fn(distance, seed);
};

/** Right hand opens at openAt (a fist before), then does `move` at the given times. */
function rightPalm(distance: number, openAt: number, move: (t: number) => Reach) {
  return (t: number) => guardState(distance, { r: t < openAt ? {} : { reach: move(t), open: true } });
}

describe('palm moves on a simulated webcam', () => {
  it('an open palm pushed forward sends one pillar, and no fist punch', () => {
    eachCase((distance, seed) => {
      const at = [2, 3];
      const script = rightPalm(distance, 1.2, t => {
        const t0 = at.filter(x => x <= t).at(-1);
        return t0 === undefined ? PALM : punchReach(t, t0, PALM_OUT, 0.15, 0.15, 0.25);
      });
      // punchReach returns to the fist guard; keep the palm pose between pushes instead
      const out = perform(t => { const s = script(t); if (s.hands.r.reach === POSES.guard && t >= 1.2) s.hands.r.reach = PALM; return s; }, 3.8, { seed });
      const p = palmsIn(out);
      expect(p.map(x => `${x.hand}:${x.kind}`), `${distance} m seed ${seed}`).toEqual(['r:push', 'r:push']);
      p.forEach((x, i) => expect(x.t - at[i], `latency ${distance} m seed ${seed}`).toBeLessThan(0.4));
      expect(punchesIn(out), `${distance} m seed ${seed}`).toHaveLength(0);
    });
  });

  it('an open palm swept up quickly raises one pillar', () => {
    eachCase((distance, seed) => {
      const at = [2, 3.2];
      const script = rightPalm(distance, 1.2, t => {
        const t0 = at.filter(x => x <= t).at(-1);
        if (t0 === undefined) return PALM_LOW;
        // up in 0.2 s, hold, then back down slowly
        return t - t0 < 0.6 ? lerpReach(PALM_LOW, PALM_HIGH, (t - t0) / 0.2) : lerpReach(PALM_HIGH, PALM_LOW, (t - t0 - 0.6) / 0.5);
      });
      const out = perform(script, 4.2, { seed });
      const p = palmsIn(out);
      expect(p.map(x => `${x.hand}:${x.kind}`), `${distance} m seed ${seed}`).toEqual(['r:rise', 'r:rise']);
      expect(punchesIn(out), `${distance} m seed ${seed}`).toHaveLength(0);
    });
  });

  it('a fist punch that opens at the end is a punch, not a palm push', () => {
    eachCase((distance, seed) => {
      const out = perform(t => guardState(distance, { r: { reach: punchReach(t, 1.5, POSES.jab), open: t > 1.62 && t < 1.9 } }), 2.6, { seed });
      expect(palmsIn(out), `${distance} m seed ${seed}`).toHaveLength(0);
    });
  });

  describe('does not fire on', () => {
    const quiet = (name: string, script: (distance: number) => (t: number) => BodyState, seconds = 4) =>
      it(name, () => eachCase((distance, seed) => {
        expect(palmsIn(perform(script(distance), seconds, { seed })), `${distance} m seed ${seed}`).toHaveLength(0);
      }));
    quiet('fists in guard and jabs', d => t => guardState(d, { r: { reach: punchReach(t, 1.5, POSES.jab) }, l: { reach: punchReach(t, 2.5, POSES.jab) } }));
    quiet('holding one palm open, still', d => rightPalm(d, 1, () => PALM));
    quiet('opening a hand slowly in guard and moving it around', d => rightPalm(d, 1, t => lerpReach(PALM, { ...PALM, up: 0.2, out: 0 }, (t - 1.5) / 1.5)));
    quiet('raising the shield', d => t => guardState(d, t < 1.5 ? {} : { l: { reach: lerpReach(POSES.guard, POSES.shield, (t - 1.5) / 0.15), open: true }, r: { reach: lerpReach(POSES.guard, POSES.shield, (t - 1.5) / 0.15), open: true } }));
    quiet('both open hands sweeping up (fire wall)', d => t => {
      const reach = t < 1.5 ? PALM_LOW : lerpReach(PALM_LOW, PALM_HIGH, (t - 1.5) / 0.2);
      return guardState(d, { l: { reach, open: t > 1 }, r: { reach, open: t > 1 } });
    });
  });
});
