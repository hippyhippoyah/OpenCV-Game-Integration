import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { guardState, lerpReach, POSES, simulate, type BodyState, type Reach, type SimOptions } from './synthetic';
import { initialState, interpret, TUNING, type Intent } from '../intent/interpret';
import { dist } from '../math';

// Detection is tuned and tested at sensitivity 1; the game's default is more sensitive (see TUNING).
let sensitivity = 1;
beforeAll(() => { sensitivity = TUNING.punchSensitivity; TUNING.punchSensitivity = 1; });
afterAll(() => { TUNING.punchSensitivity = sensitivity; });

function perform(script: (t: number) => BodyState, seconds: number, opts: SimOptions = {}): Intent[] {
  const frames = simulate(script, seconds, opts);
  const f0 = frames[0];
  const cal = { head: f0.head!, sw: dist(f0.shoulderL!, f0.shoulderR!) };
  const s = initialState();
  return frames.map(f => interpret(f, cal, s));
}
const palmsIn = (out: Intent[]) => out.flatMap((o, i) => o.palms.map(p => ({ ...p, t: i / 30 })));
const punchesIn = (out: Intent[]) => out.flatMap(o => o.punches);

/** An open palm held up in front. */
const PALM: Reach = { out: -0.06, up: 0.14, fwd: 0.27 };
const pushed = (by: number): Reach => ({ out: -0.05, up: 0.14, fwd: PALM.fwd + by });

/** A push from `from`: out over outS, hold, back over 0.3 s. */
function pushReach(t: number, t0: number, from: Reach, to: Reach, outS: number): Reach {
  if (t < t0) return from;
  if (t < t0 + outS) return lerpReach(from, to, (t - t0) / outS);
  if (t < t0 + outS + 0.15) return to;
  return lerpReach(to, from, (t - t0 - outS - 0.15) / 0.3);
}

const DISTANCES = [1.2, 1.5, 1.8], SEEDS = [1, 2, 3];
const eachCase = (fn: (distance: number, seed: number) => void) => {
  for (const distance of DISTANCES) for (const seed of SEEDS) fn(distance, seed);
};

/** Right palm pushes at each time in `at`; open from openAt on (a fist before, pushing from guard). */
function pushes(distance: number, at: number[], by: number, outS: number, openAt = 1.2) {
  return (t: number) => {
    const t0 = at.filter(x => x <= t + 0.5).find(x => t < x + outS + 0.45) ?? at.filter(x => x <= t).at(-1);
    const open = t >= openAt;
    const rest = open ? PALM : POSES.guard;
    return guardState(distance, { r: { reach: t0 === undefined ? rest : pushReach(t, t0, rest, pushed(by), outS), open } });
  };
}

describe('palm push on a simulated webcam', () => {
  const expectPushes = (out: Intent[], n: number, label: string) => {
    expect(palmsIn(out).map(x => `${x.hand}:${x.kind}`), label).toEqual(Array(n).fill('r:push'));
    expect(punchesIn(out), label).toHaveLength(0);
  };

  it('a full shove sends one pillar each time, and no fist punch', () => {
    eachCase((d, seed) => {
      const at = [2, 3];
      const out = perform(pushes(d, at, 0.25, 0.15), 3.8, { seed });
      expectPushes(out, 2, `${d} m seed ${seed}`);
      palmsIn(out).forEach((x, i) => expect(x.t - at[i], `latency ${d} m seed ${seed}`).toBeLessThan(0.4));
    });
  });

  it('a 20 cm push counts up to 1.5 m; further back it takes a longer one (25 cm)', () => {
    eachCase((d, seed) => expectPushes(perform(pushes(d, [2, 3], d > 1.5 ? 0.25 : 0.2, 0.12), 3.8, { seed }), 2, `${d} m seed ${seed}`));
  });

  it('a palm turned sideways (edge-on to the camera) shoved forward is not a push', () => {
    eachCase((d, seed) => {
      const out = perform(t => {
        const r = pushes(d, [2, 3], 0.25, 0.15)(t).hands.r;
        return guardState(d, { r: { ...r, turn: 1 } });
      }, 3.8, { seed });
      expect(palmsIn(out), `${d} m seed ${seed}`).toHaveLength(0);
    });
  });

  it('a small nudge of the palm (8 cm) is not a push', () => {
    eachCase((d, seed) => expectPushes(perform(pushes(d, [2, 3], 0.08, 0.12), 3.8, { seed }), 0, `${d} m seed ${seed}`));
  });

  it('a slower push (0.3 s) counts', () => {
    eachCase((d, seed) => expectPushes(perform(pushes(d, [2, 3], 0.27, 0.3), 3.8, { seed }), 2, `${d} m seed ${seed}`));
  });

  // opening mid-push loses a little of the push to the change of measurement; at 1.8 m, where the
  // reading is noisier, that makes it hit or miss (about 2 in 3)
  it('opening the hand while pushing counts, up to 1.5 m', () => {
    for (const d of [1.2, 1.5]) for (const seed of SEEDS) {
      // fist in guard, starts pushing at 2.0 and opens 0.06 s in
      const out = perform(t => guardState(d, { r: { reach: pushReach(t, 2, POSES.guard, pushed(0.22), 0.18), open: t >= 2.06 && t < 2.6 } }), 3, { seed });
      expect([...palmsIn(out).map(x => x.kind as string), ...punchesIn(out).map(() => 'punch')], `${d} m seed ${seed}`).toEqual(['push']);
    }
  });

  it('still counts when motion blur makes the hand tracker drop the palm', () => {
    for (const d of DISTANCES) expectPushes(perform(pushes(d, [2, 3], 0.25, 0.15), 3.8, { blurDropChance: 1, blurSpeed: 0.6 }), 2, `${d} m`);
  });

  it('a fist punch is one attack: a punch, or a push if it opens right away', () => {
    eachCase((d, seed) => {
      const out = perform(t => guardState(d, { r: { reach: pushReach(t, 1.5, POSES.guard, POSES.jab, 0.12), open: t > 1.75 && t < 1.95 } }), 2.6, { seed });
      expect(palmsIn(out).length + punchesIn(out).length, `${d} m seed ${seed}`).toBe(1);
    });
  });

  describe('does not fire on', () => {
    // like fist punches: none up close, at most a stray one over the seeds further back
    const quiet = (name: string, script: (distance: number) => (t: number) => BodyState, seconds = 5) =>
      it(name, () => {
        for (const seed of SEEDS) expect(palmsIn(perform(script(1.2), seconds, { seed })), `1.2 m seed ${seed}`).toHaveLength(0);
        for (const d of [1.5, 1.8]) {
          const strays = SEEDS.reduce((n, seed) => n + palmsIn(perform(script(d), seconds, { seed })).length, 0);
          expect(strays, `${d} m`).toBeLessThanOrEqual(1);
        }
      });
    quiet('fists in guard and jabs', d => t => guardState(d, { r: { reach: pushReach(t, 1.5, POSES.guard, POSES.jab, 0.12) }, l: { reach: pushReach(t, 2.5, POSES.guard, POSES.jab, 0.12) } }));
    quiet('holding one palm open, still', d => t => guardState(d, { r: { reach: PALM, open: t > 1 } }));
    quiet('moving an open palm around slowly', d => t => guardState(d, { r: { reach: lerpReach(PALM, { ...PALM, up: 0.22, out: 0, fwd: 0.32 }, (t - 1.5) / 1.5), open: t > 1 } }));
    quiet('raising the shield', d => t => guardState(d, t < 1.5 ? {} : { l: { reach: lerpReach(POSES.guard, POSES.shield, (t - 1.5) / 0.15), open: true }, r: { reach: lerpReach(POSES.guard, POSES.shield, (t - 1.5) / 0.15), open: true } }));
    quiet('both open hands pushed forward together', d => t => {
      const reach = pushReach(t, 2, PALM, pushed(0.25), 0.15);
      return guardState(d, { l: { reach, open: t > 1 }, r: { reach, open: t > 1 } });
    });
  });
});
