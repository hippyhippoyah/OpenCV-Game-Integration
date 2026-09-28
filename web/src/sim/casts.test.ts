import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { guardState, lerpReach, simulate, type BodyState, type Reach, type SimOptions } from './synthetic';
import { initialState, interpret, TUNING, type Intent } from '../intent/interpret';
import { dist } from '../math';

// Detection is tuned and tested at sensitivity 1; the game's default is more sensitive (see TUNING).
let sensitivity = 1;
beforeAll(() => { sensitivity = TUNING.punchSensitivity; TUNING.punchSensitivity = 1; });
afterAll(() => { TUNING.punchSensitivity = sensitivity; });

function perform(script: (t: number) => BodyState, seconds: number, opts: SimOptions = {}): Intent[] {
  const frames = simulate(script, seconds, opts);
  const cal = { head: frames[0].head!, sw: dist(frames[0].shoulderL!, frames[0].shoulderR!) };
  const s = initialState();
  return frames.map(f => interpret(f, cal, s));
}
const castsIn = (out: Intent[]) => out.flatMap(o => o.casts.map(c => c.kind));
const palmsIn = (out: Intent[]) => out.flatMap(o => o.palms);

/**
 * Both hands open, moving from `a` to `b` over moveS starting at t0 (fists before openAt), palms
 * turned in by `turn` (0 = toward the camera), or turning from turn[0] to turn[1] as they move.
 */
function twoHands(d: number, a: Reach, b: Reach, t0: number, moveS: number, openAt = 1.2, turn: number | [number, number] = 0) {
  return (t: number): BodyState => {
    const k = (t - t0) / moveS, reach = lerpReach(a, b, k), open = t >= openAt;
    const tn = typeof turn === 'number' ? turn : turn[0] + (turn[1] - turn[0]) * Math.min(1, Math.max(0, k));
    return guardState(d, { l: { reach, open, turn: tn }, r: { reach, open, turn: tn } });
  };
}

/** Palms up at shoulder width (the shield), and pushed out toward the camera. */
const SHIELD: Reach = { out: 0.08, up: 0.12, fwd: 0.28 };
const SHIELD_OUT: Reach = { out: 0.08, up: 0.14, fwd: 0.52 };
/** Palms gathered in front of the chest, then flung wide. */
const GATHERED: Reach = { out: -0.15, up: 0.1, fwd: 0.3 };
const FLUNG: Reach = { out: 0.3, up: 0.12, fwd: 0.3 };
const LOW: Reach = { out: 0.02, up: -0.2, fwd: 0.28 };
const HIGH: Reach = { out: 0.02, up: 0.25, fwd: 0.3 };

const DISTANCES = [1.2, 1.5, 1.8], SEEDS = [1, 2, 3];
const eachCase = (fn: (d: number, seed: number) => void) => { for (const d of DISTANCES) for (const seed of SEEDS) fn(d, seed); };

describe('two-hand casts on a simulated webcam', () => {
  it('both palms pushed forward roll a fire wall, not the ultimate', () => {
    eachCase((d, seed) => {
      const out = perform(twoHands(d, SHIELD, SHIELD_OUT, 2, 0.18), 3, { seed });
      expect(castsIn(out), `${d} m seed ${seed}`).toEqual(['push']);
      expect(palmsIn(out), `${d} m seed ${seed}`).toHaveLength(0);
    });
  });

  it('pushing out of a held shield works too (turning the palms forward as they go)', () => {
    eachCase((d, seed) => {
      const out = perform(twoHands(d, SHIELD, SHIELD_OUT, 2.4, 0.18, 1.2, [1, 0]), 3.4, { seed });
      expect(out[Math.round(2.3 * 30)].shield, `${d} m seed ${seed}`).toBe(true);
      expect(castsIn(out), `${d} m seed ${seed}`).toEqual(['push']);
    });
  });

  it('palms gathered together and flung apart fire the ultimate', () => {
    eachCase((d, seed) => expect(castsIn(perform(twoHands(d, GATHERED, FLUNG, 2, 0.2), 3, { seed })), `${d} m seed ${seed}`).toEqual(['ultimate']));
  });

  it('palms swept up raise a fire wall', () => {
    eachCase((d, seed) => expect(castsIn(perform(twoHands(d, LOW, HIGH, 2, 0.2), 3, { seed })), `${d} m seed ${seed}`).toEqual(['wall']));
  });

  // at 1.8 m a still open palm's reading wobbles ±15 cm, too much to be sure it isn't pushing too
  it('with both palms open, pushing just one sends a single pillar (not a wall push), up to 1.5 m', () => {
    for (const d of [1.2, 1.5]) for (const seed of SEEDS) {
      const out = perform(t => guardState(d, {
        l: { reach: SHIELD, open: t >= 1.2 },
        r: { reach: lerpReach(SHIELD, SHIELD_OUT, (t - 2) / 0.18), open: t >= 1.2 },
      }), 3, { seed });
      expect(palmsIn(out).map(p => `${p.hand}:${p.kind}`), `${d} m seed ${seed}`).toEqual(['r:push']);
      expect(castsIn(out), `${d} m seed ${seed}`).toEqual([]);
    }
  });

  it('holding the shield still casts nothing', () => {
    eachCase((d, seed) => expect(castsIn(perform(twoHands(d, SHIELD, SHIELD, 2, 1, 1.2, 1), 4, { seed })), `${d} m seed ${seed}`).toEqual([]));
  });

  it('the shield needs the palms facing each other: open palms facing the camera are not a shield', () => {
    eachCase((d, seed) => {
      expect(perform(twoHands(d, SHIELD, SHIELD, 2, 1, 1.2, 1), 3, { seed }).slice(-15).every(o => o.shield), `${d} m seed ${seed}`).toBe(true);
      expect(perform(twoHands(d, SHIELD, SHIELD, 2, 1, 1.2, 0), 3, { seed }).some(o => o.shield), `${d} m seed ${seed}`).toBe(false);
    });
  });

  it('turning the palms away from each other drops the shield', () => {
    eachCase((d, seed) => {
      const out = perform(twoHands(d, SHIELD, SHIELD, 2.5, 0.3, 1.2, [1, 0]), 3.4, { seed });
      expect(out[Math.round(2.4 * 30)].shield, `${d} m seed ${seed}`).toBe(true);
      expect(out.slice(-10).some(o => o.shield), `${d} m seed ${seed}`).toBe(false);
    });
  });

  it('both palms shoved forward while turned in (facing each other) is no wall push', () => {
    // (coming toward the camera, the hands also climb in the picture: that may read as a wall)
    eachCase((d, seed) => expect(castsIn(perform(twoHands(d, SHIELD, SHIELD_OUT, 2, 0.18, 1.2, 1), 3, { seed })), `${d} m seed ${seed}`).not.toContain('push'));
  });
});
