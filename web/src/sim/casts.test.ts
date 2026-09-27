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

/** Both hands open, moving from `a` to `b` over moveS starting at t0 (fists before openAt). */
function twoHands(d: number, a: Reach, b: Reach, t0: number, moveS: number, openAt = 1.2) {
  return (t: number): BodyState => {
    const reach = lerpReach(a, b, (t - t0) / moveS), open = t >= openAt;
    return guardState(d, { l: { reach, open }, r: { reach, open } });
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

  it('pushing out of a held shield works too', () => {
    eachCase((d, seed) => {
      const out = perform(twoHands(d, SHIELD, SHIELD_OUT, 2.4, 0.18, 1.2), 3.4, { seed });
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

  it('holding the shield still casts nothing', () => {
    eachCase((d, seed) => expect(castsIn(perform(twoHands(d, SHIELD, SHIELD, 2, 1), 4, { seed })), `${d} m seed ${seed}`).toEqual([]));
  });
});
