import { describe, expect, it } from 'vitest';
import punchesCloseRaw from '../../recordings/punches-close.json?raw';
import punches2Raw from '../../recordings/punches-2.json?raw';
import leaningRaw from '../../recordings/leaning.json?raw';
import { initialState, interpret, TUNING, type Intent } from './interpret';
import type { TrackingFrame } from '../input/types';
import type { Calibration } from './calibration';

/** A real-camera recording (K in game), trimmed to its tracking frames and what fired at the time. */
interface Fixture {
  calibration: Calibration;
  tuning: { punchSensitivity: number };
  samples: { frame: TrackingFrame; recorded: { punches: string[] } }[];
}

/** Replay a recording through the current detector, at the sensitivity it was played with. */
function replay(raw: string) {
  const rec = JSON.parse(raw) as Fixture;
  const saved = TUNING.punchSensitivity;
  TUNING.punchSensitivity = rec.tuning.punchSensitivity;
  try {
    const s = initialState(), t0 = rec.samples[0].frame.t;
    const out: { t: number; intent: Intent; headSpeed: number }[] = [];
    for (const smp of rec.samples) {
      const intent = interpret(smp.frame, rec.calibration, s);
      out.push({ t: smp.frame.t - t0, intent, headSpeed: s.headSpeed });
    }
    return out;
  } finally {
    TUNING.punchSensitivity = saved;
  }
}
const punchTimes = (out: ReturnType<typeof replay>) => out.flatMap(o => o.intent.punches.map(p => ({ t: o.t, hand: p.hand, headSpeed: o.headSpeed })));

describe('real-camera recordings', () => {
  it('still catches every punch in two punching sessions', () => {
    // counted by hand from the traces: alternating jabs, all clearly toward the camera
    expect(punchTimes(replay(punchesCloseRaw)).length).toBeGreaterThanOrEqual(15);
    expect(punchTimes(replay(punches2Raw)).length).toBeGreaterThanOrEqual(19);
  });

  it('does not punch while leaning quickly', () => {
    const p = punchTimes(replay(leaningRaw));
    // the three that fired mid-lean before (3.5 s, 5.1 s, 6.75 s: head moving 125–140 units/s)
    for (const at of [3.5, 5.11, 6.75]) expect(p.some(x => Math.abs(x.t - at) < 0.2), `${at} s`).toBe(false);
    expect(p.filter(x => x.headSpeed > 100)).toEqual([]);
  });
});
