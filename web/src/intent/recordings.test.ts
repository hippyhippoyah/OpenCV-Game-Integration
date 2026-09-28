import { describe, expect, it } from 'vitest';
import punchesCloseRaw from '../../recordings/punches-close.json?raw';
import punches2Raw from '../../recordings/punches-2.json?raw';
import punches3Raw from '../../recordings/punches-3.json?raw';
import leaningRaw from '../../recordings/leaning.json?raw';
import swayingRaw from '../../recordings/swaying.json?raw';
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
  it('still catches the punches in two punching sessions', () => {
    // alternating jabs from a steady stance: every one
    expect(punchTimes(replay(punchesCloseRaw)).length).toBeGreaterThanOrEqual(15);
    // punching while moving about: all but three slow 2–3 cm drifts (shaped like a sway), and the
    // sharp ones thrown mid-lean (e.g. 14 cm in 0.1 s at 5.8 s) still land
    const p2 = punchTimes(replay(punches2Raw));
    expect(p2.length).toBeGreaterThanOrEqual(17);
    expect(p2.some(x => x.hand === 'l' && Math.abs(x.t - 5.84) < 0.15)).toBe(true);
    // alternating jabs about every half second, recorded by mistake in open-hand mode (P), where
    // only 7 of them fired: the fist detector catches them
    const p3 = punchTimes(replay(punches3Raw));
    expect(p3.length).toBeGreaterThanOrEqual(17);
    expect(new Set(p3.map(x => x.hand))).toEqual(new Set(['l', 'r']));
  });

  it('does not punch while leaning quickly', () => {
    const p = punchTimes(replay(leaningRaw));
    // the three that fired mid-lean before (3.5 s, 5.1 s, 6.75 s: head moving 125–140 units/s)
    for (const at of [3.5, 5.11, 6.75]) expect(p.some(x => Math.abs(x.t - at) < 0.2), `${at} s`).toBe(false);
    expect(p.filter(x => x.headSpeed > 100)).toEqual([]);
  });

  it('never charges a punch by accident (none of these pull a fist back and hold it)', () => {
    for (const raw of [punchesCloseRaw, punches2Raw, punches3Raw, leaningRaw, swayingRaw]) {
      const out = replay(raw);
      expect(out.flatMap(o => o.intent.punches).filter(p => p.charged)).toEqual([]);
      expect(out.filter(o => (o.intent.hands.l?.charge ?? 0) >= 1 || (o.intent.hands.r?.charge ?? 0) >= 1)).toEqual([]);
    }
  });

  it('does not punch while swaying from side to side', () => {
    // no punches at all: the whole clip is swaying (it fired 9 before, at the turnarounds)
    expect(punchTimes(replay(swayingRaw))).toEqual([]);
  });
});
