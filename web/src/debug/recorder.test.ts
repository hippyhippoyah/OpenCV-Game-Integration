import { describe, expect, it } from 'vitest';
import { Recorder, type Recording } from './recorder';
import { initialState, interpret } from '../intent/interpret';
import { bodyFrame } from '../test/frames';

const cal = { head: { x: 0.5, y: 0.35 }, sw: 0.2 };

describe('Recorder', () => {
  it('collects frames, raw landmarks and intents for a fixed time, then hands over the recording once', () => {
    const done: Recording[] = [];
    const rec = new Recorder(r => done.push(r));
    expect(rec.active).toBe(false);
    rec.start(1, cal);
    const s = initialState();
    for (let i = 0; i <= 40; i++) {
      const f = bodyFrame(10 + i / 30);
      rec.push(f, { hands: [], handsWorld: [], pose: [], poseWorld: [] }, interpret(f, cal, s));
    }
    expect(done).toHaveLength(1);
    expect(rec.active).toBe(false);
    const r = done[0];
    expect(r.calibration).toEqual(cal);
    expect(r.samples.length).toBeGreaterThanOrEqual(30);
    expect(r.samples.length).toBeLessThanOrEqual(32);
    expect(r.samples[0].raw).not.toBeNull();
    expect(r.samples[0].intent.present).toBe(true);
    expect(r.tuning.punchTrigger).toBe('extend');
    expect(() => JSON.stringify(r)).not.toThrow();
  });

  it('ignores frames while not recording', () => {
    const done: Recording[] = [];
    const rec = new Recorder(r => done.push(r));
    const f = bodyFrame(0);
    rec.push(f, null, interpret(f, cal, initialState()));
    expect(done).toHaveLength(0);
  });
});
