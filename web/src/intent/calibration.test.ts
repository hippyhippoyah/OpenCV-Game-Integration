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
    c.add({ t: 1.1, head: null, shoulderL: null, shoulderR: null, hands: [], arms: { l: null, r: null }, face: null });
    expect(c.progress()).toBe(0);
  });
});
