import { describe, expect, it } from 'vitest';
import { Calibrator, setupChecks } from './calibration';
import { bodyFrame, hand } from '../test/frames';

const MID = { x: 0.5, y: 0.5 }, SW = 0.2;
/** Both hands brought to the chest: the start pose. */
const atChest = () => [hand(MID, SW, { x: -0.3, y: 0.4 }), hand(MID, SW, { x: 0.3, y: 0.4 })];
/** Standing ready, holding the start pose. */
const ready = (t: number, head?: { x: number; y: number }) => bodyFrame(t, { hands: atChest(), ...(head ? { head } : {}) });

describe('setupChecks', () => {
  it('both hands at the chest is the start pose', () => {
    expect(setupChecks(ready(0))).toEqual({ seen: true, handsAtChest: true, distance: 'unknown' });
  });

  it('hands down by the sides, raised overhead, or only one at the chest are not', () => {
    const down = [hand(MID, SW, { x: -0.6, y: 2.2 }), hand(MID, SW, { x: 0.6, y: 2.2 })];
    const up = [hand(MID, SW, { x: -0.3, y: -1.2 }), hand(MID, SW, { x: 0.3, y: -1.2 })];
    expect(setupChecks(bodyFrame(0, { hands: down })).handsAtChest).toBe(false);
    expect(setupChecks(bodyFrame(0, { hands: up })).handsAtChest).toBe(false);
    expect(setupChecks(bodyFrame(0, { hands: [atChest()[0]] })).handsAtChest).toBe(false);
  });

  it('reads the distance from the shoulders: too close, too far or fine', () => {
    const at = (m: number) => setupChecks({ ...ready(0), body: { span3: 0.38, span2: (1.05 * 0.38) / m } }).distance;
    expect(at(0.6)).toBe('close');
    expect(at(1.5)).toBe('ok');
    expect(at(3)).toBe('far');
  });

  it('nobody there: nothing is seen', () => {
    expect(setupChecks({ t: 0, head: null, shoulderL: null, shoulderR: null, hands: [], arms: { l: null, r: null }, face: null }).seen).toBe(false);
  });
});

describe('Calibrator', () => {
  it('needs 1.5 s of the start pose held still, then returns the average pose', () => {
    const c = new Calibrator();
    for (let t = 0; t < 1.4; t += 0.1) c.add(ready(t));
    expect(c.result()).toBeNull();
    c.add(ready(1.5));
    const r = c.result()!;
    expect(r.sw).toBeCloseTo(0.2);
    expect(r.head.x).toBeCloseTo(0.5);
    expect(r.head.y).toBeCloseTo(0.35);
  });

  it("doesn't start until both hands are at the chest", () => {
    const c = new Calibrator();
    for (let t = 0; t <= 2; t += 0.1) c.add(bodyFrame(t));
    expect(c.progress()).toBe(0);
  });

  it('a brief slip only pauses it; letting go for longer starts over', () => {
    const c = new Calibrator();
    for (let t = 0; t <= 0.8; t += 0.1) c.add(ready(t));
    const before = c.progress();
    c.add(bodyFrame(0.9)); // hands dropped for one frame
    expect(c.progress()).toBe(before);
    for (let t = 1.0; t <= 1.5; t += 0.1) c.add(bodyFrame(t)); // …and kept down
    expect(c.progress()).toBe(0);
  });

  it('restarts when the head moves too much', () => {
    const c = new Calibrator();
    for (let t = 0; t <= 1.0; t += 0.1) c.add(ready(t));
    c.add(ready(1.1, { x: 0.6, y: 0.35 }));
    expect(c.progress()).toBe(0);
  });

  it('restarts when the body is lost for more than a moment', () => {
    const c = new Calibrator();
    for (let t = 0; t <= 1.0; t += 0.1) c.add(ready(t));
    for (let t = 1.1; t <= 1.6; t += 0.1) c.add({ t, head: null, shoulderL: null, shoulderR: null, hands: [], arms: { l: null, r: null }, face: null });
    expect(c.progress()).toBe(0);
  });
});
