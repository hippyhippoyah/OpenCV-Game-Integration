import { describe, expect, it } from 'vitest';
import { MOCK_CALIBRATION, MockTracker } from './mock';
import { initialState, interpret, type Intent } from '../intent/interpret';

const identity = { screenToView: (x: number, y: number) => ({ x, y }) };

describe('MockTracker', () => {
  it('produces frames that interpret back to the mouse position', () => {
    const m = new MockTracker(identity);
    m.setMouse(10, 15);
    const r = interpret(m.poll(0), MOCK_CALIBRATION, initialState());
    expect(r.hands!.center.x).toBeCloseTo(10);
    expect(r.hands!.center.y).toBeCloseTo(15);
    expect(r.hands!.spread).toBeCloseTo(6);
  });

  it('turns a click into a push that interpret reads as one throw', () => {
    const m = new MockTracker(identity);
    m.setMouse(0, 15);
    const s = initialState();
    let thrown = 0;
    for (let i = 0; i < 40; i++) {
      if (i === 5) m.push();
      if (interpret(m.poll((i * 1000) / 60), MOCK_CALIBRATION, s).throwNow) thrown++;
    }
    expect(thrown).toBe(1);
  });

  it('leaning with D moves the camera right', () => {
    const m = new MockTracker(identity);
    m.setKey('d', true);
    const s = initialState();
    let last: Intent | null = null;
    for (let i = 0; i < 90; i++) last = interpret(m.poll((i * 1000) / 60), MOCK_CALIBRATION, s);
    expect(last!.head.x).toBeGreaterThan(15);
  });
});
