import { describe, expect, it } from 'vitest';
import { MOCK_CALIBRATION, MockTracker } from './mock';
import { initialState, interpret, type Intent } from '../intent/interpret';

const identity = { screenToView: (x: number, y: number) => ({ x, y }) };

function run(m: MockTracker, frames: number, each?: (i: number) => void): Intent[] {
  const s = initialState(), out: Intent[] = [];
  for (let i = 0; i < frames; i++) {
    each?.(i);
    out.push(interpret(m.poll((i * 1000) / 60), MOCK_CALIBRATION, s));
  }
  return out;
}

describe('MockTracker', () => {
  it('rests with both fists up in guard', () => {
    const last = run(new MockTracker(identity), 10).at(-1)!;
    expect(last.hands.l!.open).toBe(false);
    expect(last.hands.r!.open).toBe(false);
    expect(last.hands.l!.pos.x).toBeLessThan(last.hands.r!.pos.x);
    expect(last.punches).toHaveLength(0);
  });

  it('a click punches with the right hand and opens it at the mouse', () => {
    const m = new MockTracker(identity);
    m.setMouse(5, 5);
    const punches = run(m, 40, i => { if (i === 5) m.punch('r'); }).flatMap(o => o.punches);
    expect(punches).toHaveLength(1);
    expect(punches[0].hand).toBe('r');
    expect(Math.abs(punches[0].at.x - 5)).toBeLessThan(2);
    expect(Math.abs(punches[0].at.y - 5)).toBeLessThan(2);
  });

  it('holding Space opens both hands into a shield without punching', () => {
    const m = new MockTracker(identity);
    m.setMouse(0, 0);
    const out = run(m, 40, i => { if (i === 5) m.setKey(' ', true); });
    expect(out.flatMap(o => o.punches)).toHaveLength(0);
    expect(out.at(-1)!.shield).toBe(true);
  });

  it('leaning with D moves the camera right', () => {
    const m = new MockTracker(identity);
    m.setKey('d', true);
    expect(run(m, 90).at(-1)!.head.x).toBeGreaterThan(15);
  });
});
