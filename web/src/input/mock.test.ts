import { describe, expect, it } from 'vitest';
import { MOCK_CALIBRATION, MockTracker } from './mock';
import { initialState, interpret, TUNING, type Intent } from '../intent/interpret';

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

  for (const style of ['extend', 'open'] as const) {
    it(`a click punches with the right hand toward the mouse (${style} style)`, () => {
      TUNING.punchTrigger = style;
      try {
        const m = new MockTracker(identity);
        m.setMouse(5, 5);
        const out = run(m, 80, i => { if (i === 45) m.punch('r'); }); // after the reach reading warms up
        const punches = out.flatMap(o => o.punches);
        expect(punches).toHaveLength(1);
        expect(punches[0].hand).toBe('r');
        expect(Math.abs(punches[0].at.x - 5)).toBeLessThan(6);
        expect(Math.abs(punches[0].at.y - 5)).toBeLessThan(6);
        expect(out.some(o => o.hands.r!.open)).toBe(style === 'open');
      } finally {
        TUNING.punchTrigger = 'extend';
      }
    });
  }

  it('W sweeps open hands up into a fire wall, U spreads them into the ultimate, F pushes a wall', () => {
    for (const [key, kind] of [['w', 'wall'], ['u', 'ultimate'], ['f', 'push']] as const) {
      const m = new MockTracker(identity);
      m.setMouse(0, 5);
      const out = run(m, 60, i => { if (i === 5) m.cast(kind); });
      expect(out.flatMap(o => o.casts.map(c => c.kind)), key).toEqual([kind]);
      expect(out.flatMap(o => o.punches), key).toHaveLength(0);
    }
  });

  it('E pushes an open right palm (pillar)', () => {
    const m = new MockTracker(identity);
    m.setMouse(5, 5);
    const out = run(m, 90, i => { if (i === 45) m.palm('push'); });
    expect(out.flatMap(o => o.palms.map(p => `${p.hand}:${p.kind}`))).toEqual(['r:push']);
    expect(out.flatMap(o => o.punches)).toHaveLength(0);
    expect(out.flatMap(o => o.casts)).toHaveLength(0);
  });

  it('holding G pulls the right fist back to charge it; the next click throws a charged punch', () => {
    const m = new MockTracker(identity);
    m.setMouse(5, 5);
    const out = run(m, 150, i => {
      if (i === 45) m.setKey('g', true);
      if (i === 100) { m.setKey('g', false); m.punch('r'); }
    });
    expect(out[95].hands.r!.charge).toBe(1);
    expect(out.flatMap(o => o.punches).map(p => p.charged)).toEqual([true]);
  });

  it('holding Space opens both hands into a shield without punching', () => {
    const m = new MockTracker(identity);
    m.setMouse(0, 0);
    const out = run(m, 40, i => { if (i === 5) m.setKey(' ', true); });
    expect(out.flatMap(o => o.punches)).toHaveLength(0);
    expect(out.at(-1)!.shield).toBe(true);
  });

  it('holding O takes the right hand out of the picture but keeps it tracked by its arm', () => {
    const m = new MockTracker({ screenToView: (x: number, y: number) => ({ x: x - 640, y: y - 360 }) });
    const last = run(m, 30, i => { if (i === 5) m.setKey('o', true); }).at(-1)!;
    expect(last.hands.r!.inView).toBe(false);
    expect(last.hands.r!.source).toBe('estimate');
    expect(last.hands.l!.inView).toBe(true);
  });

  it('fist punches go through the same reach-from-size detection as the camera', () => {
    const m = new MockTracker(identity);
    m.setMouse(5, 5);
    const out = run(m, 80, i => { if (i === 45) m.punch('r'); }); // after the reach reading warms up
    expect(out[3].hands.r!.reach).not.toBeNull();
    expect(out.flatMap(o => o.punches)).toHaveLength(1);
  });

  it('holding X crosses the arms into an X block', () => {
    const m = new MockTracker(identity);
    const out = run(m, 20, i => { if (i === 3) m.setKey('x', true); });
    expect(out.at(-1)!.xBlock).toBe(true);
    expect(out.flatMap(o => o.punches)).toHaveLength(0);
  });

  it('leaning with D moves the camera right', () => {
    const m = new MockTracker(identity);
    m.setKey('d', true);
    expect(run(m, 90).at(-1)!.head.x).toBeGreaterThan(15);
  });
});
