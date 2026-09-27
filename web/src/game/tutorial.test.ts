import { describe, expect, it } from 'vitest';
import { Game, TUNE, type GameEvent } from './game';
import { LESSON_PAUSE_S, LESSONS, Tutorial } from './tutorial';
import type { HandState, Intent } from '../intent/interpret';
import { mulberry32 } from '../math';

const hs = (x: number, y: number, open = false): HandState =>
  ({ pos: { x, y }, vel: { x: 0, y: 0 }, openness: open ? 1 : 0, open, facing: 1, source: 'hand', inView: true, elbow: null, extension: null, punchReady: true, punchRise: null, reach: null, reachBase: null, reachNoise: null, aimDir: null, charge: 0 });
const intent = (o: Partial<Intent> = {}): Intent =>
  ({ present: true, head: { x: 0, y: 0 }, hands: { l: hs(-12, 22), r: hs(12, 22) }, shoulders: { l: { x: -20, y: 20 }, r: { x: 20, y: 20 } }, punches: [], palms: [], shield: false, xBlock: false, casts: [], face: null, bodyTilt: 0, ...o });

/** Step the game and tutorial together; `each` gives the intent for each frame. */
function play(g: Game, t: Tutorial, seconds: number, each: (i: number) => Intent = () => intent()): GameEvent[] {
  const all: GameEvent[] = [];
  for (let i = 0; i < seconds * 60; i++) {
    g.step(1 / 60, each(i));
    const ev = g.drainEvents();
    all.push(...ev);
    t.update(1 / 60, ev);
  }
  return all;
}
const setup = (start = 0) => { const g = new Game(mulberry32(1)); return { g, t: new Tutorial(g, start) }; };
const lessonIndex = (id: string) => LESSONS.findIndex(l => l.id === id);

describe('Tutorial', () => {
  it('starts with movement, then punching, and teaches every move', () => {
    expect(LESSONS.map(l => l.id)).toEqual([
      'move', 'punch', 'charge', 'flurry', 'pillar', 'wave', 'shield', 'counter', 'xblock',
      'palm', 'onetwo', 'volley', 'wall', 'wallbreaker', 'ultimate',
    ]);
  });

  it('takes over the field: no waves, and you cannot lose', () => {
    const { g, t } = setup(lessonIndex('shield'));
    expect(g.spawning).toBe(false);
    play(g, t, 20);
    expect(g.hp).toBe(TUNE.maxHp);
    expect(g.state).toBe('play');
    expect(g.enemies.every(e => e.tag)).toBe(true);
  });

  it('the movement lesson ticks off lean left, lean right and duck, then moves on', () => {
    const { g, t } = setup();
    play(g, t, 0.2, () => intent({ head: { x: -25, y: 0 } }));
    expect(t.done).toBe(1);
    play(g, t, 0.2, () => intent({ head: { x: -25, y: 0 } }));
    expect(t.done).toBe(1); // the same move twice counts once
    play(g, t, 0.2, () => intent({ head: { x: 25, y: 0 } }));
    play(g, t, 0.2, () => intent({ head: { x: 0, y: 20 } }));
    expect(t.done).toBe(3);
    expect(t.completedFor).not.toBeNull();
    play(g, t, LESSON_PAUSE_S + 0.1);
    expect(t.lesson.id).toBe('punch');
  });

  it('the punch lesson counts hits on its dummy, and brings it back when it falls', () => {
    const { g, t } = setup(lessonIndex('punch'));
    expect(g.enemies.map(e => e.dummy)).toEqual([true]);
    const shoulder = { x: 20, y: 20 };
    play(g, t, 6, i => intent({ punches: i % 30 === 0 ? [{ hand: 'r', at: { x: 0, y: 8 }, shoulder, dir: null }] : [] }));
    expect(t.lesson.id !== 'punch' || t.done === LESSONS[lessonIndex('punch')].need).toBe(true);
  });

  it('dodging lessons count dodges: the pillar lesson passes by leaning away each time', () => {
    const { g, t } = setup(lessonIndex('pillar'));
    let passed = false;
    for (let i = 0; i < 60 * 30 && !passed; i++) {
      // lean away from whichever side the pillar is coming down
      const h = g.hazards.find(x => x.kind === 'stonePillar');
      g.step(1 / 60, intent({ head: { x: h ? -Math.sign(h.laneX) * 30 : 0, y: 0 } }));
      t.update(1 / 60, g.drainEvents());
      passed = t.completedFor !== null || t.lesson.id !== 'pillar';
    }
    expect(passed).toBe(true);
  });

  it('shield and X block lessons only count blocks made with that move', () => {
    const { g, t } = setup(lessonIndex('xblock'));
    play(g, t, 8, () => intent({ shield: true, hands: { l: hs(-15, 5, true), r: hs(15, 5, true) } }));
    expect(t.done).toBe(0); // shield blocks don't count here
    for (let i = 0; i < 60 * 12 && t.completedFor === null; i++) play(g, t, 1 / 60, () => intent({ xBlock: true }));
    expect(t.done).toBe(LESSONS[lessonIndex('xblock')].need);
  });

  it('the ultimate lesson charges the ultimate for you', () => {
    const { g, t } = setup(lessonIndex('ultimate'));
    expect(g.ultimateCharge).toBe(1);
    const shoulder = { x: 20, y: 20 };
    play(g, t, 0.1, i => intent({
      punches: i === 0 ? [{ hand: 'l', at: { x: -5, y: 8 }, shoulder, dir: null }, { hand: 'r', at: { x: 5, y: 8 }, shoulder, dir: null }] : [],
      casts: i === 3 ? [{ kind: 'ultimate', at: { x: 0, y: 10 } }] : [],
    }));
    expect(t.completedFor).not.toBeNull();
  });

  it('can skip ahead, go back, and finishes after the last lesson', () => {
    const { g, t } = setup();
    t.next();
    expect(t.lesson.id).toBe('punch');
    t.back();
    expect(t.lesson.id).toBe('move');
    t.go(LESSONS.length - 1);
    t.next();
    expect(t.finished).toBe(true);
    expect(g.enemies.length).toBeGreaterThan(0);
  });
});
