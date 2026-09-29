import type { AttackKind, ComboName, Game, GameEvent } from './game';
import { lessonOn } from '../config';

/** Where a lesson's enemies stand, and what they do. Missing ones (knocked down) come back. */
interface Cast { tag: string; kind: 'dummy' | 'spirit' | 'earth'; x: number; z: number; only?: AttackKind; cd?: number; pace?: number }

export interface Lesson {
  id: string;
  title: string;
  /**
   * What to do right now, in a few words (shown big in the middle): it follows the move's steps,
   * e.g. "HOLD A FIST AT YOUR HIP" → "NOW PUNCH!".
   */
  cue(g: Game, marks: ReadonlySet<string>): string;
  /** How many times to do it to pass. */
  need: number;
  cast: Cast[];
  /** Set up anything else the lesson needs (e.g. charge the ultimate). */
  setup?(g: Game): void;
  /**
   * Progress this frame: the events that just happened, and marks already reached (for lessons
   * made of different things to do). Returns how many new steps were done.
   */
  progress(g: Game, events: GameEvent[], marks: Set<string>): number;
}

const count = (events: GameEvent[], ok: (e: GameEvent) => boolean) => events.filter(ok).length;
const combos = (events: GameEvent[], name: ComboName) => count(events, e => e.type === 'combo' && e.name === name);
const charged = (g: Game) => Math.max(g.hands.l?.charge ?? 0, g.hands.r?.charge ?? 0);
/** Tutorial enemies take a beating without falling, so the lesson never runs out of targets. */
const STURDY = 999;

export const LESSONS: Lesson[] = [
  {
    id: 'move', title: 'Move', need: 3, cast: [],
    cue: (_g, m) => (!m.has('left') ? '← LEAN LEFT' : !m.has('right') ? 'LEAN RIGHT →' : '↓ DUCK'),
    progress(g, _e, marks) {
      const before = marks.size;
      if (g.cam.x <= -20) marks.add('left');
      if (g.cam.x >= 20) marks.add('right');
      if (g.cam.y >= 15) marks.add('duck');
      return marks.size - before;
    },
  },
  {
    id: 'punch', title: 'Punch', need: 5,
    cue: () => 'PUNCH THE DUMMY',
    cast: [{ tag: 'dummy', kind: 'dummy', x: 0, z: 7 }],
    progress: (_g, events) => count(events, e => e.type === 'hitEnemy' || e.type === 'killEnemy'),
  },
  {
    id: 'charge', title: 'Charged punch', need: 2,
    cue: g => (charged(g) >= 1 ? 'NOW PUNCH!' : charged(g) > 0 ? 'HOLD IT…' : 'HOLD A FIST AT YOUR HIP'),
    cast: [{ tag: 'dummy', kind: 'dummy', x: 0, z: 7 }],
    progress: (_g, events) => combos(events, 'charged'),
  },
  {
    id: 'flurry', title: 'Flurry', need: 2,
    cue: () => '3 QUICK PUNCHES',
    cast: [{ tag: 'a', kind: 'dummy', x: -12, z: 7 }, { tag: 'b', kind: 'dummy', x: 12, z: 7 }],
    progress: (_g, events) => combos(events, 'flurry'),
  },
  {
    id: 'pillar', title: 'Dodge a pillar', need: 2,
    cue: () => 'STEP AWAY FROM THE PILLAR',
    cast: [{ tag: 'earth', kind: 'earth', x: 0, z: 8, only: 'pillar', cd: 1, pace: 2.5 }],
    progress: (_g, events) => count(events, e => e.type === 'dodged'),
  },
  {
    id: 'wave', title: 'Duck the wave', need: 2,
    cue: () => '↓ DUCK UNDER THE WAVE',
    cast: [{ tag: 'spirit', kind: 'spirit', x: 0, z: 8, only: 'slab', cd: 1, pace: 2 }],
    progress: (_g, events) => count(events, e => e.type === 'dodged'),
  },
  {
    id: 'shield', title: 'Flame shield', need: 3,
    cue: g => (g.shield.on ? 'BLOCK THE ORBS' : 'PALMS FACING · HOLD STILL'),
    cast: [{ tag: 'spirit', kind: 'spirit', x: 0, z: 8, only: 'orb', cd: 1, pace: 1.2 }],
    progress: (g, events) => (g.shield.on ? count(events, e => e.type === 'blocked') : 0),
  },
  {
    id: 'xblock', title: 'X block', need: 3,
    cue: g => (g.xBlock ? 'HOLD IT — BLOCK THE ORBS' : 'CROSS YOUR ARMS'),
    cast: [{ tag: 'spirit', kind: 'spirit', x: 0, z: 8, only: 'orb', cd: 1, pace: 1.2 }],
    progress: (g, events) => (g.xBlock ? count(events, e => e.type === 'blocked') : 0),
  },
  {
    id: 'palm', title: 'Palm push', need: 3,
    cue: () => 'OPEN HAND · SHOVE FORWARD',
    cast: [{ tag: 'a', kind: 'dummy', x: -20, z: 7 }, { tag: 'b', kind: 'dummy', x: 20, z: 9 }],
    progress: (_g, events) => count(events, e => e.type === 'pillar'),
  },
  {
    id: 'onetwo', title: 'One-two push', need: 2,
    cue: () => 'JAB · JAB · PUSH',
    cast: [{ tag: 'a', kind: 'dummy', x: -14, z: 7 }, { tag: 'b', kind: 'dummy', x: 14, z: 8 }],
    progress: (_g, events) => combos(events, 'oneTwo'),
  },
  {
    id: 'volley', title: 'Pillar volley', need: 2,
    cue: () => 'PUSH ONE HAND, THEN THE OTHER',
    cast: [{ tag: 'a', kind: 'dummy', x: -24, z: 7 }, { tag: 'b', kind: 'dummy', x: 0, z: 9 }, { tag: 'c', kind: 'dummy', x: 24, z: 7 }],
    progress: (_g, events) => combos(events, 'volley'),
  },
  {
    id: 'wall', title: 'Fire wall', need: 2,
    cue: () => 'SWEEP BOTH HANDS UP',
    cast: [{ tag: 'spirit', kind: 'spirit', x: 0, z: 9, only: 'orb', cd: 1.5, pace: 1.5 }],
    progress: (_g, events) => count(events, e => e.type === 'wall'),
  },
  {
    id: 'wallbreaker', title: 'Wall breaker', need: 2,
    cue: g => (g.walls.some(w => w.vz === 0) ? 'NOW PUSH BOTH PALMS!' : 'SWEEP BOTH HANDS UP'),
    cast: [{ tag: 'a', kind: 'dummy', x: -18, z: 7 }, { tag: 'b', kind: 'dummy', x: 18, z: 8 }],
    progress: (_g, events) => combos(events, 'wallBreaker'),
  },
  {
    id: 'ultimate', title: 'Finisher', need: 1,
    cue: g => (g.gather >= 1 ? 'SPREAD THEM WIDE!' : 'HANDS TOGETHER · HOLD'),
    cast: [{ tag: 'a', kind: 'dummy', x: -24, z: 6 }, { tag: 'b', kind: 'dummy', x: 0, z: 9 }, { tag: 'c', kind: 'dummy', x: 24, z: 7 }],
    setup: g => { g.ultimateIn = 0; },
    progress: (_g, events) => count(events, e => e.type === 'ultimate'),
  },
  {
    id: 'inferno', title: 'Blue Inferno', need: 1,
    cue: g => (g.infernoSpreadIn > 0 ? 'SPREAD YOUR HANDS!' : g.infernoPrep >= 1 ? 'SLAM THEM DOWN!' : 'HANDS TOGETHER OVERHEAD'),
    cast: [{ tag: 'a', kind: 'dummy', x: -24, z: 6 }, { tag: 'b', kind: 'dummy', x: 0, z: 10 }, { tag: 'c', kind: 'dummy', x: 24, z: 7 }],
    setup: g => { g.infernoIn = 0; },
    progress: (_g, events) => count(events, e => e.type === 'inferno'),
  },
  {
    id: 'lightning', title: 'Lightning', need: 2,
    cue: g => (g.lightningCharged ? 'AIM · THRUST TO STRIKE' : g.lightningCalling > 0 ? 'HOLD… CALL IT DOWN' : 'BOTH FINGER GUNS UP HIGH'),
    cast: [{ tag: 'a', kind: 'dummy', x: -24, z: 7 }, { tag: 'b', kind: 'dummy', x: 0, z: 9 }, { tag: 'c', kind: 'dummy', x: 24, z: 7 }],
    setup: g => { g.lightningIn = 0; },
    // (no waiting out the recharge in the lesson)
    progress: (g, events) => { const n = count(events, e => e.type === 'lightning'); if (n) g.lightningIn = 0; return n; },
  },
];

/** The lessons for moves that are switched on (see FEATURES). */
export const activeLessons = (): Lesson[] => LESSONS.filter(l => lessonOn(l.id));

/** Seconds to celebrate a finished lesson before moving on to the next. */
export const LESSON_PAUSE_S = 2;

/** Runs the lessons on a scripted game: sets each one up, tracks its goal, moves on when it's met. */
export class Tutorial {
  index = 0;
  done = 0;
  /** Seconds since the current lesson was completed (null while it's still going). */
  completedFor: number | null = null;
  /** Every lesson is done. */
  finished = false;
  private marks = new Set<string>();

  constructor(private g: Game, start = 0, readonly lessons: Lesson[] = activeLessons()) {
    g.scripted();
    this.go(start);
  }

  get lesson(): Lesson { return this.lessons[this.index]; }

  /** Start lesson i (clamped), from scratch. */
  go(i: number): void {
    this.index = Math.max(0, Math.min(this.lessons.length - 1, i));
    this.done = 0;
    this.completedFor = null;
    this.finished = false;
    this.marks.clear();
    this.g.clearField();
    this.ensureCast();
    this.lesson.setup?.(this.g);
  }

  next(): void {
    if (this.index === this.lessons.length - 1) this.finished = true;
    else this.go(this.index + 1);
  }

  back(): void { this.go(this.index - 1); }

  /** Call after each game step with the events it produced. Returns true when the lesson just got completed. */
  update(dt: number, events: GameEvent[]): boolean {
    if (this.finished) return false;
    this.ensureCast();
    if (this.completedFor !== null) {
      this.completedFor += dt;
      if (this.completedFor >= LESSON_PAUSE_S) this.next();
      return false;
    }
    this.done = Math.min(this.lesson.need, this.done + this.lesson.progress(this.g, events, this.marks));
    if (this.done < this.lesson.need) return false;
    this.completedFor = 0;
    return true;
  }

  /** What to do right now (see Lesson.cue). */
  get cue(): string { return this.lesson.cue(this.g, this.marks); }

  /** Bring back any of the lesson's enemies that were knocked down (once they've faded away). */
  private ensureCast(): void {
    for (const c of this.lesson.cast) {
      if (this.g.enemies.some(e => e.tag === c.tag)) continue;
      this.g.addEnemy({ ...c, hp: c.kind === 'dummy' ? undefined : STURDY });
    }
  }
}
