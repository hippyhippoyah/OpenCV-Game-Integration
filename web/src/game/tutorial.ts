import type { AttackKind, Game, GameEvent } from './game';

/** Where a lesson's enemies stand, and what they do. Missing ones (knocked down) come back. */
interface Cast { tag: string; kind: 'dummy' | 'spirit' | 'earth'; x: number; z: number; only?: AttackKind; cd?: number; pace?: number }

export interface Lesson {
  id: string;
  title: string;
  /** How to do the move. */
  how: string;
  /** What to do to pass, with `need` as the count. */
  goal: string;
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
/** Tutorial enemies take a beating without falling, so the lesson never runs out of targets. */
const STURDY = 999;

export const LESSONS: Lesson[] = [
  {
    id: 'move', title: 'Move',
    how: 'Your head is your body here. Lean or step to the side to move; bend your knees to duck. The world tilts with you.',
    goal: 'Lean left, lean right, and duck', need: 3, cast: [],
    progress(g, _e, marks) {
      const before = marks.size;
      if (g.cam.x <= -20) marks.add('left');
      if (g.cam.x >= 20) marks.add('right');
      if (g.cam.y >= 15) marks.add('duck');
      return marks.size - before;
    },
  },
  {
    id: 'punch', title: 'Punch',
    how: 'Hold both fists up in guard. Snap a fist toward the camera to throw fire where it points — short, quick jabs work. The ring shows where it will land.',
    goal: 'Hit the dummy with punches', need: 5,
    cast: [{ tag: 'dummy', kind: 'dummy', x: 0, z: 7 }],
    progress: (_g, events) => count(events, e => e.type === 'hitEnemy' || e.type === 'killEnemy'),
  },
  {
    id: 'pillar', title: 'Dodge a pillar',
    how: 'The earthbender raises a stone pillar and shoves it down one side of you — watch the furrow. Lean or step the other way until the cue turns green.',
    goal: 'Dodge pillars', need: 2,
    cast: [{ tag: 'earth', kind: 'earth', x: 0, z: 8, only: 'pillar', cd: 1, pace: 2.5 }],
    progress: (_g, events) => count(events, e => e.type === 'dodged'),
  },
  {
    id: 'wave', title: 'Duck the wave',
    how: 'The water spirit sends a wave at head height. Duck (bend your knees) until the cue turns green and let it roll over you.',
    goal: 'Duck under waves', need: 2,
    cast: [{ tag: 'spirit', kind: 'spirit', x: 0, z: 8, only: 'slab', cd: 1, pace: 2 }],
    progress: (_g, events) => count(events, e => e.type === 'dodged'),
  },
  {
    id: 'shield', title: 'Flame shield',
    how: 'Open both hands and hold them still in front of you: fire burns between them. Put it over the red rings where water orbs will land.',
    goal: 'Block orbs with the shield', need: 3,
    cast: [{ tag: 'spirit', kind: 'spirit', x: 0, z: 8, only: 'orb', cd: 1, pace: 1.2 }],
    progress: (g, events) => (g.shield.on ? count(events, e => e.type === 'blocked') : 0),
  },
  {
    id: 'xblock', title: 'X block',
    how: 'Cross your forearms in front of your chest, fists up. While crossed, you block every orb that reaches you.',
    goal: 'Block orbs with the X block', need: 3,
    cast: [{ tag: 'spirit', kind: 'spirit', x: 0, z: 8, only: 'orb', cd: 1, pace: 1.2 }],
    progress: (g, events) => (g.xBlock ? count(events, e => e.type === 'blocked') : 0),
  },
  {
    id: 'palm', title: 'Palm push',
    how: 'Open one hand (keep the other a fist) and shove it at the camera — or open it as you push. A pillar of fire rolls out and burns through everything in its way.',
    goal: 'Send pillars of fire', need: 3,
    cast: [{ tag: 'a', kind: 'dummy', x: -20, z: 7 }, { tag: 'b', kind: 'dummy', x: 20, z: 9 }],
    progress: (_g, events) => count(events, e => e.type === 'pillar'),
  },
  {
    id: 'wall', title: 'Fire wall',
    how: 'Open both hands low, then sweep them up quickly. A wall of fire rises in front of you and stops attacks — even pillars.',
    goal: 'Raise fire walls', need: 2,
    cast: [{ tag: 'spirit', kind: 'spirit', x: 0, z: 9, only: 'orb', cd: 1.5, pace: 1.5 }],
    progress: (_g, events) => count(events, e => e.type === 'wall'),
  },
  {
    id: 'wallpush', title: 'Wall push',
    how: 'Hold both palms open at shoulder width (like the shield), then shove them both at the camera. The fire wall rolls forward over your enemies.',
    goal: 'Push fire walls', need: 2,
    cast: [{ tag: 'a', kind: 'dummy', x: -18, z: 7 }, { tag: 'b', kind: 'dummy', x: 18, z: 8 }],
    progress: (_g, events) => count(events, e => e.type === 'wallPush'),
  },
  {
    id: 'ultimate', title: 'Ultimate',
    how: 'Bring both open hands together in front of your chest, then fling them wide apart. A spinning blade of fire cuts down the whole field. It takes a while to recharge.',
    goal: 'Unleash the ultimate', need: 1,
    cast: [{ tag: 'a', kind: 'dummy', x: -24, z: 6 }, { tag: 'b', kind: 'dummy', x: 0, z: 9 }, { tag: 'c', kind: 'dummy', x: 24, z: 7 }],
    setup: g => { g.ultimateIn = 0; },
    progress: (_g, events) => count(events, e => e.type === 'ultimate'),
  },
];

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

  constructor(private g: Game, start = 0) {
    g.scripted();
    this.go(start);
  }

  get lesson(): Lesson { return LESSONS[this.index]; }

  /** Start lesson i (clamped), from scratch. */
  go(i: number): void {
    this.index = Math.max(0, Math.min(LESSONS.length - 1, i));
    this.done = 0;
    this.completedFor = null;
    this.finished = false;
    this.marks.clear();
    this.g.clearField();
    this.ensureCast();
    this.lesson.setup?.(this.g);
  }

  next(): void {
    if (this.index === LESSONS.length - 1) this.finished = true;
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

  /** Which of the "move" lesson's steps are done (for showing them ticked). */
  get marksDone(): ReadonlySet<string> { return this.marks; }

  /** Bring back any of the lesson's enemies that were knocked down (once they've faded away). */
  private ensureCast(): void {
    for (const c of this.lesson.cast) {
      if (this.g.enemies.some(e => e.tag === c.tag)) continue;
      this.g.addEnemy({ ...c, hp: c.kind === 'dummy' ? undefined : STURDY });
    }
  }
}
