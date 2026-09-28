import { ghostAlpha } from '../campaign/ghostFade';
import type { Tutorial } from '../game/tutorial';
import { ghostPose } from '../render/ghost';
import type { LessonDemo } from '../render/lessonDemo';

const $ = (id: string) => document.getElementById(id)!;
const show = (id: string, on = true) => $(id).classList.toggle('hidden', !on);

/** Dodges are about your body, not your hands: they show the little figure instead of ghost hands. */
const FIGURE_LESSONS = new Set(['move', 'pillar', 'wave']);

/**
 * The one way lessons are taught, in the tutorial and in the campaign's practice alike: what to do
 * now in a few big words, dots for how many times, and ghost hands doing the move over your own
 * (the little figure for dodges). The ghost fades once you get it, and comes back if you stall.
 */
export class LessonCoach {
  private lesson: { t: Tutorial; index: number } | null = null;
  private done = 0;
  private sinceProgress = 0;
  private alpha = 1;

  constructor(private demo: LessonDemo) {}

  /** Per frame while a lesson runs; returns the ghost hands to draw (or null). */
  update(t: Tutorial, dt: number, now: number): { lessonId: string; alpha: number } | null {
    if (!this.lesson || this.lesson.t !== t || this.lesson.index !== t.index) {
      this.lesson = { t, index: t.index };
      this.done = t.done;
      this.sinceProgress = 0;
      this.alpha = 1;
    } else if (t.done !== this.done) {
      this.done = t.done;
      this.sinceProgress = 0;
    } else this.sinceProgress += dt;
    const complete = t.completedFor !== null;
    this.alpha = ghostAlpha(this.alpha, dt, t.done, this.sinceProgress, complete || t.finished);
    this.draw(t, complete, now);
    const id = t.lesson.id;
    return this.alpha > 0.001 && !FIGURE_LESSONS.has(id) && ghostPose(id, 0) ? { lessonId: id, alpha: this.alpha } : null;
  }

  hide(): void {
    this.lesson = null;
    show('lesson', false);
    document.body.classList.remove('tutorial');
  }

  private draw(t: Tutorial, complete: boolean, now: number): void {
    const l = t.lesson;
    show('lesson');
    document.body.classList.add('tutorial');
    $('lesson').classList.toggle('done', complete);
    $('lessonStep').textContent = t.lessons.length > 1 ? `${t.index + 1}/${t.lessons.length}` : '';
    $('lessonTitle').textContent = l.title;
    $('lessonCue').textContent = complete ? '✓ NICE!' : t.cue;
    const pips = $('lessonPips');
    if (pips.childElementCount !== l.need) pips.replaceChildren(...Array.from({ length: l.need }, () => document.createElement('i')));
    [...pips.children].forEach((p, i) => p.classList.toggle('on', i < t.done));
    const figure = FIGURE_LESSONS.has(l.id) || !ghostPose(l.id, 0);
    show('lessonDemo', figure && !complete);
    if (figure && !complete) this.demo.draw(l.id, now / 1000);
  }
}
