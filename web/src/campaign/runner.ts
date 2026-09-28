import type { Game, GameEvent, MoveName } from '../game/game';
import { LESSONS, Tutorial } from '../game/tutorial';
import { EPILOGUE, movesFor, SCROLLS, STOPS } from './chapter1';
import type { Progress, ScrollId } from './progress';
import { FightRunner } from './scripts';
import { pauses, Rail, type Pause } from '../explore/path';

export type CampaignState = 'walk' | 'scroll' | 'arena' | 'handoff' | 'countdown' | 'practice' | 'fight' | 'result' | 'lost' | 'end';
export interface Note { kind: 'scroll' | 'stop' | 'info'; text: string }
export interface Result { stop: number; flames: number; reasons: string[] }

/** Seconds the camera must see you before a fight starts, and the countdown after. */
export const HANDOFF_S = 1;
export const COUNTDOWN_S = 3;

/**
 * Chapter 1 as a state machine: walk the path ⇄ fight. Owns the fight's Game while there is one;
 * main.ts feeds it time, whether the camera sees you ready, and the game's events.
 */
export class CampaignRunner {
  state: CampaignState = 'walk';
  stop = 0;
  rail!: Rail;
  game: Game | null = null;
  practice: Tutorial | null = null;
  fight: FightRunner | null = null;
  result: Result | null = null;
  countdown = 0;
  handoffFor = 0;
  /** Ren's current lines (shown as subtitles), or null. */
  ren: string[] | null = null;
  readonly notes: Note[] = [];
  private pauseList: Pause[] = pauses();
  private usedNew = false;
  /** After Daro: the Final Flame and a finisher practice. */
  private epilogue = false;
  /** Set by `replay()`: the next `walkOn()` returns to your real progress instead of walking on from here. */
  private replaying = false;

  constructor(private progress: Progress, private makeGame: () => Game) {
    this.resume();
  }

  /** Put `stop`/`rail` at the first stop not yet done (or the last, if every stop is). Returns true if every stop is done. */
  private resume(): boolean {
    const next = STOPS.findIndex(s => !this.progress.isDone(s.id));
    this.stop = next < 0 ? STOPS.length - 1 : next;
    const prev = this.stop > 0 ? STOPS[this.stop - 1].pathAt : 0;
    this.rail = new Rail(prev + (this.stop > 0 ? 0.02 : 0));
    return next < 0;
  }

  get allowed(): Set<MoveName> { return movesFor(this.progress.data.scrolls); }

  get ghostMove(): string | null {
    return this.state === 'practice' && this.practice && !this.practice.finished ? this.practice.lesson.id : null;
  }

  update(dt: number, cameraReady: boolean, events: GameEvent[]): void {
    switch (this.state) {
      case 'walk': {
        const hit = this.rail.advance(dt, this.pauseList);
        if (!hit) break;
        // a scroll you already have, or an arena you've already cleared (replaying), doesn't stop you
        if (!this.stopsYou(hit)) { this.rail.leave(); break; }
        this.stop = hit.stop;
        this.ren = STOPS[hit.stop].ren;
        this.state = hit.kind;
        break;
      }
      case 'handoff':
        this.handoffFor = cameraReady ? this.handoffFor + dt : 0;
        if (this.handoffFor >= HANDOFF_S) { this.state = 'countdown'; this.countdown = COUNTDOWN_S; }
        break;
      case 'countdown':
        this.countdown -= dt;
        if (this.countdown <= 0) this.startStop();
        break;
      case 'practice':
        this.practice!.update(dt, events);
        if (this.practice!.finished) {
          if (this.epilogue) this.finishChapter();
          else this.startFight();
        }
        break;
      case 'fight': {
        this.noteUse(events);
        const out = this.fightOutcome(dt);
        if (out === 'won') this.win();
        else if (out === 'lost') this.state = 'lost';
        break;
      }
    }
  }

  /** Does this pause stop you: a scroll you don't have yet, or an arena you haven't cleared? */
  private stopsYou(p: Pause): boolean {
    const s = STOPS[p.stop];
    return p.kind === 'scroll' ? !!s.scroll && !this.progress.hasScroll(s.scroll) : !this.progress.isDone(s.id);
  }

  /** E: skip ahead while walking (to the next scroll or arena, never past it), pick up a scroll, or step into an arena. */
  interact(): void {
    if (this.state === 'walk') this.skip();
    else if (this.state === 'scroll') {
      const s = STOPS[this.stop], id = (this.epilogue ? s.reward : s.scroll) as ScrollId;
      if (this.progress.addScroll(id)) {
        this.notes.push({ kind: 'scroll', text: `New move: ${SCROLLS[id].name}` });
      }
      this.progress.save();
      if (this.epilogue) {
        this.ren = EPILOGUE.ren;
        this.beginPractice([EPILOGUE.lessonId]);
        return;
      }
      this.rail.leave();
      this.state = 'walk';
    } else if (this.state === 'arena') {
      this.state = 'handoff';
      this.handoffFor = 0;
    }
  }

  /** Jump straight to the next place that stops you (a scroll to pick up or a fight), never past it. */
  skip(): void {
    if (this.state !== 'walk') return;
    const next = this.pauseList.find(p => p.at > this.rail.d + 1e-6 && this.stopsYou(p));
    if (next) this.rail.skipTo(next.at);
  }

  back(): void {
    if (this.state === 'handoff' || this.state === 'countdown') this.state = 'arena';
  }

  retry(): void {
    if (this.state === 'lost') this.startFight();
  }

  walkOn(): void {
    if (this.state !== 'result') return;
    const s = STOPS[this.stop];
    this.progress.completeStop(s.id, this.result!.flames);
    this.progress.save();
    this.notes.push({ kind: 'stop', text: `${s.place} — ${'🔥'.repeat(this.result!.flames)}` });
    this.game = null;
    this.fight = null;
    if (s.reward && !this.progress.hasScroll(s.reward)) {
      this.epilogue = true;
      this.state = 'scroll';
      return;
    }
    if (this.replaying) {
      this.replaying = false;
      this.state = this.resume() ? 'end' : 'walk';
      return;
    }
    this.rail.leave();
    this.state = this.stop === STOPS.length - 1 ? 'end' : 'walk';
  }

  replay(stop: number): void {
    this.stop = stop;
    this.rail = new Rail(STOPS[stop].pathAt);
    this.ren = STOPS[stop].ren;
    this.state = 'arena';
    this.epilogue = false;
    this.replaying = true;
  }

  private startStop(): void {
    const s = STOPS[this.stop];
    if (s.practice.length) this.beginPractice(s.practice);
    else this.startFight();
  }

  private beginPractice(ids: string[]): void {
    this.game = this.makeGame();
    this.practice = new Tutorial(this.game, 0, ids.map(id => LESSONS.find(l => l.id === id)!));
    this.game.allowed = this.allowed;
    this.state = 'practice';
  }

  private startFight(): void {
    const s = STOPS[this.stop];
    this.game = this.makeGame();
    this.game.scripted();
    this.game.noDamage = false;
    this.game.allowed = this.allowed;
    this.game.label = s.place;
    this.fight = new FightRunner(this.game, s.fight);
    if (s.fight.boss) this.game.addBoss(0, 9);
    this.practice = null;
    this.usedNew = false;
    this.state = 'fight';
  }

  private fightOutcome(dt: number): 'fighting' | 'won' | 'lost' {
    const g = this.game!, out = this.fight!.update(dt);
    if (STOPS[this.stop].fight.boss) {
      if (g.state === 'over') return 'lost';
      return g.enemies.some(e => e.boss && e.hp > 0) ? 'fighting' : 'won';
    }
    return out;
  }

  /** Did you use this stop's new move? (for the third flame) */
  private noteUse(events: GameEvent[]): void {
    const m = STOPS[this.stop].newMove, g = this.game!;
    for (const e of events) {
      if ((m === 'flurry' && e.type === 'combo' && e.name === 'flurry')
        || (m === 'charge' && e.type === 'combo' && e.name === 'charged')
        || (m === 'palm' && e.type === 'pillar')
        || (m === 'wall' && e.type === 'wall')
        || (m === 'shield' && e.type === 'blocked' && g.shield.on)) this.usedNew = true;
    }
  }

  private win(): void {
    const s = STOPS[this.stop], g = this.game!, reasons = ['Finished'];
    if (g.hp >= 70) reasons.push('Took little damage');
    if (this.usedNew || s.newMove === null) reasons.push(s.newMove ? 'Used your new move' : 'Beat the boss');
    this.result = { stop: this.stop, flames: reasons.length, reasons };
    this.state = 'result';
  }

  private finishChapter(): void {
    this.progress.finishChapter();
    this.progress.save();
    this.epilogue = false;
    this.practice = null;
    this.game = null;
    this.notes.push({ kind: 'info', text: 'Chapter 1 complete' });
    this.state = 'end';
  }
}
