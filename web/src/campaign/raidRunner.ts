import type { Game } from '../game/game';
import { movesFor, SCROLLS } from './chapter1';
import type { Progress, ScrollId } from './progress';
import { RAIDS } from './raids';
import { FightRunner } from './scripts';
import { templeEffects } from './temple';
import { COUNTDOWN_S, HANDOFF_S } from './runner';

export type RaidState = 'handoff' | 'countdown' | 'fight' | 'result' | 'lost';
export interface RaidResult { flames: number; reasons: string[] }

/** A raid is open once the one before it is won (the first always is). */
export function raidOpen(p: Progress, i: number): boolean {
  return i === 0 || p.raidDone(RAIDS[i - 1].id);
}

/**
 * One raid on the temple: step back into view, count down, fight with the temple's defenses,
 * then a result (🔥 like a campaign stop, saved) or a loss (try again, nothing lost).
 */
export class RaidRunner {
  state: RaidState = 'handoff';
  game: Game | null = null;
  fight: FightRunner | null = null;
  result: RaidResult | null = null;
  countdown = 0;
  handoffFor = 0;

  constructor(private progress: Progress, readonly index: number, private makeGame: () => Game) {}

  get raid() { return RAIDS[this.index]; }

  update(dt: number, cameraReady: boolean): void {
    switch (this.state) {
      case 'handoff':
        this.handoffFor = cameraReady ? this.handoffFor + dt : 0;
        if (this.handoffFor >= HANDOFF_S) { this.state = 'countdown'; this.countdown = COUNTDOWN_S; }
        break;
      case 'countdown':
        this.countdown -= dt;
        if (this.countdown <= 0) this.start();
        break;
      case 'fight': {
        const out = this.fight!.update(dt);
        if (out === 'won') this.win();
        else if (out === 'lost') this.state = 'lost';
        break;
      }
    }
  }

  /** Fight it again (after losing, or from the result card). */
  retry(): void {
    if (this.state === 'lost' || this.state === 'result') this.start();
  }

  private start(): void {
    const g = this.makeGame();
    g.scripted();
    g.noDamage = false;
    // every move Chapter 1 teaches, and whatever the temple has built so far
    g.allowed = movesFor(Object.keys(SCROLLS) as ScrollId[]);
    g.temple = templeEffects(this.progress);
    g.label = this.raid.name;
    this.game = g;
    this.fight = new FightRunner(g, this.raid.fight);
    this.result = null;
    this.state = 'fight';
  }

  private win(): void {
    const hp = this.game!.hp, reasons = ['Held the temple'];
    if (hp >= 70) reasons.push('Took little damage');
    if (hp >= 95) reasons.push('Untouched');
    this.result = { flames: reasons.length, reasons };
    this.progress.completeRaid(this.raid.id, reasons.length);
    this.progress.save();
    this.state = 'result';
  }
}
