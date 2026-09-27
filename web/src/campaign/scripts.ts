import type { AttackKind, Game } from '../game/game';

/** One enemy in a fight script (the same options as `Game.addEnemy`). */
export interface ScriptEnemy { kind: 'dummy' | 'spirit' | 'earth'; x: number; z: number; only?: AttackKind; pace?: number; cd?: number; hp?: number }
/** Enemies that arrive `at` seconds into the fight. */
export interface Group { at: number; enemies: ScriptEnemy[] }
export type Goal = { type: 'defeat' } | { type: 'survive'; seconds: number };
/** A campaign fight: who comes when, and what wins it. `boss` fights are driven by the boss instead. */
export interface FightScript { groups: Group[]; goal: Goal; boss?: boolean }
export type FightOutcome = 'fighting' | 'won' | 'lost';

/** Runs a fight script on a scripted game: sends groups in on time and says when it's won or lost. */
export class FightRunner {
  elapsed = 0;
  private sent = 0;

  constructor(private g: Game, private script: FightScript) {}

  update(dt: number): FightOutcome {
    if (this.g.state === 'over') return 'lost';
    this.elapsed += dt;
    const groups = this.script.groups;
    while (this.sent < groups.length && groups[this.sent].at <= this.elapsed) {
      for (const e of groups[this.sent].enemies) this.g.addEnemy({ ...e, tag: `g${this.sent}` });
      this.sent++;
    }
    const goal = this.script.goal;
    if (goal.type === 'survive') return this.elapsed >= goal.seconds ? 'won' : 'fighting';
    const standing = this.g.enemies.some(e => e.hp > 0);
    return this.sent === groups.length && !standing ? 'won' : 'fighting';
  }

  get progress(): number {
    const goal = this.script.goal;
    if (goal.type === 'survive') return Math.min(1, this.elapsed / goal.seconds);
    const total = this.script.groups.reduce((n, g) => n + g.enemies.length, 0) || 1;
    const standing = this.g.enemies.filter(e => e.hp > 0).length;
    const toCome = this.script.groups.slice(this.sent).reduce((n, g) => n + g.enemies.length, 0);
    return 1 - (standing + toCome) / total;
  }
}
