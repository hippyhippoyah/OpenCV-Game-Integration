import { describe, expect, it } from 'vitest';
import { Game } from '../game/game';
import { mulberry32 } from '../math';
import { Progress } from './progress';
import { CampaignRunner } from './runner';
import { STOPS } from './chapter1';

const make = () => { const g = new Game(mulberry32(1), 70, true); return g; };
const fresh = () => new CampaignRunner(Progress.load(null), make);
/** Advance until the state changes (or give up). */
function until(r: CampaignRunner, state: string, ready = true, seconds = 200) {
  for (let t = 0; t < seconds && r.state !== state; t += 1 / 30) r.update(1 / 30, ready, r.game?.drainEvents() ?? []);
  return r.state;
}
/** Finish whatever practice or fight is running, as if the player did it. */
function winFight(r: CampaignRunner) {
  for (let i = 0; i < 20000 && (r.state === 'practice' || r.state === 'fight'); i++) {
    if (r.state === 'practice') { r.practice!.completedFor = 99; r.practice!.finished = true; }
    r.game!.enemies.forEach(e => { e.hp = 0; });
    r.update(1 / 30, true, r.game!.drainEvents());
  }
}

describe('CampaignRunner', () => {
  it('walks to the first arena, needs the camera for a second, counts down, then practises and fights', () => {
    const r = fresh();
    expect(r.state).toBe('walk');
    expect(until(r, 'arena')).toBe('arena');
    expect(r.ren).toEqual(STOPS[0].ren);
    r.interact();
    expect(r.state).toBe('handoff');
    until(r, 'countdown', false, 2);
    expect(r.state).toBe('handoff'); // no camera: waits
    expect(until(r, 'countdown')).toBe('countdown');
    expect(until(r, 'practice')).toBe('practice');
    expect(r.ghostMove).toBe(STOPS[0].practice[0]);
  });

  it('Esc during the handoff goes back to the arena', () => {
    const r = fresh();
    until(r, 'arena');
    r.interact();
    r.back();
    expect(r.state).toBe('arena');
  });

  it('winning shows a result with flames and walking on saves the stop', () => {
    const p = Progress.load(null), r = new CampaignRunner(p, make);
    until(r, 'arena'); r.interact(); until(r, 'practice');
    winFight(r);
    expect(r.state).toBe('result');
    expect(r.result!.flames).toBeGreaterThanOrEqual(1);
    r.walkOn();
    expect(p.isDone(STOPS[0].id)).toBe(true);
    expect(r.state).toBe('walk');
  });

  it('a scroll on the path stops the walk; picking it up unlocks its move with a notification', () => {
    const r = fresh();
    until(r, 'arena'); r.interact(); until(r, 'practice'); winFight(r); r.walkOn();
    expect(until(r, 'scroll')).toBe('scroll');
    expect(r.allowed.has('shield')).toBe(false);
    r.interact();
    expect(r.allowed.has('shield')).toBe(true);
    expect(r.notes.some(n => n.kind === 'scroll' && n.text.includes('Flame Shield'))).toBe(true);
    expect(r.state).toBe('walk');
  });

  it('losing a fight offers a retry of the fight only', () => {
    const r = fresh();
    until(r, 'arena'); r.interact(); until(r, 'practice');
    r.practice!.finished = true;
    r.update(1 / 30, true, []);
    expect(r.state).toBe('fight');
    r.game!.hp = 0; r.game!.state = 'over';
    r.update(1 / 30, true, []);
    expect(r.state).toBe('lost');
    r.retry();
    expect(r.state).toBe('fight');
    expect(r.game!.hp).toBeGreaterThan(0);
  });

  it('skip jumps ahead to the next pause', () => {
    const r = fresh();
    r.skip();
    r.update(1, true, []);
    expect(r.state).toBe('arena');
  });

  it('beating Daro gives the Final Flame, a finisher practice, then the end of the chapter', () => {
    const p = Progress.load(null);
    for (const s of STOPS.slice(0, -1)) p.completeStop(s.id, 1);
    for (const s of STOPS) if (s.scroll) p.addScroll(s.scroll);
    const r = new CampaignRunner(p, make);
    r.replay(STOPS.length - 1);
    expect(r.state).toBe('arena');
    r.interact(); until(r, 'fight');
    expect(r.game!.boss).not.toBeNull();
    winFight(r);
    expect(r.state).toBe('result');
    r.walkOn();
    expect(r.state).toBe('scroll');
    r.interact();
    expect(r.allowed.has('finisher')).toBe(true);
    expect(r.state).toBe('practice');
    expect(r.ghostMove).toBe('ultimate');
    r.practice!.finished = true;
    r.update(1 / 30, true, []);
    expect(r.state).toBe('end');
    expect(p.data.chapterDone).toBe(true);
  });
});
