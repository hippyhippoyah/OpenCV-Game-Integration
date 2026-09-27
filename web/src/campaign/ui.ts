import { BOSS } from '../game/boss';
import { LessonDemo } from '../render/lessonDemo';
import { SCROLLS, STOPS } from './chapter1';
import type { Progress, ScrollId } from './progress';
import type { CampaignRunner } from './runner';

export interface HandoffCheck { seen: boolean; handsUp: boolean; distance: 'ok' | 'close' | 'far' | 'unknown' }

const $ = (id: string) => document.getElementById(id)!;
const show = (id: string, on = true) => $(id).classList.toggle('hidden', !on);

/** Everything the campaign shows on top of the world and the fights. */
export class CampaignUI {
  private demos = new Map<ScrollId, LessonDemo>();
  private noteTimes: { el: HTMLElement; until: number }[] = [];

  constructor(private progress: Progress) {
    for (const sc of Object.values(SCROLLS)) {
      const box = document.createElement('div');
      box.className = 'scroll';
      box.id = `scroll-${sc.id}`;
      const cv = document.createElement('canvas');
      const name = document.createElement('b');
      name.textContent = sc.name;
      box.append(cv, name);
      $('campScrollList').append(box);
      this.demos.set(sc.id, new LessonDemo(cv));
    }
  }

  get overlayOpen(): boolean { return !$('campScrolls').classList.contains('hidden'); }

  toggleScrolls(on = $('campScrolls').classList.contains('hidden')): void { show('campScrolls', on); }

  hideAll(): void { show('camp', false); show('campPause', false); document.body.classList.remove('campaign'); }

  update(r: CampaignRunner, check: HandoffCheck, now: number, paused = false): void {
    show('camp');
    document.body.classList.add('campaign');
    const s = r.state, exploring = s === 'walk' || s === 'scroll' || s === 'arena';
    const pausable = s === 'practice' || s === 'fight';
    $('campControls').innerHTML = exploring
      ? '<span><kbd>E</kbd> interact</span><span><kbd>Space</kbd> skip ahead</span><span><kbd>Tab</kbd> scrolls</span><span><kbd>Click</kbd> a lit stop to replay it</span><span><kbd>Esc</kbd> menu</span>'
      : s === 'result' ? '<span><kbd>Enter</kbd> walk on</span><span><kbd>R</kbd> try again</span>'
      : s === 'lost' ? '<span><kbd>R</kbd> try again</span>'
      : pausable && paused ? '<span><kbd>Esc</kbd> resume</span>' : '<span><kbd>Esc</kbd> pause</span>';
    show('campPause', pausable && paused);
    // Ren's lines while exploring or at the start of a fight
    const ren = r.ren && (exploring || s === 'handoff');
    show('campRen', !!ren);
    if (ren) $('campRen').innerHTML = r.ren!.map(l => `<b>Ren:</b> ${l}`).join('<br>');
    // prompt
    const stop = STOPS[r.stop];
    const prompt = s === 'scroll' ? `<kbd>E</kbd> Pick up the scroll` : s === 'arena' ? `<kbd>E</kbd> Enter ${stop.place}` : '';
    show('campPrompt', !!prompt);
    $('campPrompt').innerHTML = prompt;
    // camera handoff
    show('campHandoff', s === 'handoff');
    if (s === 'handoff') {
      $('chkSeen').classList.toggle('ok', check.seen);
      $('chkHands').classList.toggle('ok', check.handsUp);
      $('chkDist').classList.toggle('ok', check.distance === 'ok');
      $('chkDist').textContent = check.distance === 'close' ? 'Step back a little' : check.distance === 'far' ? 'Come a little closer' : 'Good distance';
      $('campHandoffFill').style.width = `${Math.round(Math.min(1, r.handoffFor) * 100)}%`;
    }
    show('campCount', s === 'countdown');
    if (s === 'countdown') $('campCount').textContent = String(Math.max(1, Math.ceil(r.countdown)));
    // result / lost / end
    show('campResult', s === 'result');
    if (s === 'result' && r.result) {
      $('campResultTitle').textContent = STOPS[r.result.stop].place;
      $('campFlames').innerHTML = [0, 1, 2].map(i => `<span class="${i < r.result!.flames ? '' : 'off'}">🔥</span>`).join('');
      $('campReasons').innerHTML = r.result.reasons.map(x => `<li class="ok">${x}</li>`).join('');
    }
    show('campLost', s === 'lost');
    show('campEnd', s === 'end');
    // boss bar
    const boss = r.game?.boss ?? null;
    show('campBoss', s === 'fight' && !!boss);
    if (boss) $('campBossFill').style.width = `${Math.round((boss.hp / (boss.maxHp ?? BOSS.hp)) * 100)}%`;
    // notifications
    for (const n of r.notes.splice(0)) {
      const el = document.createElement('div');
      el.className = `note ${n.kind}`;
      el.textContent = n.text;
      $('campNotes').append(el);
      this.noteTimes.push({ el, until: now + 5000 });
    }
    this.noteTimes = this.noteTimes.filter(n => (now < n.until ? true : (n.el.remove(), false)));
    // scrolls inventory
    if (!$('campScrolls').classList.contains('hidden')) {
      for (const [id, demo] of this.demos) {
        const have = this.progress.hasScroll(id);
        $(`scroll-${id}`).classList.toggle('locked', !have);
        $(`scroll-${id}`).title = have ? '' : 'Found later on the path';
        if (have) demo.draw(SCROLLS[id].lessonId, now / 1000);
      }
    }
  }
}
