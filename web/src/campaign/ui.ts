import { BOSS } from '../game/boss';
import { LessonDemo } from '../render/lessonDemo';
import { pathOutline } from '../explore/world3d';
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

  constructor(private progress: Progress, private onMapPick: (stop: number) => void) {
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
    ($('campMapCanvas') as HTMLCanvasElement).addEventListener('click', e => this.mapClick(e));
  }

  get overlayOpen(): boolean { return !$('campMap').classList.contains('hidden') || !$('campScrolls').classList.contains('hidden'); }

  toggleMap(on = $('campMap').classList.contains('hidden')): void { show('campScrolls', false); show('campMap', on); if (on) this.drawMap(); }
  toggleScrolls(on = $('campScrolls').classList.contains('hidden')): void { show('campMap', false); show('campScrolls', on); }

  hideAll(): void { show('camp', false); document.body.classList.remove('campaign'); }

  update(r: CampaignRunner, check: HandoffCheck, now: number): void {
    show('camp');
    document.body.classList.add('campaign');
    const s = r.state, exploring = s === 'walk' || s === 'scroll' || s === 'arena';
    $('campControls').innerHTML = exploring
      ? '<span><kbd>Mouse</kbd> look</span><span><kbd>E</kbd> interact</span><span><kbd>Space</kbd> skip ahead</span><span><kbd>M</kbd> map</span><span><kbd>Tab</kbd> scrolls</span><span><kbd>Esc</kbd> menu</span>'
      : '<span><kbd>Esc</kbd> pause</span>';
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

  private mapLanterns: { x: number; y: number; stop: number }[] = [];

  private drawMap(): void {
    const cv = $('campMapCanvas') as HTMLCanvasElement, dpr = Math.min(devicePixelRatio || 1, 2);
    cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
    const c = cv.getContext('2d')!, W = cv.clientWidth, H = cv.clientHeight;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = '#140d1c'; c.fillRect(0, 0, W, H);
    const pts = pathOutline().map(p => ({ x: 30 + p.x * (W - 60), y: 20 + p.y * (H - 40) }));
    c.strokeStyle = '#6a5a70'; c.lineWidth = 4; c.beginPath();
    pts.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y))); c.stroke();
    this.mapLanterns = STOPS.map((s, i) => { const k = Math.round((s.pathAt / 300) * 60); return { ...pts[k], stop: i }; });
    const next = STOPS.findIndex(s => !this.progress.isDone(s.id));
    for (const l of this.mapLanterns) {
      const done = this.progress.isDone(STOPS[l.stop].id), isNext = l.stop === next;
      c.fillStyle = done ? '#ffb35c' : isNext ? '#ffe08a' : '#3a3040';
      c.beginPath(); c.arc(l.x, l.y, isNext ? 10 : 8, 0, 7); c.fill();
      c.fillStyle = '#fff4e4'; c.font = '12px Inter';
      c.fillText(`${STOPS[l.stop].place}${done ? ' ' + '🔥'.repeat(this.progress.flames(STOPS[l.stop].id)) : ''}`, l.x + 14, l.y + 4);
    }
  }

  private mapClick(e: MouseEvent): void {
    const r = (e.target as HTMLCanvasElement).getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const hit = this.mapLanterns.find(l => Math.hypot(l.x - x, l.y - y) < 14);
    if (hit && this.progress.isDone(STOPS[hit.stop].id)) { this.toggleMap(false); this.onMapPick(hit.stop); }
  }
}
