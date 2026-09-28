import type { UiSound } from '../audio/sfx';
import { STOPS } from '../campaign/chapter1';
import type { Progress } from '../campaign/progress';
import { raidOpen } from '../campaign/raidRunner';
import { RAIDS } from '../campaign/raids';
import { BUILDINGS, embersAvailable, MAX_LEVEL, nextCost, upgrade, type BuildingId } from '../campaign/temple';

const $ = (id: string) => document.getElementById(id)!;
const show = (id: string, on = true) => $(id).classList.toggle('hidden', !on);

/** Little pictures of each building, in the fire palette. */
const BRAZIER = '<svg viewBox="0 0 54 54"><path d="M27 6c3 6 9 8 9 15a9 9 0 0 1-18 0c0-4 2-6 4-8 0 3 1 5 4 5 0-5-2-8 1-12z" fill="#ff8a3d"/><path d="M27 14c2 3 4 5 4 8a4 4 0 0 1-8 0c0-2 1-3 2-4 0 1 1 2 2 2 0-2 0-4 0-6z" fill="#ffd27a"/><path d="M12 28h30l-5 9H17z" fill="#8b6d4c" stroke="#3a2a1c" stroke-width="1.5"/><rect x="24" y="37" width="6" height="9" fill="#6b5238"/><rect x="16" y="46" width="22" height="4" rx="1" fill="#4a3526"/></svg>';
const ICONS: Record<BuildingId, string> = {
  brazierL: BRAZIER, brazierR: BRAZIER,
  wall: '<svg viewBox="0 0 54 54"><g fill="#8b6d4c" stroke="#3a2a1c" stroke-width="1.5"><rect x="4" y="30" width="15" height="10"/><rect x="19" y="30" width="15" height="10"/><rect x="34" y="30" width="16" height="10"/><rect x="11" y="20" width="15" height="10"/><rect x="26" y="20" width="15" height="10"/><rect x="4" y="40" width="46" height="8"/></g><path d="M4 20h7v-6h8v6h7v-6h8v6h7v-6h9v6" fill="none" stroke="#b08a60" stroke-width="2"/></svg>',
  shrine: '<svg viewBox="0 0 54 54"><path d="M6 20 27 8l21 12z" fill="#a8452a"/><rect x="8" y="20" width="38" height="4" fill="#5a2a1a"/><rect x="13" y="24" width="4" height="22" fill="#6a3a24"/><rect x="37" y="24" width="4" height="22" fill="#6a3a24"/><rect x="6" y="46" width="42" height="4" fill="#4a3526"/><path d="M27 28c2 4 5 5 5 9a5 5 0 0 1-10 0c0-2 1-3 2-4 0 2 1 3 2 3 0-3-1-5 1-8z" fill="#ffb45e"/></svg>',
};

export interface TempleHandlers {
  sound(s: UiSound): void;
  onRaid(index: number): void;
  onBack(): void;
}

/** The temple screen: spend embers on the defenses, and pick a raid from the ladder. */
export class TempleScreen {
  open = false;
  private justBuilt: BuildingId | null = null;

  constructor(private progress: Progress, private h: TempleHandlers) {}

  show(): void {
    this.open = true;
    show('temple');
    document.body.classList.add('front');
    this.render();
  }

  hide(): void {
    this.open = false;
    show('temple', false);
  }

  key(e: KeyboardEvent): boolean {
    if (!this.open) return false;
    if (e.key === 'Escape') { this.h.sound('back'); this.h.onBack(); return true; }
    return false;
  }

  private render(): void {
    const p = this.progress, have = embersAvailable(p);
    $('templeEmbers').textContent = have.toLocaleString();
    $('templeBuildings').replaceChildren(...BUILDINGS.map(b => {
      const lvl = p.level(b.id), cost = nextCost(p, b.id);
      const card = document.createElement('div');
      card.className = `building${lvl >= MAX_LEVEL ? ' max' : ''}${this.justBuilt === b.id ? ' just' : ''}`;
      card.innerHTML = `${ICONS[b.id]}<div class="info"><b></b><span></span><div class="pips">${[0, 1, 2].map(i => `<i class="${i < lvl ? 'on' : ''}"></i>`).join('')}</div></div>`;
      card.querySelector('b')!.textContent = b.name;
      card.querySelector('span')!.textContent = b.does;
      const btn = document.createElement('button');
      btn.textContent = cost === null ? 'Max' : `${lvl ? 'Upgrade' : 'Build'} ${cost.toLocaleString()}`;
      btn.disabled = cost === null || cost > have;
      btn.addEventListener('click', () => {
        if (!upgrade(p, b.id)) return;
        p.save();
        this.h.sound('flame');
        this.justBuilt = b.id;
        this.render();
      });
      card.append(btn);
      return card;
    }));
    this.justBuilt = null;
    // how to earn more, when there's more to earn
    const missing = STOPS.reduce((n, s) => n + 3 - p.flames(s.id), 0);
    $('templeNote').textContent = missing ? `${missing} campaign 🔥 still to earn — replay stops for more embers.` : '';
    $('templeRaidList').replaceChildren(...RAIDS.map((r, i) => {
      const open = raidOpen(p, i), flames = p.raidFlames(r.id), btn = document.createElement('button');
      btn.className = open ? '' : 'locked';
      btn.innerHTML = `<em>${i + 1}</em><b></b><span class="fl">${open ? [0, 1, 2].map(k => `<span class="${k < flames ? '' : 'off'}">🔥</span>`).join('') : '🔒'}</span>`;
      btn.querySelector('b')!.textContent = r.name;
      if (open) btn.addEventListener('click', () => { this.h.sound('select'); this.h.onRaid(i); });
      return btn;
    }));
  }
}
