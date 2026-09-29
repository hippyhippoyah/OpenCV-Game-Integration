import { STOPS } from '../campaign/chapter1';
import type { Progress } from '../campaign/progress';
import { activeLessons } from '../game/tutorial';
import type { UiSound } from '../audio/sfx';
import { embersAvailable, templeOpen } from '../campaign/temple';
import { FEATURES } from '../config';
import { clampSensitivity, SENSITIVITY_MAX, SENSITIVITY_MIN, type InputKind, type Settings } from './settings';

export type PlayMode = 'campaign' | 'tutorial' | 'training';
type ItemId = PlayMode | 'temple' | 'settings';
export type Screen = 'title' | 'menu' | 'settings';

interface Item { id: ItemId; title: string; blurb: string }

const ITEMS: Item[] = [
  { id: 'campaign', title: 'Campaign', blurb: 'Chapter 1 · The Ember Path' },
  { id: 'temple', title: 'Temple', blurb: 'Build it up · hold off raids' },
  { id: 'tutorial', title: 'Tutorial', blurb: 'Learn every move' },
  { id: 'training', title: 'Training', blurb: 'Dummies that never fight back' },
  { id: 'settings', title: 'Settings', blurb: 'Input, sensitivity, progress' },
];

const $ = (id: string) => document.getElementById(id)!;
const RAIDS_WON = (p: Progress) => Object.keys(p.data.raids ?? {}).length;
const show = (id: string, on = true) => $(id).classList.toggle('hidden', !on);
const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls = '', text = ''): HTMLElementTagNameMap[K] => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
};

export interface MenuHandlers {
  onPlay(mode: PlayMode, lesson?: number): void;
  onVolume(v: number): void;
  sound(s: UiSound): void;
  onInput(kind: InputKind): void;
  onSensitivity(v: number): void;
  onResetCampaign(): void;
  onTemple(): void;
}

/** The front of the game: title screen, main menu (with a detail panel per mode) and settings. */
export class Menu {
  screen: Screen | null = null;
  private sel = 0;
  private resetArmed = false;

  constructor(private settings: Settings, private progress: Progress, private h: MenuHandlers) {
    $('title').addEventListener('click', () => { this.h.sound('select'); this.showMenu(); });
    $('setBack').addEventListener('click', () => { this.h.sound('back'); this.showMenu(); });
    for (const b of document.querySelectorAll<HTMLButtonElement>('#setInput button')) {
      b.addEventListener('click', () => this.setInput(b.dataset.input as InputKind));
    }
    const range = $('setSens') as HTMLInputElement;
    range.min = String(SENSITIVITY_MIN);
    range.max = String(SENSITIVITY_MAX);
    range.step = '0.1';
    range.addEventListener('input', () => this.setSensitivity(Number(range.value)));
    const vol = $('setVol') as HTMLInputElement;
    vol.addEventListener('input', () => {
      this.settings.data.volume = Math.min(1, Math.max(0, Number(vol.value) / 100));
      this.settings.save();
      this.h.onVolume(this.settings.data.volume);
      $('setVolVal').textContent = `${Math.round(this.settings.data.volume * 100)}%`;
    });
    vol.addEventListener('change', () => this.h.sound('select'));
    $('setReset').addEventListener('click', () => this.reset());
  }

  showTitle(): void { this.go('title'); }

  /** What the menu offers: the temple only once Chapter 1 is done (and while it's switched on, see FEATURES). */
  private get items(): Item[] {
    return ITEMS.filter(it => it.id !== 'temple' || (FEATURES.temple && templeOpen(this.progress)));
  }

  showMenu(note = ''): void {
    this.go('menu');
    $('menuNote').textContent = note;
    show('menuNote', !!note);
    this.renderList();
    this.select(this.sel);
  }

  showSettings(): void {
    this.go('settings');
    this.resetArmed = false;
    this.renderSettings();
  }

  hide(): void { this.go(null); }

  /** Keyboard on the front screens; returns true if the key was used. */
  key(e: KeyboardEvent): boolean {
    if (!this.screen) return false;
    const k = e.key;
    if (this.screen === 'title') {
      if (k === 'Tab' || k === 'Shift' || k === 'Meta' || k === 'Alt' || k === 'Control') return false;
      e.preventDefault();
      this.h.sound('select');
      this.showMenu();
      return true;
    }
    if (this.screen === 'settings') {
      if (k === 'Escape') { this.h.sound('back'); this.showMenu(); return true; }
      return false;
    }
    if (k === 'ArrowDown' || k === 's' || k === 'S') { e.preventDefault(); this.select((this.sel + 1) % this.items.length); return true; }
    if (k === 'ArrowUp' || k === 'w' || k === 'W') { e.preventDefault(); this.select((this.sel + this.items.length - 1) % this.items.length); return true; }
    if (k === 'Enter' || k === ' ') { e.preventDefault(); this.activate(this.sel); return true; }
    if (k === 'Escape') { this.h.sound('back'); this.showTitle(); return true; }
    return false;
  }

  private go(s: Screen | null): void {
    this.screen = s;
    show('title', s === 'title');
    show('menu', s === 'menu');
    show('settings', s === 'settings');
    document.body.classList.toggle('front', s !== null);
  }

  private get started(): boolean {
    return Object.keys(this.progress.data.stops).length > 0 || this.progress.data.scrolls.length > 0;
  }

  private renderList(): void {
    const list = $('menuList');
    list.replaceChildren(...this.items.map((it, i) => {
      const b = el('button', 'item');
      b.dataset.id = it.id;
      b.append(el('b', '', it.title), el('span', '', it.blurb));
      b.addEventListener('mouseenter', () => this.select(i));
      b.addEventListener('focus', () => this.select(i));
      b.addEventListener('click', () => this.activate(i));
      return b;
    }));
    this.sel = Math.min(this.sel, this.items.length - 1);
    const camp = $('menuList').querySelector<HTMLElement>('[data-id="campaign"] b')!;
    camp.textContent = this.started ? 'Continue' : 'Campaign';
    $('menuInput').textContent = this.settings.data.input === 'camera' ? 'Camera' : 'Mouse & keys';
  }

  private select(i: number): void {
    if (i !== this.sel && this.screen === 'menu') this.h.sound('hover');
    this.sel = i;
    $('menuList').querySelectorAll('.item').forEach((b, j) => b.classList.toggle('on', j === i));
    this.renderDetail(this.items[i].id);
  }

  private activate(i: number): void {
    const id = this.items[i].id;
    this.h.sound('select');
    if (id === 'settings') this.showSettings();
    else if (id === 'temple') this.h.onTemple();
    else this.h.onPlay(id);
  }

  private renderDetail(id: ItemId): void {
    const d = $('menuDetail');
    d.replaceChildren();
    const head = (kicker: string, title: string) => d.append(el('div', 'kicker', kicker), el('h2', '', title));
    const para = (t: string) => d.append(el('p', '', t));
    const stat = (rows: [string, string][]) => {
      const s = el('div', 'stats');
      for (const [v, l] of rows) { const c = el('div'); c.append(el('b', '', v), el('span', '', l)); s.append(c); }
      d.append(s);
    };
    switch (id) {
      case 'campaign': {
        const done = STOPS.filter(s => this.progress.isDone(s.id)).length;
        const flames = STOPS.reduce((n, s) => n + this.progress.flames(s.id), 0);
        head('Chapter 1', 'The Ember Path');
        para('Master Ren is away and the Spirit Moon is rising. Walk down the mountain from the Ember Temple, find Ren\'s scrolls, learn their moves — and stop Daro Stonefist at the village gate.');
        stat([[`${done} / ${STOPS.length}`, 'Stops cleared'], [`${this.progress.data.scrolls.length}`, 'Scrolls found'], [`${flames} / ${STOPS.length * 3}`, 'Flames']]);
        d.append(el('div', 'cta', this.started ? 'Enter — continue your journey' : 'Enter — begin'));
        break;
      }
      case 'temple': {
        head('After Chapter 1', 'The Temple');
        para('Spend the embers the campaign earned you on braziers, a wall and a shrine. Then hold off the raids.');
        stat([[embersAvailable(this.progress).toLocaleString(), 'Embers'], [`${RAIDS_WON(this.progress)} / 6`, 'Raids held']]);
        d.append(el('div', 'cta', 'Enter — go to the temple'));
        break;
      }
      case 'tutorial': {
        head('Learn', 'Tutorial');
        para('Every move, one lesson at a time, with nothing that can hurt you. Start from the top, or jump to a lesson:');
        const chips = el('div', 'lessons');
        activeLessons().forEach((l, i) => {
          const b = el('button', '', `${i + 1}. ${l.title}`);
          b.addEventListener('click', e => { e.stopPropagation(); this.h.onPlay('tutorial', i); });
          chips.append(b);
        });
        d.append(chips);
        break;
      }
      case 'training':
        head('Practice', 'Training');
        para('A quiet yard of straw dummies that never fight back. Try any move, combo or ultimate as often as you like.');
        d.append(el('div', 'cta', 'Enter — start'));
        break;
      case 'settings':
        head('Options', 'Settings');
        para('Choose how you play — your webcam, or mouse and keys — tune how easily punches trigger, or start the campaign over.');
        break;
    }
  }

  private renderSettings(): void {
    const input = this.settings.data.input;
    for (const b of document.querySelectorAll<HTMLButtonElement>('#setInput button')) b.classList.toggle('on', b.dataset.input === input);
    const range = $('setSens') as HTMLInputElement;
    range.value = String(this.settings.data.sensitivity);
    $('setSensVal').textContent = `×${this.settings.data.sensitivity.toFixed(1)}`;
    ($('setVol') as HTMLInputElement).value = String(Math.round(this.settings.data.volume * 100));
    $('setVolVal').textContent = `${Math.round(this.settings.data.volume * 100)}%`;
    $('setReset').textContent = this.started ? 'Reset campaign progress' : 'No campaign progress yet';
    ($('setReset') as HTMLButtonElement).disabled = !this.started;
    $('setReset').classList.remove('confirm');
  }

  private setInput(kind: InputKind): void {
    this.h.sound('select');
    this.settings.data.input = kind;
    this.settings.save();
    this.h.onInput(kind);
    this.renderSettings();
  }

  private setSensitivity(v: number): void {
    this.settings.data.sensitivity = clampSensitivity(v);
    this.settings.save();
    this.h.onSensitivity(this.settings.data.sensitivity);
    $('setSensVal').textContent = `×${this.settings.data.sensitivity.toFixed(1)}`;
  }

  /** Reset asks twice: the first click arms it, the second wipes scrolls, stops and flames. */
  private reset(): void {
    if (!this.resetArmed) {
      this.resetArmed = true;
      $('setReset').textContent = 'Click again to erase all scrolls, stops and flames';
      $('setReset').classList.add('confirm');
      return;
    }
    this.resetArmed = false;
    this.h.sound('reset');
    this.h.onResetCampaign();
    this.renderSettings();
    $('setReset').textContent = 'Campaign progress reset';
  }
}
