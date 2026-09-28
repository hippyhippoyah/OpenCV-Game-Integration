/** Scrolls are moves, found along the campaign path. */
export type ScrollId = 'flameShield' | 'risingPillar' | 'heldBreath' | 'burningWall' | 'finalFlame';

/** The bit of `localStorage` progress needs (tests pass an in-memory one). */
export interface StoreLike { getItem(k: string): string | null; setItem(k: string, v: string): void }

export interface ProgressData {
  version: 1;
  scrolls: ScrollId[];
  /** Finished stops and the most flames earned there. */
  stops: Record<string, { flames: number }>;
  chapterDone: boolean;
  /** The temple's building levels (0 = not built); missing in saves from before the temple. */
  temple?: Record<string, number>;
  /** Raids won and the most flames earned in each. */
  raids?: Record<string, { flames: number }>;
}

export const PROGRESS_KEY = 'firebending.campaign.v1';

const fresh = (): ProgressData => ({ version: 1, scrolls: [], stops: {}, chapterDone: false });

/** Campaign progress, kept on this device. Storage failing never breaks the game. */
export class Progress {
  private constructor(private store: StoreLike | null, public data: ProgressData) {}

  static load(store: StoreLike | null): Progress {
    try {
      const raw = store?.getItem(PROGRESS_KEY);
      const d = raw ? (JSON.parse(raw) as ProgressData) : null;
      if (d && d.version === 1 && Array.isArray(d.scrolls) && d.stops) return new Progress(store, d);
    } catch { /* unreadable: start fresh */ }
    return new Progress(store, fresh());
  }

  hasScroll(id: ScrollId): boolean { return this.data.scrolls.includes(id); }

  /** Returns true if it's a new scroll. */
  addScroll(id: ScrollId): boolean {
    if (this.hasScroll(id)) return false;
    this.data.scrolls.push(id);
    return true;
  }

  /** A stop finished with `flames` (0–3); the best is kept. */
  completeStop(stopId: string, flames: number): void {
    this.data.stops[stopId] = { flames: Math.max(flames, this.data.stops[stopId]?.flames ?? 0) };
  }

  flames(stopId: string): number { return this.data.stops[stopId]?.flames ?? 0; }
  isDone(stopId: string): boolean { return stopId in this.data.stops; }
  finishChapter(): void { this.data.chapterDone = true; }

  /** A temple building's level (0 = not built). */
  level(building: string): number { return this.data.temple?.[building] ?? 0; }
  setLevel(building: string, level: number): void { (this.data.temple ??= {})[building] = level; }

  /** A raid won with `flames` (0–3); the best is kept. */
  completeRaid(raidId: string, flames: number): void {
    const raids = (this.data.raids ??= {});
    raids[raidId] = { flames: Math.max(flames, raids[raidId]?.flames ?? 0) };
  }
  raidFlames(raidId: string): number { return this.data.raids?.[raidId]?.flames ?? 0; }
  raidDone(raidId: string): boolean { return !!this.data.raids && raidId in this.data.raids; }

  save(): void {
    try { this.store?.setItem(PROGRESS_KEY, JSON.stringify(this.data)); } catch { /* storage full or blocked */ }
  }

  reset(): void {
    this.data = fresh();
    this.save();
  }
}
