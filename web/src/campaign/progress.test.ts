import { describe, expect, it } from 'vitest';
import { Progress, PROGRESS_KEY, type StoreLike } from './progress';

const memory = (): StoreLike & { data: Record<string, string> } => {
  const data: Record<string, string> = {};
  return { data, getItem: k => data[k] ?? null, setItem: (k, v) => { data[k] = v; } };
};

describe('Progress', () => {
  it('starts empty', () => {
    const p = Progress.load(memory());
    expect(p.data.scrolls).toEqual([]);
    expect(p.isDone('courtyard')).toBe(false);
  });

  it('remembers scrolls, stops (best flames) and the chapter across loads', () => {
    const store = memory();
    const p = Progress.load(store);
    expect(p.addScroll('flameShield')).toBe(true);
    expect(p.addScroll('flameShield')).toBe(false);
    p.completeStop('courtyard', 2);
    p.completeStop('courtyard', 1);
    p.finishChapter();
    p.save();
    const q = Progress.load(store);
    expect(q.hasScroll('flameShield')).toBe(true);
    expect(q.flames('courtyard')).toBe(2);
    expect(q.data.chapterDone).toBe(true);
  });

  it('starts fresh when storage is missing, broken or throws', () => {
    expect(Progress.load(null).data.scrolls).toEqual([]);
    const bad = memory();
    bad.data[PROGRESS_KEY] = '{not json';
    expect(Progress.load(bad).data.scrolls).toEqual([]);
    const throwing: StoreLike = { getItem: () => { throw new Error('denied'); }, setItem: () => { throw new Error('denied'); } };
    const p = Progress.load(throwing);
    p.addScroll('heldBreath');
    expect(() => p.save()).not.toThrow();
  });

  it('can be reset', () => {
    const store = memory();
    const p = Progress.load(store);
    p.addScroll('flameShield');
    p.reset();
    expect(Progress.load(store).data.scrolls).toEqual([]);
  });
});
