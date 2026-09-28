import { describe, expect, it } from 'vitest';
import { SETTINGS_KEY, Settings } from './settings';

const memory = () => { const m = new Map<string, string>(); return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v) }; };

describe('Settings', () => {
  it('defaults to the camera at sensitivity 1.4, and survives no storage at all', () => {
    const s = Settings.load(null);
    expect(s.data.input).toBe('camera');
    expect(s.data.sensitivity).toBe(1.4);
    s.save();
  });

  it('saves and loads, clamping a bad sensitivity', () => {
    const store = memory();
    const s = Settings.load(store);
    s.data.input = 'mock';
    s.data.sensitivity = 1.8;
    s.save();
    expect(Settings.load(store).data).toMatchObject({ input: 'mock', sensitivity: 1.8 });
    store.setItem(SETTINGS_KEY, JSON.stringify({ version: 1, sensitivity: 99 }));
    expect(Settings.load(store).data.sensitivity).toBe(2.5);
    store.setItem(SETTINGS_KEY, '{nope');
    expect(Settings.load(store).data.input).toBe('camera');
  });
});
