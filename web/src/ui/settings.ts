import type { StoreLike } from '../campaign/progress';

/** How you play: bending in front of the webcam, or mouse and keys standing in for your hands. */
export type InputKind = 'camera' | 'mock';

export interface SettingsData {
  version: 1;
  input: InputKind;
  /** Fist punch sensitivity (TUNING.punchSensitivity). */
  sensitivity: number;
  /** Sound effects volume, 0 … 1. */
  volume: number;
}

export const SETTINGS_KEY = 'flowbound.settings.v1';
export const SENSITIVITY_MIN = 0.5, SENSITIVITY_MAX = 2.5;

const fresh = (): SettingsData => ({ version: 1, input: 'camera', sensitivity: 1.4, volume: 0.7 });

/** Player settings, kept on this device. Storage failing never breaks the game. */
export class Settings {
  private constructor(private store: StoreLike | null, public data: SettingsData) {}

  static load(store: StoreLike | null): Settings {
    try {
      const raw = store?.getItem(SETTINGS_KEY);
      const d = raw ? (JSON.parse(raw) as Partial<SettingsData>) : null;
      if (d && d.version === 1) return new Settings(store, { ...fresh(), ...d, sensitivity: clampSensitivity(d.sensitivity ?? 1.4) });
    } catch { /* unreadable: defaults */ }
    return new Settings(store, fresh());
  }

  save(): void {
    try { this.store?.setItem(SETTINGS_KEY, JSON.stringify(this.data)); } catch { /* storage full or blocked */ }
  }
}

export function clampSensitivity(v: number): number {
  return Math.round(Math.min(SENSITIVITY_MAX, Math.max(SENSITIVITY_MIN, v)) * 10) / 10;
}
