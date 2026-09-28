import type { GameEvent } from '../game/game';

/**
 * Sound effects, synthesised with Web Audio — no audio files. Every sound is built from a few
 * pieces (filtered noise for fire, air and rumble; swept tones for impacts and chimes) and each
 * play is varied (pitch, filter, length, stereo position), so an attack spammed ten times never
 * sounds the same twice. The audio context starts on the first key press or click (browsers
 * don't allow sound before that).
 */
export type UiSound = 'hover' | 'select' | 'back' | 'tick' | 'go' | 'scroll' | 'flame' | 'charged' | 'gathered' | 'blueReady' | 'reset';

const rnd = (a: number, b: number) => a + Math.random() * (b - a);
/** Nudge a value by up to ±k (a fraction). */
const vary = (v: number, k: number) => v * (1 + rnd(-k, k));

interface NoiseOpts {
  dur: number; type: BiquadFilterType; f0: number; f1?: number; q?: number;
  gain: number; attack?: number; pan?: number; delay?: number; crackle?: number;
}
interface ToneOpts {
  dur: number; wave: OscillatorType; f0: number; f1?: number;
  gain: number; attack?: number; pan?: number; delay?: number;
}

export class Sfx {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private comp: DynamicsCompressorNode | null = null;
  private noise: AudioBuffer | null = null;
  private volume = 0.7;
  /** Last time each sound played, to keep bursts from stacking into a roar. */
  private lastAt = new Map<string, number>();
  /** Continuous sounds (see ambient): a noise bed and a tone each, faded in and out. */
  private loops = new Map<string, { filter: BiquadFilterNode; ng: GainNode; osc: OscillatorNode; og: GainNode }>();

  constructor() {
    const wake = () => this.resume();
    addEventListener('keydown', wake);
    addEventListener('pointerdown', wake);
  }

  setVolume(v: number): void {
    this.volume = v;
    if (this.master) this.master.gain.value = v;
  }

  private resume(): void {
    if (!this.ctx) {
      const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AC) return;
      this.ctx = new AC();
      this.comp = this.ctx.createDynamicsCompressor();
      this.comp.threshold.value = -14;
      this.comp.ratio.value = 4;
      this.master = this.ctx.createGain();
      this.master.gain.value = this.volume;
      this.comp.connect(this.master).connect(this.ctx.destination);
      const len = this.ctx.sampleRate * 2;
      this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
      const d = this.noise.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    }
    if (this.ctx.state === 'suspended') void this.ctx.resume();
  }

  private ready(name: string, minGapS: number): AudioContext | null {
    const c = this.ctx;
    if (!c || c.state !== 'running' || this.volume <= 0) return null;
    const last = this.lastAt.get(name) ?? -1;
    if (c.currentTime - last < minGapS) return null;
    this.lastAt.set(name, c.currentTime);
    return c;
  }

  private out(c: AudioContext, pan = 0): AudioNode {
    if (!pan) return this.comp!;
    const p = c.createStereoPanner();
    p.pan.value = Math.max(-1, Math.min(1, pan));
    p.connect(this.comp!);
    return p;
  }

  /** Filtered noise with an attack/decay envelope and a filter sweep: fire, air, rumble, hiss. */
  private burst(c: AudioContext, o: NoiseOpts): void {
    const t = c.currentTime + (o.delay ?? 0), src = c.createBufferSource(), f = c.createBiquadFilter(), g = c.createGain();
    src.buffer = this.noise;
    src.playbackRate.value = rnd(0.9, 1.1);
    f.type = o.type;
    f.Q.value = o.q ?? 1;
    f.frequency.setValueAtTime(o.f0, t);
    if (o.f1) f.frequency.exponentialRampToValueAtTime(o.f1, t + o.dur);
    const a = o.attack ?? 0.01;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(o.gain, t + a);
    if (o.crackle) {
      // flickering amplitude: the crackle of a real fire
      for (let x = t + a; x < t + o.dur; x += rnd(0.015, 0.05)) g.gain.setValueAtTime(o.gain * rnd(1 - o.crackle, 1), x);
    }
    g.gain.exponentialRampToValueAtTime(0.0001, t + o.dur);
    src.connect(f).connect(g).connect(this.out(c, o.pan));
    src.start(t, rnd(0, 1.5));
    src.stop(t + o.dur + 0.05);
  }

  /** A swept tone: thumps, impacts, chimes. */
  private tone(c: AudioContext, o: ToneOpts): void {
    const t = c.currentTime + (o.delay ?? 0), osc = c.createOscillator(), g = c.createGain();
    osc.type = o.wave;
    osc.frequency.setValueAtTime(o.f0, t);
    if (o.f1) osc.frequency.exponentialRampToValueAtTime(o.f1, t + o.dur);
    const a = o.attack ?? 0.005;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(o.gain, t + a);
    g.gain.exponentialRampToValueAtTime(0.0001, t + o.dur);
    osc.connect(g).connect(this.out(c, o.pan));
    osc.start(t);
    osc.stop(t + o.dur + 0.05);
  }

  /**
   * The sounds that last as long as you hold something, set every frame: the shield's fire
   * crackling while it's up; the finisher's gather and the blue inferno's charge warming up — a
   * swell that rises in pitch and loudness as each fills. Pass null (menus, pauses) to fade all out.
   */
  ambient(g: { shield: { on: boolean }; gather: number; infernoPrep: number; ultimateIn: number; infernoIn: number } | null): void {
    const c = this.ctx;
    if (!c || c.state !== 'running') return;
    const shield = g?.shield.on ? 1 : 0, p = g?.gather ?? 0, q = g?.infernoPrep ?? 0;
    // (a move still recharging warms up only faintly)
    const pk = g && g.ultimateIn > 0 ? 0.35 : 1, qk = g && g.infernoIn > 0 ? 0.35 : 1;
    this.loop(c, 'shield', { type: 'lowpass', freq: rnd(900, 1300), noise: shield * rnd(0.35, 0.7), wave: 'sine', tone: 70, toneGain: shield * 0.06 });
    this.loop(c, 'warm', { type: 'bandpass', freq: 300 + 900 * p, noise: p > 0 ? pk * (0.25 + 0.9 * p) * rnd(0.8, 1) : 0, wave: 'sine', tone: 110 + 220 * p, toneGain: p > 0 ? pk * (0.05 + 0.14 * p) : 0 });
    this.loop(c, 'blue', { type: 'bandpass', freq: 600 + 1800 * q, noise: q > 0 ? qk * (0.2 + 0.8 * q) * rnd(0.8, 1) : 0, wave: 'triangle', tone: 220 + 440 * q, toneGain: q > 0 ? qk * (0.04 + 0.12 * q) : 0 });
  }

  private loop(c: AudioContext, id: string, o: { type: BiquadFilterType; freq: number; noise: number; wave: OscillatorType; tone: number; toneGain: number }): void {
    let L = this.loops.get(id);
    if (!L) {
      if (o.noise <= 0 && o.toneGain <= 0) return;
      const src = c.createBufferSource(), filter = c.createBiquadFilter(), ng = c.createGain(), osc = c.createOscillator(), og = c.createGain();
      src.buffer = this.noise;
      src.loop = true;
      filter.type = o.type;
      filter.Q.value = 0.9;
      ng.gain.value = 0;
      og.gain.value = 0;
      osc.type = o.wave;
      src.connect(filter).connect(ng).connect(this.comp!);
      osc.connect(og).connect(this.comp!);
      src.start();
      osc.start();
      L = { filter, ng, osc, og };
      this.loops.set(id, L);
    }
    const now = c.currentTime;
    L.filter.type = o.type;
    L.filter.frequency.setTargetAtTime(o.freq, now, 0.05);
    L.ng.gain.setTargetAtTime(o.noise, now, 0.06);
    L.osc.frequency.setTargetAtTime(o.tone, now, 0.08);
    L.og.gain.setTargetAtTime(o.toneGain, now, 0.08);
  }

  /** Stereo position of something at world x (the player is at camX). */
  static panOf(x: number, camX = 0): number { return Math.max(-0.8, Math.min(0.8, (x - camX) / 70)); }

  onEvent(e: GameEvent, camX = 0): void {
    const pan = 'x' in e ? Sfx.panOf(e.x, camX) : 0;
    switch (e.type) {
      case 'punch': this.punch(e.side === 'l' ? -0.35 : 0.35); break;
      case 'combo': this.combo(e.name, pan); break;
      case 'pillar': this.pillar(pan); break;
      case 'wall': this.wall(pan); break;
      case 'wallPush': this.wall(pan, true); break;
      case 'ultimate': this.ultimate(); break;
      case 'inferno': this.inferno(); break;
      case 'hitEnemy': this.hit(pan, false); break;
      case 'killEnemy': this.hit(pan, true); break;
      case 'blocked': this.block(pan); break;
      case 'clash': this.clash(pan); break;
      case 'cut': this.cut(pan); break;
      case 'playerHit': this.hurt(); break;
      case 'dodged': this.dodge(); break;
      case 'fizzle': this.fizzle(pan); break;
      case 'stonePillar': this.rumble(e.side * 0.6); break;
      case 'slab': this.wave(pan); break;
      case 'gameOver': this.gameOver(); break;
    }
  }

  // ---------- attacks ----------

  /** A jab: a quick whoosh of flame with a soft thump; never the same twice. */
  private punch(pan: number): void {
    const c = this.ready('punch', 0.03);
    if (!c) return;
    const f = vary(900, 0.3), d = vary(0.2, 0.25), p = pan + rnd(-0.15, 0.15);
    this.burst(c, { dur: d, type: 'bandpass', f0: f, f1: f * rnd(2.2, 3.4), q: rnd(0.7, 1.4), gain: vary(2.4, 0.2), attack: 0.012, pan: p });
    this.burst(c, { dur: d * 1.3, type: 'lowpass', f0: vary(1400, 0.2), f1: 300, gain: 1.0, attack: 0.02, pan: p, crackle: 0.5 });
    this.tone(c, { dur: 0.1, wave: 'sine', f0: vary(150, 0.2), f1: 55, gain: 0.7, pan: p });
  }

  private combo(name: string, pan: number): void {
    const c = this.ready(`combo-${name}`, 0.1);
    if (!c) return;
    switch (name) {
      case 'charged':
        // a deep roar with a crackle of blue energy on top
        this.burst(c, { dur: 0.7, type: 'lowpass', f0: 250, f1: vary(1800, 0.2), gain: 0.7, attack: 0.02, pan, crackle: 0.6 });
        this.burst(c, { dur: 0.35, type: 'highpass', f0: 3500, gain: 0.25, pan, crackle: 0.9 });
        this.tone(c, { dur: 0.45, wave: 'sine', f0: 90, f1: 40, gain: 0.6, pan });
        this.tone(c, { dur: 0.5, wave: 'triangle', f0: vary(1300, 0.05), f1: 700, gain: 0.08, pan, delay: 0.02 });
        break;
      case 'flurry':
        this.burst(c, { dur: 0.45, type: 'bandpass', f0: 600, f1: 2600, q: 0.8, gain: 2.2, pan, crackle: 0.4 });
        this.tone(c, { dur: 0.25, wave: 'sine', f0: 120, f1: 45, gain: 0.5, pan });
        this.chime(c, [660, 880], 0.06, 0.08);
        break;
      case 'finisher':
        break; // the ultimate event plays the big sound
      default:
        // one-two, volley, wall breaker, counter: a bright sting on top of their own sound
        this.chime(c, [587, 784, 988], 0.07, 0.08);
    }
  }

  private pillar(pan: number): void {
    const c = this.ready('pillar', 0.08);
    if (!c) return;
    this.burst(c, { dur: vary(0.75, 0.15), type: 'lowpass', f0: 180, f1: vary(1100, 0.25), q: 1.2, gain: 1.3, attack: 0.04, pan, crackle: 0.5 });
    this.tone(c, { dur: 0.5, wave: 'sine', f0: vary(80, 0.15), f1: 40, gain: 0.45, pan });
  }

  private wall(pan: number, push = false): void {
    const c = this.ready(push ? 'wallPush' : 'wall', 0.15);
    if (!c) return;
    this.burst(c, { dur: 1.2, type: 'lowpass', f0: 150, f1: vary(2200, 0.15), q: 0.8, gain: 0.75, attack: 0.08, pan, crackle: 0.55 });
    this.burst(c, { dur: 1.0, type: 'bandpass', f0: 400, f1: 1500, q: 0.6, gain: 0.3, attack: 0.15, pan: -pan, crackle: 0.6 });
    this.tone(c, { dur: push ? 0.9 : 0.6, wave: 'sine', f0: push ? 70 : 60, f1: 30, gain: push ? 0.8 : 0.5, pan });
  }

  private ultimate(): void {
    const c = this.ready('ultimate', 0.3);
    if (!c) return;
    // a rising rush, a boom, then the blade's long whirl
    this.burst(c, { dur: 0.35, type: 'bandpass', f0: 300, f1: 3000, q: 1, gain: 0.5, attack: 0.25 });
    this.tone(c, { dur: 1.2, wave: 'sine', f0: 70, f1: 28, gain: 0.9, delay: 0.3 });
    this.burst(c, { dur: 1.6, type: 'lowpass', f0: 2500, f1: 200, gain: 0.7, delay: 0.3, crackle: 0.5, pan: -0.4 });
    this.burst(c, { dur: 1.6, type: 'lowpass', f0: 2200, f1: 180, gain: 0.7, delay: 0.32, crackle: 0.5, pan: 0.4 });
    this.chime(c, [392, 523, 784], 0.09, 0.1, 0.3);
  }

  private inferno(): void {
    const c = this.ready('inferno', 0.3);
    if (!c) return;
    // the slam, then the ground roaring blue for its five seconds
    this.tone(c, { dur: 1.0, wave: 'sine', f0: 60, f1: 25, gain: 1 });
    this.burst(c, { dur: 0.4, type: 'lowpass', f0: 1200, f1: 150, gain: 0.8 });
    this.burst(c, { dur: 5, type: 'lowpass', f0: 900, f1: 400, gain: 0.45, attack: 0.3, delay: 0.1, crackle: 0.7, pan: -0.5 });
    this.burst(c, { dur: 5, type: 'lowpass', f0: 1000, f1: 450, gain: 0.45, attack: 0.3, delay: 0.15, crackle: 0.7, pan: 0.5 });
    this.burst(c, { dur: 4.5, type: 'highpass', f0: 4000, gain: 0.08, attack: 0.5, delay: 0.2, crackle: 0.95 });
    this.tone(c, { dur: 2.5, wave: 'triangle', f0: 1568, f1: 1175, gain: 0.05, attack: 0.3, delay: 0.1 });
  }

  // ---------- hits and defence ----------

  private hit(pan: number, kill: boolean): void {
    const c = this.ready(kill ? 'kill' : 'hit', 0.04);
    if (!c) return;
    this.tone(c, { dur: vary(0.14, 0.2), wave: 'sine', f0: vary(200, 0.2), f1: 60, gain: 0.95, pan });
    this.burst(c, { dur: 0.12, type: 'lowpass', f0: vary(2500, 0.3), f1: 400, gain: 1.3, pan });
    if (kill) {
      // the spirit breaks apart: a falling shimmer and a puff
      this.burst(c, { dur: 0.6, type: 'bandpass', f0: 2500, f1: 400, q: 2, gain: 1.2, attack: 0.03, pan, delay: 0.05 });
      this.tone(c, { dur: 0.5, wave: 'triangle', f0: vary(900, 0.1), f1: 300, gain: 0.16, pan, delay: 0.05 });
    }
  }

  private block(pan: number): void {
    const c = this.ready('block', 0.05);
    if (!c) return;
    this.burst(c, { dur: 0.25, type: 'bandpass', f0: vary(1800, 0.15), f1: 900, q: 3, gain: 1.8, pan });
    this.tone(c, { dur: 0.35, wave: 'triangle', f0: vary(620, 0.08), f1: 560, gain: 0.25, pan });
    this.tone(c, { dur: 0.15, wave: 'sine', f0: 140, f1: 60, gain: 0.6, pan });
  }

  private clash(pan: number): void {
    const c = this.ready('clash', 0.05);
    if (!c) return;
    this.burst(c, { dur: 0.35, type: 'highpass', f0: vary(2500, 0.2), gain: 1.1, pan, crackle: 0.8 });
    this.tone(c, { dur: 0.12, wave: 'sine', f0: 180, f1: 70, gain: 0.3, pan });
  }

  private cut(pan: number): void {
    const c = this.ready('cut', 0.03);
    if (!c) return;
    this.burst(c, { dur: 0.18, type: 'bandpass', f0: vary(3000, 0.2), f1: 800, q: 2, gain: 1.4, pan });
  }

  private hurt(): void {
    const c = this.ready('hurt', 0.2);
    if (!c) return;
    this.tone(c, { dur: 0.35, wave: 'sine', f0: 110, f1: 45, gain: 1.2 });
    this.burst(c, { dur: 0.4, type: 'lowpass', f0: 600, f1: 120, gain: 1.4 });
  }

  private dodge(): void {
    const c = this.ready('dodge', 0.2);
    if (!c) return;
    this.burst(c, { dur: 0.3, type: 'bandpass', f0: 500, f1: 1800, q: 1, gain: 0.9, attack: 0.05 });
    this.chime(c, [784, 1175], 0.07, 0.07);
  }

  private fizzle(pan: number): void {
    const c = this.ready('fizzle', 0.15);
    if (!c) return;
    this.burst(c, { dur: 0.35, type: 'highpass', f0: 2200, f1: 5000, gain: 0.7, attack: 0.01, pan, crackle: 0.9 });
  }

  // ---------- enemies ----------

  private rumble(pan: number): void {
    const c = this.ready('rumble', 0.2);
    if (!c) return;
    this.burst(c, { dur: 1.1, type: 'lowpass', f0: 90, f1: 260, q: 1, gain: 0.8, attack: 0.1, pan, crackle: 0.4 });
    this.tone(c, { dur: 0.8, wave: 'sine', f0: 50, f1: 35, gain: 0.5, attack: 0.08, pan });
  }

  private wave(pan: number): void {
    const c = this.ready('wave', 0.2);
    if (!c) return;
    this.burst(c, { dur: 1.2, type: 'bandpass', f0: 1800, f1: 350, q: 0.7, gain: 0.45, attack: 0.25, pan });
  }

  private gameOver(): void {
    const c = this.ready('over', 1);
    if (!c) return;
    this.chime(c, [392, 330, 262], 0.25, 0.12);
    this.burst(c, { dur: 1.5, type: 'lowpass', f0: 800, f1: 100, gain: 0.3, attack: 0.1 });
  }

  // ---------- interface ----------

  ui(s: UiSound): void {
    const c = this.ready(`ui-${s}`, s === 'hover' ? 0.04 : 0.05);
    if (!c) return;
    switch (s) {
      case 'hover': this.tone(c, { dur: 0.05, wave: 'sine', f0: vary(1400, 0.03), gain: 0.1 }); break;
      case 'select':
        this.chime(c, [523, 784], 0.06, 0.18);
        this.burst(c, { dur: 0.25, type: 'bandpass', f0: 600, f1: 2000, gain: 0.12, attack: 0.02 });
        break;
      case 'back': this.chime(c, [659, 440], 0.06, 0.08); break;
      case 'tick': this.tone(c, { dur: 0.15, wave: 'triangle', f0: 880, gain: 0.15 }); break;
      case 'go': this.chime(c, [880, 1320], 0.05, 0.16); this.burst(c, { dur: 0.5, type: 'lowpass', f0: 300, f1: 1800, gain: 0.35, crackle: 0.5 }); break;
      case 'scroll': this.chime(c, [523, 659, 784, 1047], 0.09, 0.12); break;
      case 'flame': this.chime(c, [vary(784, 0.02)], 0.1, 0.14); this.burst(c, { dur: 0.3, type: 'lowpass', f0: 400, f1: 1500, gain: 0.2 }); break;
      case 'charged': this.tone(c, { dur: 0.5, wave: 'triangle', f0: 1100, f1: 1600, gain: 0.08, attack: 0.05 }); this.burst(c, { dur: 0.3, type: 'highpass', f0: 3000, gain: 0.12, crackle: 0.9 }); break;
      case 'gathered': this.tone(c, { dur: 0.6, wave: 'sine', f0: 220, f1: 440, gain: 0.2, attack: 0.1 }); this.burst(c, { dur: 0.6, type: 'lowpass', f0: 300, f1: 1400, gain: 0.3, attack: 0.1, crackle: 0.5 }); break;
      case 'blueReady': this.tone(c, { dur: 0.8, wave: 'triangle', f0: 440, f1: 880, gain: 0.14, attack: 0.1 }); this.burst(c, { dur: 0.7, type: 'bandpass', f0: 800, f1: 3000, gain: 0.6, attack: 0.1, crackle: 0.7 }); break;
      case 'reset': this.chime(c, [440, 330, 220], 0.08, 0.1); break;
    }
  }

  /** A few bell-like notes in a row. */
  private chime(c: AudioContext, freqs: number[], step: number, gain: number, delay = 0): void {
    freqs.forEach((f, i) => {
      this.tone(c, { dur: 0.5, wave: 'triangle', f0: f, gain, delay: delay + i * step });
      this.tone(c, { dur: 0.35, wave: 'sine', f0: f * 2, gain: gain * 0.3, delay: delay + i * step });
    });
  }
}
