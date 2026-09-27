import type { BodyPoint, HandObs, Side, TrackingFrame } from '../input/types';
import type { Calibration } from './calibration';
import { clamp, dist, lerp, type Vec2 } from '../math';

/**
 * View space: world units relative to the eyes, x right, y down.
 * The screen shows roughly x ∈ ±80, y ∈ −45…55; the shoulders sit near y = 20.
 */
export interface HandState {
  pos: Vec2;
  vel: Vec2;
  /** 0 = fist … 1 = open, smoothed. */
  openness: number;
  /** Debounced open / closed. */
  open: boolean;
  /** 1 = palm faces the camera, 0 = edge-on (palms facing each other). */
  facing: number;
  /** Where the position came from: the hand tracker, the pose wrist, or a pose guess outside the picture. */
  source: 'hand' | 'arm' | 'estimate';
  /** The hand is inside the camera picture. */
  inView: boolean;
  /** View-space elbow from the pose, if the arm is tracked. */
  elbow: Vec2 | null;
  /** 0 = elbow bent … 1 = straight arm (3D), if known. */
  extension: number | null;
  /** Fist-punch mode: this arm has been pulled back and can punch again. */
  punchReady: boolean;
}

export type { Side };

export type PunchTrigger = 'extend' | 'open';

export interface Punch {
  hand: Side;
  /** Where the hand was (view space) when the punch fired. */
  at: Vec2;
  shoulder: Vec2;
  /** Aim from the 3D arm: sideways/vertical tangent of the punch angle; null without 3D pose. */
  dir: Vec2 | null;
}

export type CastKind = 'wall' | 'ultimate';

/** A two-hand move: fire wall (open hands sweep up) or ultimate (open hands spread apart). */
export interface Cast {
  kind: CastKind;
  /** View-space point between the hands when it was cast. */
  at: Vec2;
}

export interface Intent {
  /** A head and shoulders are visible. */
  present: boolean;
  /** Camera offset in world units: lean → x, duck → y (down is +). */
  head: Vec2;
  hands: { l: HandState | null; r: HandState | null };
  shoulders: { l: Vec2; r: Vec2 } | null;
  punches: Punch[];
  /** Both hands held open. */
  shield: boolean;
  casts: Cast[];
  /** Head turn/tilt, when the face is clearly visible. */
  face: TrackingFrame['face'];
  /** Shoulder line angle in radians (+ = right shoulder lower). */
  bodyTilt: number;
}

export const TUNING = {
  /**
   * What fires a punch. 'extend': a fist driven out by a fast-straightening arm (the hand stays
   * closed). 'open': a fist that opens at the end of a fast move.
   */
  punchTrigger: 'open' as PunchTrigger,
  /** 'extend': fire once the arm straightens past extendFireAbove (after rising by punchExtendRise); re-arm below extendRearmBelow. */
  extendFireAbove: 0.75, extendRearmBelow: 0.5, extendConfirmS: 0.05,
  leanUnitsPerSw: 40, maxLean: 30,
  duckUnitsPerSw: 40, minDuck: -10, maxDuck: 25,
  /** Hand offset from the shoulder centre (in shoulder widths) × scale = view units. */
  handScaleX: 40, handScaleY: 32, handOffsetY: 20,
  /** Hands below this (view y) are resting, not attacking. */
  raisedAboveY: 40,
  /** Exponential smoothing rate, 1/s. Higher = snappier but jittery. */
  smoothing: 18,
  lostGraceS: 0.5,
  /** Openness hysteresis: open above openAbove, back to a fist below fistBelow. */
  openAbove: 0.65, fistBelow: 0.35,
  /**
   * A punch needs, within punchWindowS before opening: a speed over punchSpeed (view units/s),
   * the hand growing by punchGrowth×, or the arm straightening by punchExtendRise.
   */
  punchWindowS: 0.35, punchSpeed: 60, punchGrowth: 1.12, punchExtendRise: 0.3,
  /** Wait this long before firing, so opening both hands for a shield doesn't also punch. */
  punchConfirmS: 0.08,
  /** Shield: both hands open and held (nearly) still for shieldHoldS; then it stays up while both are open. */
  shieldHoldS: 0.15, shieldMaxSpeed: 35,
  /**
   * Two-hand casts, judged on movement since both hands opened (at most castWindowS ago):
   * rising by wallRise view units = fire wall; spreading apart by ultimateSpread = ultimate.
   */
  castWindowS: 0.4, wallRise: 14, ultimateSpread: 24, castRefractoryS: 0.6,
  /** Experimental: only count palms facing each other (edge-on to the camera) as a shield. */
  shieldNeedsEdgeOnPalms: false, edgeOnBelow: 0.5,
  /** Pose wrists below this confidence are treated as guesses. */
  minWristVis: 0.5,
  /** The palm sits this fraction of the forearm beyond the pose wrist. */
  palmBeyondWrist: 0.25,
};

interface Track extends HandState {
  armed: boolean;
  lastSeen: number;
  hist: { t: number; x: number; y: number; speed: number; size: number | null; ext: number | null }[];
}

export interface InterpretState {
  head: Vec2 | null;
  l: Track | null;
  r: Track | null;
  lastT: number | null;
  pending: (Punch & { t: number })[];
  /** When both hands were first seen open together (null if they aren't). */
  bothOpenAt: number | null;
  /** When both open hands have been still since, working toward the shield. */
  stillSince: number | null;
  shieldOn: boolean;
  castReadyAt: number;
}

export const initialState = (): InterpretState => ({
  head: null, l: null, r: null, lastT: null, pending: [], bothOpenAt: null, stillSince: null, shieldOn: false, castReadyAt: -Infinity,
});

const SIDES = ['l', 'r'] as const;
const other = (s: Side): Side => (s === 'l' ? 'r' : 'l');

const smooth = (prev: Vec2 | null, next: Vec2, k: number): Vec2 =>
  prev ? { x: prev.x + (next.x - prev.x) * k, y: prev.y + (next.y - prev.y) * k } : { ...next };

const snapshot = (t: Track | null): HandState | null =>
  t && {
    pos: { ...t.pos }, vel: { ...t.vel }, openness: t.openness, open: t.open, facing: t.facing,
    source: t.source, inView: t.inView, elbow: t.elbow && { ...t.elbow }, extension: t.extension,
    punchReady: t.armed,
  };
const inPicture = (p: BodyPoint) => p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;

export function interpret(f: TrackingFrame, cal: Calibration, s: InterpretState): Intent {
  const dt = s.lastT === null ? 0 : Math.max(1e-3, f.t - s.lastT);
  s.lastT = f.t;
  const k = dt === 0 ? 1 : 1 - Math.exp(-TUNING.smoothing * dt);

  if (!f.head || !f.shoulderL || !f.shoulderR) {
    s.pending = [];
    s.bothOpenAt = s.stillSince = null;
    s.shieldOn = false;
    return {
      present: false, head: s.head ? { ...s.head } : { x: 0, y: 0 }, hands: { l: null, r: null },
      shoulders: null, punches: [], shield: false, casts: [], face: null, bodyTilt: 0,
    };
  }
  const sw = dist(f.shoulderL, f.shoulderR) || cal.sw;
  const mid = { x: (f.shoulderL.x + f.shoulderR.x) / 2, y: (f.shoulderL.y + f.shoulderR.y) / 2 };
  const toView = (p: Vec2): Vec2 => ({
    x: ((p.x - mid.x) / sw) * TUNING.handScaleX,
    y: TUNING.handOffsetY + ((p.y - mid.y) / sw) * TUNING.handScaleY,
  });

  s.head = smooth(s.head, {
    x: clamp(((f.head.x - cal.head.x) / sw) * TUNING.leanUnitsPerSw, -TUNING.maxLean, TUNING.maxLean),
    y: clamp(((f.head.y - cal.head.y) / sw) * TUNING.duckUnitsPerSw, TUNING.minDuck, TUNING.maxDuck),
  }, k);
  const shoulders = { l: toView(f.shoulderL), r: toView(f.shoulderR) };

  // Hands labelled by the arm they belong to; without a body, follow them from frame to frame.
  const obs = f.hands.slice(0, 2).map(h => ({ pos: toView(h.center), h }));
  const labelled = obs.length > 0 && obs.every(o => o.h.side) && new Set(obs.map(o => o.h.side)).size === obs.length;
  const picked = labelled ? bySide(obs.map(o => o.h.side!)) : assign(obs.map(o => o.pos), s.l, s.r, dt);
  const opened: Side[] = [];
  for (const side of SIDES) {
    const arm = f.arms[side], i = picked[side];
    const ext = arm?.extension ?? null;
    if (i !== null) {
      const o = obs[i];
      const r = updateTrack(s[side], o.pos, o.h, o.h.size / sw, ext, f.t, dt, k);
      s[side] = r.track;
      r.track.source = 'hand';
      r.track.inView = true;
      if (r.opened) opened.push(side);
    } else if (arm) {
      // The hand tracker lost this hand (blur, edge of frame): follow the pose wrist instead.
      const w = toView(arm.wrist), e = toView(arm.elbow);
      const palm = { x: w.x + (w.x - e.x) * TUNING.palmBeyondWrist, y: w.y + (w.y - e.y) * TUNING.palmBeyondWrist };
      const tr = updateTrack(s[side], palm, null, null, ext, f.t, dt, k).track;
      tr.inView = arm.wrist.vis >= TUNING.minWristVis && inPicture(arm.wrist);
      tr.source = tr.inView ? 'arm' : 'estimate';
      s[side] = tr;
    } else if (s[side] && f.t - s[side]!.lastSeen > TUNING.lostGraceS) {
      s[side] = null;
    }
    const tr = s[side];
    if (tr) tr.elbow = arm ? toView(arm.elbow) : null;
  }

  const extendMode = TUNING.punchTrigger === 'extend';
  const confirmS = extendMode ? TUNING.extendConfirmS : TUNING.punchConfirmS;
  if (extendMode) {
    // A fist driven out by a fast-straightening arm; pull the arm back to re-arm.
    for (const side of SIDES) {
      const tr = s[side];
      if (!tr || tr.extension === null || tr.source === 'estimate') continue;
      if (tr.extension < TUNING.extendRearmBelow) tr.armed = true;
      else if (tr.armed && !tr.open && tr.extension >= TUNING.extendFireAbove
        && extensionRise(tr) >= TUNING.punchExtendRise && tr.pos.y < TUNING.raisedAboveY) {
        tr.armed = false;
        s.pending.push({ hand: side, at: { ...tr.pos }, shoulder: { ...shoulders[side] }, dir: null, t: f.t });
      }
    }
  } else {
    // A hand that shot open after a fast move.
    for (const side of opened) {
      const tr = s[side]!;
      if (tr.pos.y > TUNING.raisedAboveY || !movedRecently(tr)) continue;
      s.pending.push({ hand: side, at: { ...tr.pos }, shoulder: { ...shoulders[side] }, dir: null, t: f.t });
    }
  }
  // Wait briefly before firing: if either hand opens meanwhile it's a shield, not a punch.
  const punches: Punch[] = [];
  s.pending = s.pending.filter(p => {
    const tr = s[p.hand];
    if (!tr || s[other(p.hand)]?.open || (extendMode && tr.open)) return false;
    if (f.t - p.t < confirmS) return true;
    punches.push({ hand: p.hand, at: { ...tr.pos }, shoulder: p.shoulder, dir: aimDir(f.arms[p.hand]?.reach ?? null) });
    return false;
  });

  // Two open hands: a quick sweep up is a fire wall, a quick spread is the ultimate, held still is the shield.
  const casts: Cast[] = [];
  const l = s.l, r = s.r;
  const bothOpen = !!l && !!r && l.inView && r.inView && l.open && r.open;
  if (!bothOpen) {
    s.bothOpenAt = s.stillSince = null;
    s.shieldOn = false;
  } else {
    if (s.bothOpenAt === null) s.bothOpenAt = f.t;
    const kind = f.t >= s.castReadyAt ? twoHandGesture(l, r, Math.max(s.bothOpenAt, f.t - TUNING.castWindowS)) : null;
    if (kind) {
      casts.push({ kind, at: { x: (l.pos.x + r.pos.x) / 2, y: (l.pos.y + r.pos.y) / 2 } });
      s.castReadyAt = f.t + TUNING.castRefractoryS;
      s.shieldOn = false;
      s.stillSince = null;
    } else if (!s.shieldOn) {
      const edgeOn = (t: Track) => !TUNING.shieldNeedsEdgeOnPalms || t.facing < TUNING.edgeOnBelow;
      const still = (t: Track) => Math.hypot(t.vel.x, t.vel.y) < TUNING.shieldMaxSpeed;
      if (!still(l) || !still(r) || !edgeOn(l) || !edgeOn(r)) s.stillSince = null;
      else if (s.stillSince === null) s.stillSince = f.t;
      else if (f.t - s.stillSince >= TUNING.shieldHoldS) s.shieldOn = true;
    }
  }
  const shield = s.shieldOn;

  return {
    present: true, head: { ...s.head }, hands: { l: snapshot(s.l), r: snapshot(s.r) }, shoulders, punches, shield, casts,
    face: f.face, bodyTilt: Math.atan2(f.shoulderR.y - f.shoulderL.y, f.shoulderR.x - f.shoulderL.x),
  };
}

/** How the two hands moved since `from`: mostly up → wall, mostly apart → ultimate. */
function twoHandGesture(l: Track, r: Track, from: number): CastKind | null {
  const l0 = l.hist.find(h => h.t >= from), r0 = r.hist.find(h => h.t >= from);
  if (!l0 || !r0) return null;
  const rise = Math.min(l0.y - l.pos.y, r0.y - r.pos.y);
  const spread = Math.hypot(l.pos.x - r.pos.x, l.pos.y - r.pos.y) - Math.hypot(l0.x - r0.x, l0.y - r0.y);
  if (rise >= TUNING.wallRise && rise > spread) return 'wall';
  if (spread >= TUNING.ultimateSpread && spread > rise) return 'ultimate';
  return null;
}

/** Sideways/vertical tangent of the punch angle from the 3D shoulder → wrist direction. */
function aimDir(reach: { x: number; y: number; z: number } | null): Vec2 | null {
  if (!reach) return null;
  // guard against arms pointing sideways (almost no forward component)
  const forward = Math.max(-reach.z, 0.25 * Math.hypot(reach.x, reach.y, reach.z));
  return forward > 0 ? { x: reach.x / forward, y: reach.y / forward } : null;
}

function bySide(sides: Side[]): Record<Side, number | null> {
  const i = (side: Side) => { const n = sides.indexOf(side); return n < 0 ? null : n; };
  return { l: i('l'), r: i('r') };
}

/** Match up to two observed hands to the left/right tracks by predicted position. */
function assign(obs: Vec2[], l: Track | null, r: Track | null, dt: number): Record<Side, number | null> {
  const predict = (t: Track): Vec2 => ({ x: t.pos.x + t.vel.x * dt, y: t.pos.y + t.vel.y * dt });
  if (obs.length === 0) return { l: null, r: null };
  if (obs.length >= 2) {
    if (l && r) {
      const pl = predict(l), pr = predict(r);
      const keep = dist(pl, obs[0]) + dist(pr, obs[1]), swap = dist(pl, obs[1]) + dist(pr, obs[0]);
      return keep <= swap ? { l: 0, r: 1 } : { l: 1, r: 0 };
    }
    const known = l ?? r;
    if (known) {
      const near = dist(predict(known), obs[0]) <= dist(predict(known), obs[1]) ? 0 : 1;
      return l ? { l: near, r: 1 - near } : { l: 1 - near, r: near };
    }
    return obs[0].x <= obs[1].x ? { l: 0, r: 1 } : { l: 1, r: 0 };
  }
  const o = obs[0];
  if (l && r) return dist(predict(l), o) <= dist(predict(r), o) ? { l: 0, r: null } : { l: null, r: 0 };
  // Only one track: a hand appearing well to its other side is the other hand.
  if (l) return o.x > l.pos.x + 25 ? { l: null, r: 0 } : { l: 0, r: null };
  if (r) return o.x < r.pos.x - 25 ? { l: 0, r: null } : { l: null, r: 0 };
  return o.x < 0 ? { l: 0, r: null } : { l: null, r: 0 };
}

/**
 * Move a track to a new position. `h` (the hand tracker's view of the hand) updates its shape;
 * without it — following the pose wrist — the last known shape is kept.
 */
function updateTrack(
  tr: Track | null, pos: Vec2, h: HandObs | null, size: number | null, ext: number | null, t: number, dt: number, k: number,
): { track: Track; opened: boolean } {
  if (!tr) {
    // A hand that appears already open doesn't count as opening.
    const open = h ? h.open : 0;
    const track: Track = {
      pos: { ...pos }, vel: { x: 0, y: 0 }, openness: open, open: open >= 0.5, facing: h ? h.facing : 1,
      source: 'hand', inView: true, elbow: null, extension: ext, punchReady: true, armed: true, lastSeen: t,
      hist: [{ t, x: pos.x, y: pos.y, speed: 0, size, ext }],
    };
    return { track, opened: false };
  }
  const prev = tr.pos;
  tr.pos = smooth(prev, pos, k);
  if (dt > 0) {
    const kv = 1 - Math.exp(-12 * dt);
    tr.vel = { x: lerp(tr.vel.x, (tr.pos.x - prev.x) / dt, kv), y: lerp(tr.vel.y, (tr.pos.y - prev.y) / dt, kv) };
  }
  tr.extension = ext === null ? null : tr.extension === null ? ext : lerp(tr.extension, ext, k);
  let opened = false;
  if (h) {
    tr.openness = lerp(tr.openness, h.open, k);
    tr.facing = lerp(tr.facing, h.facing, k);
    if (!tr.open && tr.openness > TUNING.openAbove) { tr.open = true; opened = true; }
    else if (tr.open && tr.openness < TUNING.fistBelow) tr.open = false;
  }
  tr.lastSeen = t;
  tr.hist.push({ t, x: tr.pos.x, y: tr.pos.y, speed: Math.hypot(tr.vel.x, tr.vel.y), size, ext: tr.extension });
  const keepS = Math.max(TUNING.punchWindowS, TUNING.castWindowS);
  while (tr.hist.length && t - tr.hist[0].t > keepS) tr.hist.shift();
  return { track: tr, opened };
}

/** History samples within the punch window (the history itself is kept longer, for casts). */
function punchWindow(tr: Track): Track['hist'] {
  const now = tr.hist[tr.hist.length - 1]?.t ?? 0;
  return tr.hist.filter(h => now - h.t <= TUNING.punchWindowS);
}

/** How much the arm straightened within the recent window (largest rise from an earlier low). */
function extensionRise(tr: Track): number {
  let minExt = Infinity, rise = 0;
  for (const h of punchWindow(tr)) {
    if (h.ext === null) continue;
    minExt = Math.min(minExt, h.ext);
    rise = Math.max(rise, h.ext - minExt);
  }
  return rise;
}

/** Fast across the screen, quickly growing (moving toward the camera), or the arm quickly straightening. */
function movedRecently(tr: Track): boolean {
  let minSize = Infinity, growth = 1, minExt = Infinity, rise = 0;
  for (const h of punchWindow(tr)) {
    if (h.speed > TUNING.punchSpeed) return true;
    if (h.size !== null) {
      minSize = Math.min(minSize, h.size);
      growth = Math.max(growth, h.size / minSize);
    }
    if (h.ext !== null) {
      minExt = Math.min(minExt, h.ext);
      rise = Math.max(rise, h.ext - minExt);
    }
  }
  return growth >= TUNING.punchGrowth || rise >= TUNING.punchExtendRise;
}
