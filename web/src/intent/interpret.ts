import type { BodyPoint, HandObs, Side, TrackingFrame } from '../input/types';
import type { Calibration } from './calibration';
import { clamp, dist, lerp, type Vec2 } from '../math';
import { OneEuro } from './oneEuro';
import { FOCAL_H } from '../input/landmarks';

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
  /** Fist-punch mode: how far the fist came forward within the last quickWindowS (m). */
  punchRise: number | null;
  /** How far the hand is in front of the shoulders (m, filtered), and its guard baseline; null without 3D data. */
  reach: number | null;
  reachBase: number | null;
  /** How much that reading wobbles while the fist is still (m): grows with distance from the camera. */
  reachNoise: number | null;
  /** Where a punch from this hand would go: sideways/vertical tangent from its 3D position; null if unknown. */
  aimDir: Vec2 | null;
  /** Charged punch: 0 … 1 while a fist is pulled back and held; 1 = charged, the next punch is blue. */
  charge: number;
}

export type { Side };

export type PunchTrigger = 'extend' | 'open';

export interface Punch {
  hand: Side;
  /** Where the hand was (view space) when the punch fired. */
  at: Vec2;
  shoulder: Vec2;
  /** Aim from the fist's 3D position: sideways/vertical tangent of the punch angle; null if unknown. */
  dir: Vec2 | null;
  /** Thrown from a charged fist (pulled back and held): a blue fireball. */
  charged?: boolean;
}

export type PalmKind = 'push';

/**
 * A heavy single-hand move with an open palm (the other hand not open): a push toward the camera
 * sends a pillar of fire rolling forward.
 */
export interface Palm {
  kind: PalmKind;
  hand: Side;
  /** Where the hand was (view space) when it fired. */
  at: Vec2;
  shoulder: Vec2;
  dir: Vec2 | null;
}

export type CastKind = 'wall' | 'ultimate' | 'push';

/**
 * A two-hand move: fire wall (open hands sweep up), ultimate (open hands start together and fly
 * apart) or wall push (both open palms shoved toward the camera: a fire wall rolls forward).
 */
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
  palms: Palm[];
  /** Both hands held open. */
  shield: boolean;
  /** Forearms crossed in front of the chest. */
  xBlock: boolean;
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
  punchTrigger: 'extend' as PunchTrigger,
  /**
   * Fist punches are a quick jolt toward the camera: a fist that came quickRise metres closer within
   * quickWindowS — any distance out, even partway — and more than the other fist did by quickLead
   * (so moving your whole body doesn't count). It re-arms as soon as it comes back rearmDrop from
   * the punch's peak, so short jabs can be fired rapidly; refireS apart at the least. See
   * fistThresholds(). reachRearm: a fist within this of its guard counts as resting.
   */
  quickRise: 0.065, quickWindowS: 0.2, quickLead: 0.05, rearmDrop: 0.05, refireS: 0.12, reachRearm: 0.08, extendConfirmS: 0.05,
  /** A fist's reach reading settles for this long (s) after it appears before it can punch. */
  reachWarmupS: 0.6,
  /**
   * With a noisy camera the jolt needed rises with the reading's wobble (noiseQuick × wobble), but
   * never past quickRiseCap. Above reachNoiseMax the HUD suggests stepping closer.
   */
  noiseQuick: 8, noiseLead: 5, noiseRearm: 4, quickRiseCap: 0.2, quickLeadCap: 0.1, reachNoiseMax: 0.017,
  /**
   * Leaning, stepping or ducking moves your fists too (a lean twists the torso), which the reach
   * reading can't tell from a punch. While the head moves faster than leanFreeSpeed (view
   * units/s), a punch or palm push needs leanPenalty m more forward movement per unit/s over it.
   * It uses the fastest the head moved in the last leanMemoryS, so it holds through the turnaround
   * at the end of a sway — the moment the leaning-side fist sits furthest forward.
   * Punches thrown from a steady stance measured under ~65 on a real camera; fast leans 125–140.
   */
  leanFreeSpeed: 70, leanPenalty: 0.0015, headSpeedRate: 15, leanMemoryS: 0.5,
  /**
   * An unmistakably sharp jolt — sharpJolt m forward within sharpWindowS — is a punch even while
   * the body moves (swaying on a real camera moved a fist at most ~6 cm in 0.1 s).
   */
  sharpJolt: 0.09, sharpWindowS: 0.1,
  /**
   * Charged punch: a fist held for chargeHoldS in a charging pose, with the body steady (swaying
   * moves fists about too), charges; it stays charged for chargeKeepS after leaving the pose, until
   * it punches or opens. Poses, in view units (shoulder widths × 32) from its shoulder — screen
   * positions and arm shape only, as the camera's distance reading is too rough for this:
   * - at the hip: the fist hipBelow…hipBelowMax below the shoulder, with the elbow flared out
   *   elbowFlare past it or the arm bent (extension under hipMaxExtension) — unlike a relaxed arm
   *   hanging straight down;
   * - cocked by the ear: the fist raised earAbove above the shoulder — about eye or temple height,
   *   above a guard at the chin (jabs and uppercuts pass through up there, but aren't held).
   *   The pose's elbow reading was too wobbly to use.
   * The fist must be held nearly still (screen speed under chargeMaxSpeed).
   */
  chargeHoldS: 0.5, chargeKeepS: 1.5, chargeMaxSpeed: 35,
  hipBelow: 28, hipBelowMax: 60, elbowFlare: 8, hipMaxExtension: 0.5, earAbove: 32,
  /**
   * Live punch and push sensitivity ([ and ] in game): thresholds are divided by this. Tuned and tested
   * at 1; the default is set higher by preference (more misses caught, some more misfires).
   */
  punchSensitivity: 1.4,
  /** Fist punches: a hand only stops a punch (or counts as opening for a shield) once it is clearly open. */
  clearlyOpen: 0.8,
  /**
   * The wobble comes from the camera and grows with distance², so it is modelled as
   * noiseCoef × (distance to the body)². noiseCoef starts at noiseCoefStart and learns this camera's
   * floor from both fists while they are in guard (falling quickly, rising slowly).
   */
  noiseCoefStart: 0.006, noiseCoefDownRate: 1, noiseCoefUpRate: 0.15,
  /**
   * The wobble is measured from the jitter of each hand's raw distance reading (mean |second
   * difference| frame to frame), which smooth movement barely affects — unlike deviation from an
   * average, which counted a fist drifting back to guard as wobble and made punches ever stricter.
   * × jitterToWobble gives the same scale as the filtered reading's wobble (calibrated in simulation).
   */
  jitterToWobble: 0.21,
  /** The guard baseline follows a resting fist at this rate (1/s), and drops quickly if the fist is further back. */
  reachBaseRate: 0.7, reachBaseDropRate: 3,
  /** Fallback without 3D hand data: fire once the arm straightens past extendFireAbove; re-arm below extendRearmBelow. */
  extendFireAbove: 0.75, extendRearmBelow: 0.5,
  /** Shoulder half-width (m) and a full punch's reach (m), for aiming from a fist's position. */
  shoulderHalfM: 0.19, punchReachM: 0.45,
  /**
   * One Euro filters: smoothing at rest (Hz) and how quickly it loosens with speed — for hand
   * positions (view units), hand distance (m) and body distance (m, slower: bodies move slower).
   */
  posMinCutoff: 1, posBeta: 0.015, handDepthMinCutoff: 2, handDepthBeta: 4, bodyDepthMinCutoff: 0.8, bodyDepthBeta: 0.3,
  /** Shoulder width in metres doesn't change: it is learned at this rate (1/s) instead of re-read each frame. */
  shoulderSpanRate: 0.5,
  /** Arm labels are overridden when following hands frame to frame is this much (view units) more consistent. */
  relabelMargin: 12,
  /**
   * X block: forearms crossed — each wrist past the body's centre line by xCrossSw shoulder widths
   * (a cross punch moves only one), wrists no lower than xMaxWristSw below the shoulders — for xHoldS.
   */
  xCrossSw: 0.05, xMaxWristSw: 1.0, xHoldS: 0.08,
  /**
   * Leaning, stepping and ducking are the only way to move, so they are exaggerated: a head moved
   * a third of a shoulder width shifts the view ~23 units, about a quake dodge.
   */
  leanUnitsPerSw: 70, maxLean: 60,
  duckUnitsPerSw: 55, minDuck: -10, maxDuck: 35,
  /** Hand offset from the shoulder centre (in shoulder widths) × scale = view units. */
  handScaleX: 40, handScaleY: 32, handOffsetY: 20,
  /** Hands below this (view y) are resting, not attacking. */
  raisedAboveY: 40,
  /** Head smoothing rate, 1/s. Higher = snappier but jittery. */
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
  /**
   * Ultimate vs wall push: pushing both palms at the camera also makes them look further apart
   * (they get closer to it), so the ultimate is judged in 3D where known — the hands start at most
   * ultimateStartM apart (gathered together) and spread ultimateSpreadM — or on screen without 3D
   * data, starting at most ultimateStartSw shoulder widths apart. The push is checked first: both
   * palms' pushes reach twoPushShare of a single push's threshold at sensitivity 1.
   */
  ultimateStartM: 0.4, ultimateSpreadM: 0.3, ultimateStartSw: 0.9, twoPushShare: 0.8,
  /**
   * Palm push (fist-punch mode only; the open-hand punch style already uses opening hands): one hand
   * open, shoved toward the camera — it may open on the way. The other hand is a fist, or open but
   * held still (it came forward less than palmOtherStill as far; both pushing is a wall push). With
   * both hands open it needs the threshold at sensitivity 1 (like the wall push) and waits
   * palmBothOpenConfirmS instead, dropped if the other hand starts pushing too or a wall push goes off. The reach, averaged
   * over 3 frames, rises palmPushRise within palmPushWindowS (more with a wobbly reading:
   * palmNoise × wobble, up to palmPushCap; an open hand's reading wobbles more than a fist's),
   * leading the other hand like a fist punch. Like punches, thresholds divide by punchSensitivity.
   * It waits palmConfirmS (a second hand opening means shield or cast instead); the hand rests
   * palmRefractoryS afterwards. A fist punch whose hand opens while it is confirming becomes a push.
   */
  palmPushRise: 0.08, palmPushWindowS: 0.35, palmNoise: 8.5, palmPushCap: 0.15, palmConfirmS: 0.08, palmRefractoryS: 0.5, palmOtherStill: 0.5, palmBothOpenConfirmS: 0.15,
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
  /** bx: the hand's 3D sideways position (m), when known. */
  hist: { t: number; x: number; y: number; speed: number; size: number | null; ext: number | null; reach: number | null; bx: number | null }[];
  filters: { x: OneEuro; y: OneEuro; depth: OneEuro; bx: OneEuro; by: OneEuro };
  /** Filtered 3D palm position (m, relative to the shoulder centre). */
  body3: { x: number; y: number; z: number } | null;
  /**
   * How much this hand's reach reading wobbles at rest (m), from the jitter of its raw distance
   * (see TUNING.jitterToWobble), and the last few raw distances it is measured from.
   */
  reachDev: number;
  rawDepths: (number | null)[];
  /** When this hand last threw a punch (its reading takes a while to settle afterwards). */
  lastPunchT: number;
  /** Furthest reach since that punch fired (it re-arms once the fist comes back from here). */
  peakReach: number;
  /** When it got there: the next punch only counts forward movement after this. */
  peakT: number;
  /** When this hand's reach reading started (it needs a moment to settle before it can punch). */
  reachSince: number | null;
  /** Hand tracker palm − pose-wrist palm estimate, so switching between them doesn't jump. */
  armOffset: Vec2;
  /** Charged punch: since when it has been held in a charging pose, and when it was last fully charged. */
  chamberSince: number | null;
  chargedAt: number | null;
  /** When the hand can push again. */
  palmReadyAt: number;
  /** The shape the hand tracker saw last frame: its size (so its distance) is measured differently open and closed. */
  shapeOpen: boolean | null;
}

/** What a frame says about one hand. */
interface HandInput {
  pos: Vec2;
  /** The hand tracker's view (shape, size); null when following the pose wrist. */
  h: HandObs | null;
  size: number | null;
  ext: number | null;
  /** Filtered distance from the camera to the shoulders, m (null without 3D data). */
  bodyDist: number | null;
}

export interface InterpretState {
  head: Vec2 | null;
  l: Track | null;
  r: Track | null;
  lastT: number | null;
  pending: (Punch & { t: number })[];
  palmPending: (Palm & { t: number })[];
  /** When both hands were first seen open together (null if they aren't). */
  bothOpenAt: number | null;
  /** When both open hands have been still since, working toward the shield. */
  stillSince: number | null;
  shieldOn: boolean;
  castReadyAt: number;
  /** When the last two-hand cast went off (movement before it doesn't count toward the next). */
  lastCastT: number;
  crossedSince: number | null;
  /** Learned shoulder width (m) and the filtered distance to the shoulders (m). */
  shoulderSpan: number | null;
  bodyDist: OneEuro;
  /** This camera's reach wobble per metre² of distance. */
  noiseCoef: number;
  /** How fast the head (your body) is moving, view units/s, smoothed; and its recent history. */
  headSpeed: number;
  headSpeeds: { t: number; v: number }[];
}

export const initialState = (): InterpretState => ({
  head: null, l: null, r: null, lastT: null, pending: [], palmPending: [], bothOpenAt: null, stillSince: null, shieldOn: false,
  castReadyAt: -Infinity, lastCastT: -Infinity, crossedSince: null,
  shoulderSpan: null, bodyDist: new OneEuro(TUNING.bodyDepthMinCutoff, TUNING.bodyDepthBeta), noiseCoef: TUNING.noiseCoefStart,
  headSpeed: 0, headSpeeds: [],
});

const SIDES = ['l', 'r'] as const;
const other = (s: Side): Side => (s === 'l' ? 'r' : 'l');

const smooth = (prev: Vec2 | null, next: Vec2, k: number): Vec2 =>
  prev ? { x: prev.x + (next.x - prev.x) * k, y: prev.y + (next.y - prev.y) * k } : { ...next };

const snapshot = (t: Track | null): HandState | null =>
  t && {
    pos: { ...t.pos }, vel: { ...t.vel }, openness: t.openness, open: t.open, facing: t.facing,
    source: t.source, inView: t.inView, elbow: t.elbow && { ...t.elbow }, extension: t.extension,
    punchReady: t.armed, punchRise: t.reach === null ? null : reachRise(t, TUNING.quickWindowS),
    reach: t.reach, reachBase: t.reachBase, reachNoise: t.reach === null ? null : t.reachNoise,
    aimDir: t.aimDir && { ...t.aimDir }, charge: t.charge,
  };
const inPicture = (p: BodyPoint) => p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;

export function interpret(f: TrackingFrame, cal: Calibration, s: InterpretState): Intent {
  const dt = s.lastT === null ? 0 : Math.max(1e-3, f.t - s.lastT);
  s.lastT = f.t;
  const k = dt === 0 ? 1 : 1 - Math.exp(-TUNING.smoothing * dt);

  if (!f.head || !f.shoulderL || !f.shoulderR) {
    s.pending = [];
    s.palmPending = [];
    s.bothOpenAt = s.stillSince = null;
    s.shieldOn = false;
    s.crossedSince = null;
    return {
      present: false, head: s.head ? { ...s.head } : { x: 0, y: 0 }, hands: { l: null, r: null },
      shoulders: null, punches: [], palms: [], shield: false, xBlock: false, casts: [], face: null, bodyTilt: 0,
    };
  }
  const sw = dist(f.shoulderL, f.shoulderR) || cal.sw;
  const mid = { x: (f.shoulderL.x + f.shoulderR.x) / 2, y: (f.shoulderL.y + f.shoulderR.y) / 2 };
  const toView = (p: Vec2): Vec2 => ({
    x: ((p.x - mid.x) / sw) * TUNING.handScaleX,
    y: TUNING.handOffsetY + ((p.y - mid.y) / sw) * TUNING.handScaleY,
  });

  const headBefore = s.head && { ...s.head };
  s.head = smooth(s.head, {
    x: clamp(((f.head.x - cal.head.x) / sw) * TUNING.leanUnitsPerSw, -TUNING.maxLean, TUNING.maxLean),
    y: clamp(((f.head.y - cal.head.y) / sw) * TUNING.duckUnitsPerSw, TUNING.minDuck, TUNING.maxDuck),
  }, k);
  if (headBefore && dt > 0) {
    const v = Math.hypot(s.head.x - headBefore.x, s.head.y - headBefore.y) / dt;
    s.headSpeed = lerp(s.headSpeed, v, Math.min(1, dt * TUNING.headSpeedRate));
  }
  s.headSpeeds.push({ t: f.t, v: s.headSpeed });
  while (s.headSpeeds.length && f.t - s.headSpeeds[0].t > TUNING.leanMemoryS) s.headSpeeds.shift();
  // moving your body makes fists move too: attacks need more while it does (and just after)
  const recentSpeed = Math.max(...s.headSpeeds.map(x => x.v));
  const leanExtra = TUNING.leanPenalty * Math.max(0, recentSpeed - TUNING.leanFreeSpeed);
  const shoulders = { l: toView(f.shoulderL), r: toView(f.shoulderR) };

  // How far away the body is: learned shoulder width in metres over its apparent width, filtered.
  let bodyDist: number | null = null;
  if (f.body) {
    s.shoulderSpan = s.shoulderSpan === null ? f.body.span3 : lerp(s.shoulderSpan, f.body.span3, Math.min(1, dt * TUNING.shoulderSpanRate));
    bodyDist = s.bodyDist.filter((FOCAL_H * s.shoulderSpan) / f.body.span2, dt);
  }

  // Hands labelled by the arm they belong to; without a body, follow them from frame to frame.
  const obs = f.hands.slice(0, 2).map(h => ({ pos: toView(h.center), h }));
  const labelled = obs.length > 0 && obs.every(o => o.h.side) && new Set(obs.map(o => o.h.side)).size === obs.length;
  let picked = labelled ? bySide(obs.map(o => o.h.side!)) : assign(obs.map(o => o.pos), s.l, s.r, dt);
  // Two hands overlapping in the picture (a cross passing the other fist) can get each other's arm
  // label; keep following them frame to frame when that is clearly more consistent.
  if (labelled && obs.length === 2 && s.l && s.r && picked.l !== null && picked.r !== null) {
    const near = (t: Track, i: number) => dist({ x: t.pos.x + t.vel.x * dt, y: t.pos.y + t.vel.y * dt }, obs[i].pos);
    const byLabel = near(s.l, picked.l) + near(s.r, picked.r), swapped = near(s.l, picked.r) + near(s.r, picked.l);
    if (swapped + TUNING.relabelMargin < byLabel) picked = { l: picked.r, r: picked.l };
  }
  const opened: Side[] = [];
  for (const side of SIDES) {
    const arm = f.arms[side], i = picked[side];
    const ext = arm?.extension ?? null;
    // where the pose wrist puts the palm (just past the wrist, along the forearm)
    const armPalm = arm && (() => {
      const w = toView(arm.wrist), e = toView(arm.elbow);
      return { x: w.x + (w.x - e.x) * TUNING.palmBeyondWrist, y: w.y + (w.y - e.y) * TUNING.palmBeyondWrist };
    })();
    if (i !== null) {
      const o = obs[i];
      const r = updateTrack(s[side], { pos: o.pos, h: o.h, size: o.h.size / sw, ext, bodyDist }, f.t, dt, k);
      s[side] = r.track;
      r.track.source = 'hand';
      r.track.inView = true;
      if (armPalm) {
        const off = { x: o.pos.x - armPalm.x, y: o.pos.y - armPalm.y };
        r.track.armOffset = { x: lerp(r.track.armOffset.x, off.x, 0.3), y: lerp(r.track.armOffset.y, off.y, 0.3) };
      }
      if (r.opened) opened.push(side);
    } else if (arm && armPalm) {
      // The hand tracker lost this hand (blur, edge of frame): follow the pose wrist instead,
      // keeping the last offset between the two (fading) so the hand doesn't jump.
      const prev = s[side], fade = Math.exp(-dt / 1.0);
      const offset = prev ? { x: prev.armOffset.x * fade, y: prev.armOffset.y * fade } : { x: 0, y: 0 };
      const tr = updateTrack(prev, { pos: { x: armPalm.x + offset.x, y: armPalm.y + offset.y }, h: null, size: null, ext, bodyDist }, f.t, dt, k).track;
      tr.armOffset = offset;
      tr.inView = arm.wrist.vis >= TUNING.minWristVis && inPicture(arm.wrist);
      tr.source = tr.inView ? 'arm' : 'estimate';
      s[side] = tr;
    } else if (s[side] && f.t - s[side]!.lastSeen > TUNING.lostGraceS) {
      s[side] = null;
    }
    const tr = s[side];
    if (tr) {
      tr.elbow = arm ? toView(arm.elbow) : null;
      tr.aimDir = aimFromBody(tr, side) ?? aimDir(arm?.reach ?? null);
    }
  }

  // Learn this camera's wobble floor from fists resting in guard, and predict each fist's wobble.
  if (bodyDist !== null) {
    // only fists genuinely at rest: back in guard, not moving across the screen, not just after a punch
    const resting = (t: Track) => t.armed && t.reach !== null && t.reachBase !== null && t.source === 'hand'
      && t.reach - t.reachBase < TUNING.reachRearm && Math.hypot(t.vel.x, t.vel.y) < 30 && f.t - t.lastPunchT > 0.8;
    const samples = SIDES.map(side => s[side]).filter((t): t is Track => !!t && resting(t))
      .map(t => t.reachDev / (bodyDist * bodyDist));
    if (samples.length) {
      const sample = samples.reduce((a, b) => a + b, 0) / samples.length;
      const rate = sample < s.noiseCoef ? TUNING.noiseCoefDownRate : TUNING.noiseCoefUpRate;
      s.noiseCoef = lerp(s.noiseCoef, sample, Math.min(1, dt * rate));
    }
    for (const side of SIDES) { const t = s[side]; if (t) t.reachNoise = s.noiseCoef * bodyDist * bodyDist; }
  }

  // X block: forearms crossed in front of the chest (from the pose, which tracks fists well).
  const al = f.arms.l, ar = f.arms.r;
  const crossed = al && ar
    ? al.wrist.x > mid.x + TUNING.xCrossSw * sw && ar.wrist.x < mid.x - TUNING.xCrossSw * sw
      && Math.max(al.wrist.y, ar.wrist.y) < mid.y + TUNING.xMaxWristSw * sw
    : !!s.l && !!s.r && s.l.pos.x > s.r.pos.x + 4 && Math.max(s.l.pos.y, s.r.pos.y) < TUNING.raisedAboveY;
  if (!crossed) s.crossedSince = null;
  else if (s.crossedSince === null) s.crossedSince = f.t;
  const xBlock = s.crossedSince !== null && f.t - s.crossedSince >= TUNING.xHoldS;

  const extendMode = TUNING.punchTrigger === 'extend';
  const confirmS = extendMode ? TUNING.extendConfirmS : TUNING.punchConfirmS;
  if (xBlock) {
    s.pending = [];
    s.palmPending = [];
  } else if (extendMode) {
    // A quick jolt of a fist toward the camera (more than the other fist moved); re-arms on a short pull-back.
    for (const side of SIDES) {
      const tr = s[side], o = s[other(side)];
      if (!tr) continue;
      // (a fist at the hip may be below the picture: charging still counts it)
      updateCharge(tr, side, shoulders[side], f.t, recentSpeed > TUNING.leanFreeSpeed);
      if (tr.source === 'estimate') continue;
      let fire = false, jolt = false, shove = false;
      if (tr.reach !== null) {
        const need = fistThresholds(tr.reachNoise ?? 0, leanExtra);
        if (!tr.armed) {
          if (tr.reach > tr.peakReach) { tr.peakReach = tr.reach; tr.peakT = f.t; }
          if (tr.reach <= tr.peakReach - Math.max(TUNING.rearmDrop, TUNING.noiseRearm * (tr.reachNoise ?? 0))) tr.armed = true;
        }
        const rise = reachRise(tr, TUNING.quickWindowS);
        const otherRise = o && o.reach !== null ? reachRise(o, TUNING.quickWindowS) : 0;
        const settled = tr.reachSince !== null && f.t - tr.reachSince >= TUNING.reachWarmupS;
        // while the body moves, the lean allowance raises the bar, but a sharp jolt still counts
        const sharp = leanExtra > 0 && reachRise(tr, TUNING.sharpWindowS) >= TUNING.sharpJolt;
        jolt = settled && tr.armed && f.t - tr.lastPunchT >= TUNING.refireS && (rise >= need.rise || sharp) && rise - otherRise >= need.lead;
        fire = jolt && tr.openness < TUNING.clearlyOpen;
        // only movement since this hand's last punch or push counts (not the tail of that one)
        const push = palmPushRise(tr.hist, TUNING.palmPushWindowS, tr.lastPunchT);
        // with the other hand open too (shield territory) a stray push costs more: no sensitivity boost
        const pushNeed = pushThreshold(tr.reachNoise ?? 0) * (o?.open ? TUNING.punchSensitivity : 1) + leanExtra;
        shove = settled && tr.armed && f.t - tr.lastPunchT >= TUNING.refireS && push >= pushNeed && push - otherRise >= need.lead;
        if (fire) { tr.peakReach = tr.reach; tr.peakT = f.t; }
      } else if (tr.extension !== null) {
        // no 3D hand data: fall back to the arm straightening
        if (tr.extension < TUNING.extendRearmBelow) tr.armed = true;
        fire = tr.armed && tr.extension >= TUNING.extendFireAbove && extensionRise(tr) >= TUNING.punchExtendRise;
      }
      if (fire && tr.openness < TUNING.clearlyOpen && tr.pos.y < TUNING.raisedAboveY) {
        tr.armed = false;
        tr.lastPunchT = f.t;
        s.pending.push({ hand: side, at: { ...tr.pos }, shoulder: { ...shoulders[side] }, dir: null, t: f.t, charged: tr.charge >= 1 });
        tr.charge = 0;
        tr.chargedAt = null;
        continue;
      }
      // One open palm shoved forward, the other hand a fist or an open palm held still: a pillar.
      if (shove && tr.open && (!o || !o.open || heldStill(o, tr)) && f.t >= tr.palmReadyAt && tr.pos.y < TUNING.raisedAboveY && !s.palmPending.some(p => p.hand === side)) {
        startPush(s, side, f.t, shoulders[side]);
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
    // a fist that reads half-open mid-punch still counts; only clearly open hands mean shield
    const opens = (t: Track | null) => !!t && (extendMode ? t.openness >= TUNING.clearlyOpen : t.open);
    if (!tr || opens(s[other(p.hand)])) return false;
    if (extendMode && opens(tr)) {
      // the fist opened as it went out: that's a palm push
      if (f.t >= tr.palmReadyAt && !s.palmPending.some(q => q.hand === p.hand)) startPush(s, p.hand, p.t, p.shoulder);
      return false;
    }
    if (f.t - p.t < confirmS) return true;
    punches.push({ hand: p.hand, at: { ...tr.pos }, shoulder: p.shoulder, dir: tr.aimDir && { ...tr.aimDir }, charged: p.charged });
    return false;
  });
  // Palm moves wait too: a second hand opening means shield or a two-hand cast; a closing hand, never mind.
  const palms: Palm[] = [];
  s.palmPending = s.palmPending.filter(p => {
    const tr = s[p.hand], o = s[other(p.hand)];
    // the other hand opening and pushing too makes it a wall push instead
    if (!tr || !tr.open || (o?.open && !heldStill(o, tr))) return false;
    if (f.t - p.t < (o?.open ? TUNING.palmBothOpenConfirmS : TUNING.palmConfirmS)) return true;
    palms.push({ ...p, at: { ...tr.pos }, dir: tr.aimDir && { ...tr.aimDir } });
    return false;
  });

  // Two open hands: a quick sweep up is a fire wall, a quick spread is the ultimate, held still is the shield.
  const casts: Cast[] = [];
  const l = s.l, r = s.r;
  const bothOpen = !xBlock && !!l && !!r && l.inView && r.inView && l.open && r.open;
  if (!bothOpen) {
    s.bothOpenAt = s.stillSince = null;
    s.shieldOn = false;
  } else {
    if (s.bothOpenAt === null) s.bothOpenAt = f.t;
    const kind = f.t >= s.castReadyAt ? twoHandGesture(l, r, Math.max(s.bothOpenAt, f.t - TUNING.castWindowS), s.lastCastT) : null;
    if (kind) {
      casts.push({ kind, at: { x: (l.pos.x + r.pos.x) / 2, y: (l.pos.y + r.pos.y) / 2 } });
      if (kind === 'push') s.palmPending = []; // it was both palms, not one
      s.castReadyAt = f.t + TUNING.castRefractoryS;
      s.lastCastT = f.t;
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
    present: true, head: { ...s.head }, hands: { l: snapshot(s.l), r: snapshot(s.r) }, shoulders, punches, palms, shield, xBlock, casts,
    face: f.face, bodyTilt: Math.atan2(f.shoulderR.y - f.shoulderL.y, f.shoulderR.x - f.shoulderL.x),
  };
}

/** The other open hand isn't pushing too: it came forward much less than the pushing hand did. */
function heldStill(o: Track, pushing: Track): boolean {
  const w = TUNING.palmPushWindowS;
  return palmPushRise(o.hist, w, o.lastPunchT) < TUNING.palmOtherStill * palmPushRise(pushing.hist, w);
}

/** Is this fist in a charging pose: down at the hip (elbow flared or arm bent), or raised up by the ear? */
export function chargePose(tr: HandState, side: Side, shoulder: Vec2): 'hip' | 'ear' | null {
  if (tr.openness >= TUNING.clearlyOpen) return null;
  const out = side === 'l' ? -1 : 1, below = tr.pos.y - shoulder.y;
  if (tr.source === 'hand' && -below >= TUNING.earAbove) return 'ear';
  const flared = !!tr.elbow && (tr.elbow.x - tr.pos.x) * out >= TUNING.elbowFlare;
  if (below >= TUNING.hipBelow && below <= TUNING.hipBelowMax && (flared || (tr.extension ?? 1) < TUNING.hipMaxExtension)) return 'hip';
  return null;
}

/**
 * Charged punch: fills `charge` 0 → 1 while the fist is held still in a charging pose (with the
 * body steady); a full charge lasts chargeKeepS after it leaves the pose.
 */
function updateCharge(tr: Track, side: Side, shoulder: Vec2, t: number, bodyMoving: boolean): void {
  const fist = tr.openness < TUNING.clearlyOpen;
  const held = !bodyMoving && Math.hypot(tr.vel.x, tr.vel.y) < TUNING.chargeMaxSpeed && chargePose(tr, side, shoulder) !== null;
  if (held) {
    tr.chamberSince ??= t;
    // a full charge stays full; otherwise it fills while held
    tr.charge = tr.chargedAt !== null ? 1 : Math.min(1, (t - tr.chamberSince) / TUNING.chargeHoldS);
    if (tr.charge >= 1) tr.chargedAt = t;
    return;
  }
  tr.chamberSince = null;
  // left the pose: a full charge is kept a moment (to punch with); a partial one is lost
  const kept = tr.chargedAt !== null && t - tr.chargedAt <= TUNING.chargeKeepS && fist;
  if (kept) tr.charge = 1;
  else { tr.charge = 0; tr.chargedAt = null; }
}

/** Queue a palm push from this hand (it confirms after palmConfirmS). */
function startPush(s: InterpretState, side: Side, t: number, shoulder: Vec2): void {
  const tr = s[side]!;
  tr.armed = false;
  tr.lastPunchT = t;
  if (tr.reach !== null) { tr.peakReach = tr.reach; tr.peakT = t; }
  tr.palmReadyAt = t + TUNING.palmRefractoryS;
  s.palmPending.push({ kind: 'push', hand: side, at: { ...tr.pos }, shoulder: { ...shoulder }, dir: null, t });
}

/** The shove (m) a palm push needs, given the reading's wobble: capped, and scaled by sensitivity. */
export function pushThreshold(noise: number): number {
  return Math.min(TUNING.palmPushCap, Math.max(TUNING.palmPushRise, TUNING.palmNoise * noise)) / TUNING.punchSensitivity;
}

/**
 * How the two open hands moved: both shoved toward the camera → wall push (checked first);
 * mostly up since `from` → wall; gathered together, then flung apart → ultimate.
 */
function twoHandGesture(l: Track, r: Track, from: number, lastCastT: number): CastKind | null {
  // palm pushes may start just before both hands read open, so they look back their own window
  // (not scaled by punch sensitivity: stealing a wall or the ultimate costs more than a stray punch)
  const need = TUNING.twoPushShare * TUNING.punchSensitivity * pushThreshold(Math.max(l.reachNoise ?? 0, r.reachNoise ?? 0));
  if (Math.min(palmPushRise(l.hist, TUNING.palmPushWindowS, lastCastT), palmPushRise(r.hist, TUNING.palmPushWindowS, lastCastT)) >= need) return 'push';
  const l0 = l.hist.find(h => h.t >= from), r0 = r.hist.find(h => h.t >= from);
  if (!l0 || !r0) return null;
  const rise = Math.min(l0.y - l.pos.y, r0.y - r.pos.y);
  const screenSpread = Math.hypot(l.pos.x - r.pos.x, l.pos.y - r.pos.y) - Math.hypot(l0.x - r0.x, l0.y - r0.y);
  let ultimate: boolean;
  if (l0.bx !== null && r0.bx !== null && l.body3 && r.body3) {
    // in metres, unaffected by how close the hands are to the camera
    const start = Math.abs(r0.bx - l0.bx);
    ultimate = start <= TUNING.ultimateStartM && Math.abs(r.body3.x - l.body3.x) - start >= TUNING.ultimateSpreadM;
  } else {
    ultimate = Math.abs(r0.x - l0.x) <= TUNING.ultimateStartSw * TUNING.handScaleX && screenSpread >= TUNING.ultimateSpread;
  }
  if (rise >= TUNING.wallRise && rise > screenSpread) return 'wall';
  if (ultimate && screenSpread > rise) return 'ultimate';
  return null;
}

/**
 * Aim from the fist's 3D position relative to its own shoulder: its sideways/vertical offset over
 * how far forward it is (at least a full punch's reach, so a fist resting in guard previews where a
 * straight punch from there would land). A straight jab goes straight; a cross goes across.
 */
function aimFromBody(tr: Track, side: Side): Vec2 | null {
  const b = tr.body3;
  if (!b) return null;
  const shoulderX = side === 'l' ? -TUNING.shoulderHalfM : TUNING.shoulderHalfM, forward = Math.max(b.z, TUNING.punchReachM);
  return { x: (b.x - shoulderX) / forward, y: b.y / forward };
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
 * Move a track to a new position. `in.h` (the hand tracker's view of the hand) updates its shape
 * and 3D position; without it — following the pose wrist — the last known shape is kept.
 */
function updateTrack(tr: Track | null, input: HandInput, t: number, dt: number, k: number): { track: Track; opened: boolean } {
  const { pos, h, size, ext, bodyDist } = input;
  const b3 = h?.body3 ?? null, depth = h?.depth ?? null;
  // how far in front of the shoulders: filtered body distance − filtered hand distance
  const reachNow = (t: Track) => (depth !== null && bodyDist !== null ? bodyDist - t.filters.depth.filter(depth, dt) : null);
  if (!tr) {
    // A hand that appears already open doesn't count as opening.
    const open = h ? h.open : 0;
    const filters = {
      x: new OneEuro(TUNING.posMinCutoff, TUNING.posBeta), y: new OneEuro(TUNING.posMinCutoff, TUNING.posBeta),
      depth: new OneEuro(TUNING.handDepthMinCutoff, TUNING.handDepthBeta),
      bx: new OneEuro(TUNING.handDepthMinCutoff, TUNING.handDepthBeta), by: new OneEuro(TUNING.handDepthMinCutoff, TUNING.handDepthBeta),
    };
    filters.x.filter(pos.x, 0); filters.y.filter(pos.y, 0);
    if (b3) { filters.bx.filter(b3.x, 0); filters.by.filter(b3.y, 0); }
    const track: Track = {
      pos: { ...pos }, vel: { x: 0, y: 0 }, openness: open, open: open >= 0.5, facing: h ? h.facing : 1,
      source: 'hand', inView: true, elbow: null, extension: ext, punchReady: true, armed: true, lastSeen: t,
      reach: null, reachBase: null, reachNoise: 0.02, reachDev: 0.02, rawDepths: [depth], lastPunchT: -Infinity, peakReach: -Infinity, peakT: -Infinity,
      reachSince: null,
      punchRise: null,
      aimDir: null, body3: b3 && { ...b3 },
      armOffset: { x: 0, y: 0 }, filters, palmReadyAt: -Infinity, shapeOpen: h ? h.open >= 0.5 : null,
      chamberSince: null, chargedAt: null, charge: 0,
      hist: [],
    };
    track.reach = reachNow(track);
    track.reachBase = track.reach;
    if (track.reach !== null) track.reachSince = t;
    if (track.body3 && track.reach !== null) track.body3.z = track.reach;
    track.hist.push({ t, x: pos.x, y: pos.y, speed: 0, size, ext, reach: track.reach, bx: track.body3?.x ?? null });
    return { track, opened: false };
  }
  // Opening or closing the hand switches how its distance is measured, which jumps the reading:
  // restart the distance filter and shift the reach history by the jump, so the jump itself isn't
  // read as movement.
  const shapeOpen = h ? h.open >= 0.5 : tr.shapeOpen;
  const reshaped = tr.shapeOpen !== null && shapeOpen !== tr.shapeOpen;
  if (reshaped) tr.filters.depth = new OneEuro(TUNING.handDepthMinCutoff, TUNING.handDepthBeta);
  tr.shapeOpen = shapeOpen;
  const prev = tr.pos;
  tr.pos = { x: tr.filters.x.filter(pos.x, dt), y: tr.filters.y.filter(pos.y, dt) };
  if (dt > 0) {
    const kv = 1 - Math.exp(-12 * dt);
    tr.vel = { x: lerp(tr.vel.x, (tr.pos.x - prev.x) / dt, kv), y: lerp(tr.vel.y, (tr.pos.y - prev.y) / dt, kv) };
  }
  tr.extension = ext === null ? null : tr.extension === null ? ext : lerp(tr.extension, ext, k);
  const reach = reachNow(tr);
  if (reshaped && reach !== null && tr.reach !== null) {
    // reach is body − hand distance, so the new way reads (old-way distance − new-way distance) further out
    const jump = reach - tr.reach;
    for (const x of tr.hist) if (x.reach !== null) x.reach += jump;
    if (tr.reachBase !== null) tr.reachBase += jump;
    tr.peakReach += jump;
    tr.rawDepths = [];
  }
  // the wobble: jitter (second difference) of the raw distance — smooth movement barely shows in it
  if (h) {
    tr.rawDepths = [...tr.rawDepths, depth].slice(-3);
    const [a, b, c] = tr.rawDepths;
    if (tr.rawDepths.length === 3 && a != null && b != null && c != null) {
      tr.reachDev = lerp(tr.reachDev, TUNING.jitterToWobble * Math.abs(c - 2 * b + a), Math.min(1, dt * 1));
    }
  }
  if (reach !== null && b3) {
    tr.reachSince ??= t;
    tr.reach = reach;
    tr.body3 = { x: tr.filters.bx.filter(b3.x, dt), y: tr.filters.by.filter(b3.y, dt), z: reach };
    updateReachBase(tr, dt);
  }
  let opened = false;
  if (h) {
    tr.openness = lerp(tr.openness, h.open, k);
    tr.facing = lerp(tr.facing, h.facing, k);
    if (!tr.open && tr.openness > TUNING.openAbove) { tr.open = true; opened = true; }
    else if (tr.open && tr.openness < TUNING.fistBelow) tr.open = false;
  }
  tr.lastSeen = t;
  tr.hist.push({ t, x: tr.pos.x, y: tr.pos.y, speed: Math.hypot(tr.vel.x, tr.vel.y), size, ext: tr.extension, reach: reach !== null ? tr.reach : null, bx: reach !== null && b3 ? tr.body3!.x : null });
  const keepS = Math.max(TUNING.punchWindowS, TUNING.castWindowS, TUNING.quickWindowS, TUNING.palmPushWindowS);
  while (tr.hist.length && t - tr.hist[0].t > keepS) tr.hist.shift();
  return { track: tr, opened };
}

/**
 * The guard baseline: where this fist rests. It follows a fist that is back in guard and still,
 * and drops quickly if the fist sits further back than it thought. Also tracks the reading's
 * wobble floor: it drops quickly whenever the fist is still, and creeps up only slowly.
 */
function updateReachBase(tr: Track, dt: number): void {
  if (tr.reach === null) return;
  if (tr.reachBase === null) { tr.reachBase = tr.reach; return; }
  const still = Math.hypot(tr.vel.x, tr.vel.y) < 40;
  if (tr.reach < tr.reachBase) tr.reachBase = lerp(tr.reachBase, tr.reach, Math.min(1, dt * TUNING.reachBaseDropRate));
  else if (tr.armed && still && tr.reach - tr.reachBase < TUNING.reachRearm * 1.5) {
    tr.reachBase = lerp(tr.reachBase, tr.reach, Math.min(1, dt * TUNING.reachBaseRate));
  }
}

/**
 * The jolt (m) a fist punch needs, given the reading's wobble: capped, and scaled by sensitivity;
 * plus `leanExtra` while your body is moving fast.
 */
export function fistThresholds(noise: number, leanExtra = 0): { rise: number; lead: number } {
  const t = TUNING, k = t.punchSensitivity;
  return {
    rise: Math.min(t.quickRiseCap, Math.max(t.quickRise, t.noiseQuick * noise)) / k + leanExtra,
    lead: Math.min(t.quickLeadCap, Math.max(t.quickLead, t.noiseLead * noise)) / k,
  };
}

/**
 * How far the fist has come forward within the last `windowS`: its reach now minus its lowest in
 * that time, metres — counting only since its last punch peaked, so pulling back from one punch
 * never counts toward the next.
 */
function reachRise(tr: Track, windowS: number): number {
  if (tr.reach === null) return 0;
  let low = tr.reach;
  const now = tr.hist[tr.hist.length - 1]?.t ?? 0;
  for (const h of tr.hist) {
    if (now - h.t <= windowS && h.t >= tr.peakT && h.reach !== null) low = Math.min(low, h.reach);
  }
  return tr.reach - low;
}

/**
 * Like reachRise, but on the reach averaged over the last few frames and over a longer window: an
 * open hand's reading wobbles more, and a palm push is a bigger, slightly slower move.
 */
export function palmPushRise(hist: { t: number; reach: number | null }[], windowS: number, since = -Infinity, n = 3): number {
  // frames where the hand tracker lost the (blurred) hand have no reading: skip them
  const now = hist[hist.length - 1]?.t ?? 0;
  const r = hist.filter(h => now - h.t <= windowS && h.t > since && h.reach !== null).map(h => h.reach!);
  if (r.length < n) return 0;
  const avg: number[] = [];
  for (let i = n - 1; i < r.length; i++) avg.push((r[i] + r[i - 1] + r[i - 2]) / 3);
  return avg[avg.length - 1] - Math.min(...avg);
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
