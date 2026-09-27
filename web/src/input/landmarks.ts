import type { ArmObs, BodyPoint, HandObs, Side, TrackingFrame } from './types';
import { clamp, dist, type Vec2 } from '../math';

/** The part of MediaPipe's landmark types we use. */
export interface Landmark { x: number; y: number; z?: number; visibility?: number }

const NOSE = 0, EYE_L = 2, EYE_R = 5, EAR_L = 7, EAR_R = 8, L_SHOULDER = 11, R_SHOULDER = 12;
/** Pose points per arm, by the person's own left/right — which the mirrored view shows on that side. */
const ARM = { l: { shoulder: 11, elbow: 13, wrist: 15 }, r: { shoulder: 12, elbow: 14, wrist: 16 } } as const;
/** Elbow angle (degrees) that counts as fully bent / fully straight. */
const BENT_DEG = 70, STRAIGHT_DEG = 165;
const PALM = [0, 5, 9, 13, 17], WRIST = 0, INDEX_KNUCKLE = 5, MIDDLE_KNUCKLE = 9, PINKY_KNUCKLE = 17;
/** Knuckle, two middle joints and tip of each finger (the thumb is left out). */
const FINGERS = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const MIN_VISIBILITY = 0.5;
/** Finger straightness (knuckle→tip ÷ total bone length) that counts as fully curled / fully straight. */
const CURLED = 0.55, STRAIGHT = 0.9;

const mirror = (p: Vec2): Vec2 => ({ x: 1 - p.x, y: p.y });
const visible = (p: Landmark | undefined): p is Landmark => !!p && (p.visibility ?? 1) >= MIN_VISIBILITY;
const bodyPoint = (p: Landmark): BodyPoint => ({ ...mirror(p), vis: p.visibility ?? 1 });
const d3 = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));

/** 0 = fist … 1 = flat open hand. Uses 3D finger straightness, so it works whichever way the hand points. */
export function openness(lm: Landmark[]): number {
  let sum = 0;
  for (const [k, p, d, t] of FINGERS) {
    const bones = d3(lm[k], lm[p]) + d3(lm[p], lm[d]) + d3(lm[d], lm[t]);
    const straight = bones > 0 ? d3(lm[k], lm[t]) / bones : 0;
    sum += clamp((straight - CURLED) / (STRAIGHT - CURLED), 0, 1);
  }
  return sum / FINGERS.length;
}

/** 1 = palm (or back of the hand) faces the camera; 0 = edge-on, e.g. palms facing each other. */
export function palmFacing(lm: Landmark[]): number {
  const w = lm[WRIST], a = lm[INDEX_KNUCKLE], b = lm[PINKY_KNUCKLE];
  const ax = a.x - w.x, ay = a.y - w.y, az = (a.z ?? 0) - (w.z ?? 0);
  const bx = b.x - w.x, by = b.y - w.y, bz = (b.z ?? 0) - (w.z ?? 0);
  const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
  const len = Math.hypot(nx, ny, nz);
  return len > 0 ? Math.abs(nz) / len : 0;
}

/** How straight the elbow is in 3D: 0 = fully bent … 1 = straight. */
export function armExtension(w: Landmark[], side: Side): number {
  const a = ARM[side], s = w[a.shoulder], e = w[a.elbow], r = w[a.wrist];
  const v1 = [s.x - e.x, s.y - e.y, (s.z ?? 0) - (e.z ?? 0)], v2 = [r.x - e.x, r.y - e.y, (r.z ?? 0) - (e.z ?? 0)];
  const len = Math.hypot(...v1) * Math.hypot(...v2);
  if (len === 0) return 0;
  const deg = (Math.acos(clamp((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / len, -1, 1)) * 180) / Math.PI;
  return clamp((deg - BENT_DEG) / (STRAIGHT_DEG - BENT_DEG), 0, 1);
}

/**
 * Raw MediaPipe results → mirrored TrackingFrame (moving right moves right on screen).
 * `handsWorld` / `poseWorld` are MediaPipe's 3D landmarks in metres; hand shape and arm
 * straightness come from them when present.
 */
export function toFrame(t: number, hands: Landmark[][], pose: Landmark[] | undefined, handsWorld: Landmark[][] = [], poseWorld?: Landmark[]): TrackingFrame {
  const head = pose && visible(pose[NOSE]) ? mirror(pose[NOSE]) : null;
  let shoulderL: Vec2 | null = null, shoulderR: Vec2 | null = null;
  if (pose && visible(pose[L_SHOULDER]) && visible(pose[R_SHOULDER])) {
    [shoulderL, shoulderR] = [mirror(pose[L_SHOULDER]), mirror(pose[R_SHOULDER])].sort((a, b) => a.x - b.x);
  }
  const armOf = (side: Side): ArmObs | null => {
    const a = ARM[side];
    if (!pose || !visible(pose[a.shoulder])) return null;
    return {
      shoulder: bodyPoint(pose[a.shoulder]),
      elbow: bodyPoint(pose[a.elbow]),
      wrist: bodyPoint(pose[a.wrist]),
      extension: poseWorld ? armExtension(poseWorld, side) : null,
    };
  };
  const arms = { l: armOf('l'), r: armOf('r') };
  const handObsList = hands.map((lm, i) => handObs(lm, handsWorld[i] ?? lm));
  matchHandsToArms(handObsList, arms);
  return { t, head, shoulderL, shoulderR, hands: handObsList, arms, face: pose ? faceOf(pose) : null };
}

/** Label each hand with the arm whose pose wrist is nearest (both hands matched jointly). */
function matchHandsToArms(hands: HandObs[], arms: Record<Side, ArmObs | null>): void {
  const wrists = (['l', 'r'] as const).flatMap(side => (arms[side] ? [{ side, p: arms[side]!.wrist }] : []));
  if (!wrists.length || !hands.length) return;
  if (hands.length >= 2 && wrists.length === 2) {
    const [a, b] = hands, [wl, wr] = wrists;
    const keep = dist(a.center, wl.p) + dist(b.center, wr.p), swap = dist(a.center, wr.p) + dist(b.center, wl.p);
    [a.side, b.side] = keep <= swap ? [wl.side, wr.side] : [wr.side, wl.side];
    return;
  }
  if (wrists.length === 1) {
    // one arm known: only the hand nearest its wrist can be labelled with confidence
    const w = wrists[0], nearest = hands.reduce((best, h) => (dist(h.center, w.p) < dist(best.center, w.p) ? h : best));
    nearest.side = w.side;
    return;
  }
  const h = hands[0];
  h.side = dist(h.center, wrists[0].p) <= dist(h.center, wrists[1].p) ? wrists[0].side : wrists[1].side;
}

/** Head turn and tilt from nose, eyes and ears; null unless all are clearly visible. */
function faceOf(pose: Landmark[]): TrackingFrame['face'] {
  const ids = [NOSE, EYE_L, EYE_R, EAR_L, EAR_R];
  if (!ids.every(i => visible(pose[i]))) return null;
  const [nose, e1, e2, a1, a2] = ids.map(i => mirror(pose[i]));
  const [eyeL, eyeR] = [e1, e2].sort((a, b) => a.x - b.x), [earL, earR] = [a1, a2].sort((a, b) => a.x - b.x);
  const width = earR.x - earL.x;
  if (width <= 0) return null;
  return { yaw: (nose.x - (earL.x + earR.x) / 2) / width, roll: Math.atan2(eyeR.y - eyeL.y, eyeR.x - eyeL.x) };
}

function handObs(lm: Landmark[], world: Landmark[]): HandObs {
  const c = { x: 0, y: 0 };
  for (const i of PALM) { c.x += lm[i].x / PALM.length; c.y += lm[i].y / PALM.length; }
  return {
    center: mirror(c),
    size: dist(lm[WRIST], lm[MIDDLE_KNUCKLE]),
    open: openness(world),
    facing: palmFacing(world),
  };
}
