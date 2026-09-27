import type { HandObs, TrackingFrame } from './types';
import { clamp, dist, type Vec2 } from '../math';

/** The part of MediaPipe's landmark types we use. */
export interface Landmark { x: number; y: number; z?: number; visibility?: number }

const NOSE = 0, L_SHOULDER = 11, R_SHOULDER = 12;
const PALM = [0, 5, 9, 13, 17], WRIST = 0, INDEX_KNUCKLE = 5, MIDDLE_KNUCKLE = 9, PINKY_KNUCKLE = 17;
/** Knuckle, two middle joints and tip of each finger (the thumb is left out). */
const FINGERS = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const MIN_VISIBILITY = 0.5;
/** Finger straightness (knuckle→tip ÷ total bone length) that counts as fully curled / fully straight. */
const CURLED = 0.55, STRAIGHT = 0.9;

const mirror = (p: Vec2): Vec2 => ({ x: 1 - p.x, y: p.y });
const visible = (p: Landmark | undefined): p is Landmark => !!p && (p.visibility ?? 1) >= MIN_VISIBILITY;
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

/**
 * Raw MediaPipe results → mirrored TrackingFrame (moving right moves right on screen).
 * `handsWorld` are MediaPipe's 3D hand landmarks in metres; hand shape comes from them when present.
 */
export function toFrame(t: number, hands: Landmark[][], pose: Landmark[] | undefined, handsWorld: Landmark[][] = []): TrackingFrame {
  const head = pose && visible(pose[NOSE]) ? mirror(pose[NOSE]) : null;
  let shoulderL: Vec2 | null = null, shoulderR: Vec2 | null = null;
  if (pose && visible(pose[L_SHOULDER]) && visible(pose[R_SHOULDER])) {
    [shoulderL, shoulderR] = [mirror(pose[L_SHOULDER]), mirror(pose[R_SHOULDER])].sort((a, b) => a.x - b.x);
  }
  return { t, head, shoulderL, shoulderR, hands: hands.map((lm, i) => handObs(lm, handsWorld[i] ?? lm)) };
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
