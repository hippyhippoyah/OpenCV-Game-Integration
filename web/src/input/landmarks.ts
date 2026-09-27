import type { HandObs, TrackingFrame } from './types';
import { dist, type Vec2 } from '../math';

/** The part of MediaPipe's NormalizedLandmark we use. */
export interface Landmark { x: number; y: number; visibility?: number }

const NOSE = 0, L_SHOULDER = 11, R_SHOULDER = 12;
const PALM = [0, 5, 9, 13, 17], WRIST = 0, MIDDLE_KNUCKLE = 9;
const MIN_VISIBILITY = 0.5;

const mirror = (p: Vec2): Vec2 => ({ x: 1 - p.x, y: p.y });
const visible = (p: Landmark | undefined): p is Landmark => !!p && (p.visibility ?? 1) >= MIN_VISIBILITY;

/** Raw MediaPipe results → mirrored TrackingFrame (moving right moves right on screen). */
export function toFrame(t: number, hands: Landmark[][], pose: Landmark[] | undefined): TrackingFrame {
  const head = pose && visible(pose[NOSE]) ? mirror(pose[NOSE]) : null;
  let shoulderL: Vec2 | null = null, shoulderR: Vec2 | null = null;
  if (pose && visible(pose[L_SHOULDER]) && visible(pose[R_SHOULDER])) {
    [shoulderL, shoulderR] = [mirror(pose[L_SHOULDER]), mirror(pose[R_SHOULDER])].sort((a, b) => a.x - b.x);
  }
  return { t, head, shoulderL, shoulderR, hands: hands.map(handObs) };
}

function handObs(lm: Landmark[]): HandObs {
  const c = { x: 0, y: 0 };
  for (const i of PALM) { c.x += lm[i].x / PALM.length; c.y += lm[i].y / PALM.length; }
  return { center: mirror(c), size: dist(lm[WRIST], lm[MIDDLE_KNUCKLE]) };
}
