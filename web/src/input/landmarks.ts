import type { ArmObs, BodyPoint, HandObs, Side, TrackingFrame } from './types';
import { clamp, dist, type Vec2, type Vec3 } from '../math';

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
/**
 * Assumed camera focal length in picture heights (≈ a 65° wide laptop webcam at 4:3). Distances
 * scale with it, but comparisons between hand and body don't depend on it.
 */
export const FOCAL_H = 1.05;

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

/**
 * Which way the palm faces, as a unit vector in the mirrored view: x right on screen, y down,
 * z toward the camera. Measured as if this were a right hand; a left hand's palm faces the
 * opposite way (see palmOf). Uses the picture landmarks (their z is depth on the same scale as x),
 * which hold the hand's orientation more steadily than the 3D ones.
 */
export function palmNormal(lm: Landmark[], aspect = 4 / 3): Vec3 | null {
  const w = lm[WRIST], a = lm[INDEX_KNUCKLE], b = lm[PINKY_KNUCKLE];
  const ax = (a.x - w.x) * aspect, ay = a.y - w.y, az = ((a.z ?? 0) - (w.z ?? 0)) * aspect;
  const bx = (b.x - w.x) * aspect, by = b.y - w.y, bz = ((b.z ?? 0) - (w.z ?? 0)) * aspect;
  const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
  const len = Math.hypot(nx, ny, nz);
  // (picture x is mirrored; MediaPipe's z grows away from the camera)
  return len > 1e-9 ? { x: -nx / len, y: ny / len, z: -nz / len } : null;
}

/** The way a hand's palm faces, from its right-hand-convention normal and which hand it is. */
export function palmOf(normal: Vec3, side: Side): Vec3 {
  return side === 'r' ? { ...normal } : { x: -normal.x, y: -normal.y, z: -normal.z };
}

/**
 * Apparent size ÷ real size of the palm (picture heights per metre), which is FOCAL_H / distance.
 *
 * The palm (wrist + four knuckles) is a flat rigid plate. Fit the 2×2 map from its own plane to the
 * picture; the plate always contains one direction lying flat to the camera, which that map
 * stretches by exactly the scale — its largest singular value. Exact at any hand angle, and the
 * least-squares fit over five points damps noise.
 */
export function palmScale(img: Landmark[], world: Landmark[], aspect: number): number | null {
  const ids = [WRIST, INDEX_KNUCKLE, MIDDLE_KNUCKLE, 13, PINKY_KNUCKLE];
  const P = ids.map(i => world[i]), Q = ids.map(i => ({ x: img[i].x * aspect, y: img[i].y }));
  const n = ids.length;
  const c3 = { x: 0, y: 0, z: 0 }, c2 = { x: 0, y: 0 };
  for (let i = 0; i < n; i++) {
    c3.x += P[i].x / n; c3.y += P[i].y / n; c3.z += (P[i].z ?? 0) / n;
    c2.x += Q[i].x / n; c2.y += Q[i].y / n;
  }
  // an orthonormal basis of the palm plane: along the knuckle row, and toward the wrist
  const sub3 = (a: Landmark, b: Landmark) => [a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0)];
  const unit = (v: number[]) => { const l = Math.hypot(...v); return l > 1e-6 ? v.map(x => x / l) : null; };
  const e1 = unit(sub3(world[PINKY_KNUCKLE], world[INDEX_KNUCKLE]));
  const toWrist = sub3(world[WRIST], world[MIDDLE_KNUCKLE]);
  if (!e1) return null;
  const along = toWrist[0] * e1[0] + toWrist[1] * e1[1] + toWrist[2] * e1[2];
  const e2 = unit(toWrist.map((x, i) => x - along * e1[i]));
  if (!e2) return null;
  // least squares: picture offset ≈ A · (u, v) plane coordinates
  let suu = 0, suv = 0, svv = 0, xu = 0, xv = 0, yu = 0, yv = 0;
  for (let i = 0; i < n; i++) {
    const d = [P[i].x - c3.x, P[i].y - c3.y, (P[i].z ?? 0) - c3.z];
    const u = d[0] * e1[0] + d[1] * e1[1] + d[2] * e1[2], v = d[0] * e2[0] + d[1] * e2[1] + d[2] * e2[2];
    const qx = Q[i].x - c2.x, qy = Q[i].y - c2.y;
    suu += u * u; suv += u * v; svv += v * v;
    xu += qx * u; xv += qx * v; yu += qy * u; yv += qy * v;
  }
  const det = suu * svv - suv * suv;
  if (det < 1e-12) return null;
  const a = (xu * svv - xv * suv) / det, b = (xv * suu - xu * suv) / det;
  const c = (yu * svv - yv * suv) / det, d = (yv * suu - yu * suv) / det;
  // largest singular value of [[a, b], [c, d]]
  const T = a * a + b * b + c * c + d * d, D = a * d - b * c;
  const sMax = Math.sqrt((T + Math.sqrt(Math.max(0, T * T - 4 * D * D))) / 2);
  return sMax > 0 ? sMax : null;
}

/**
 * Apparent size ÷ real size of the whole hand (picture heights per metre) = FOCAL_H / distance.
 *
 * Fits the 2×3 map from the hand's 3D shape (all 21 points) to the picture by least squares. For a
 * camera this is the scale times two rows of a rotation, so its largest singular value is the scale;
 * that also holds for a flat open hand (where one direction is unconstrained — a small ridge keeps
 * it at zero). Using every point averages out much more jitter than a few palm bones.
 */
export function handScale(img: Landmark[], world: Landmark[], aspect: number): number | null {
  const n = Math.min(img.length, world.length);
  if (n < 5) return null;
  let cx = 0, cy = 0, cz = 0, qx = 0, qy = 0;
  for (let i = 0; i < n; i++) {
    cx += world[i].x / n; cy += world[i].y / n; cz += (world[i].z ?? 0) / n;
    qx += (img[i].x * aspect) / n; qy += img[i].y / n;
  }
  // S = Σ X Xᵀ (3×3, with a small ridge), B = Σ q Xᵀ (2×3)
  const S = [[1e-6, 0, 0], [0, 1e-6, 0], [0, 0, 1e-6]], B = [[0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++) {
    const X = [world[i].x - cx, world[i].y - cy, (world[i].z ?? 0) - cz];
    const q = [img[i].x * aspect - qx, img[i].y - qy];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) S[r][c] += X[r] * X[c];
      B[0][r] += q[0] * X[r];
      B[1][r] += q[1] * X[r];
    }
  }
  const inv = invert3(S);
  if (!inv) return null;
  const M = B.map(row => [0, 1, 2].map(c => row[0] * inv[0][c] + row[1] * inv[1][c] + row[2] * inv[2][c]));
  // largest singular value of the 2×3 M: from the 2×2 M·Mᵀ
  const a = M[0][0] ** 2 + M[0][1] ** 2 + M[0][2] ** 2, d = M[1][0] ** 2 + M[1][1] ** 2 + M[1][2] ** 2;
  const b = M[0][0] * M[1][0] + M[0][1] * M[1][1] + M[0][2] * M[1][2];
  const T = a + d, D = a * d - b * b;
  const sMax = Math.sqrt((T + Math.sqrt(Math.max(0, T * T - 4 * D))) / 2);
  return sMax > 0 ? sMax : null;
}

function invert3(m: number[][]): number[][] | null {
  const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-18) return null;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
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

/** Shoulder → wrist in 3D, with x flipped to match the mirrored picture. */
function reachOf(w: Landmark[], side: Side): { x: number; y: number; z: number } {
  const s = w[ARM[side].shoulder], r = w[ARM[side].wrist];
  return { x: -(r.x - s.x), y: r.y - s.y, z: (r.z ?? 0) - (s.z ?? 0) };
}

/**
 * Raw MediaPipe results → mirrored TrackingFrame (moving right moves right on screen).
 * `handsWorld` / `poseWorld` are MediaPipe's 3D landmarks in metres; hand shape and arm
 * straightness come from them when present.
 */
export function toFrame(
  t: number, hands: Landmark[][], pose: Landmark[] | undefined, handsWorld: Landmark[][] = [], poseWorld?: Landmark[],
  /** Picture width ÷ height. */
  aspect = 4 / 3,
): TrackingFrame {
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
      reach: poseWorld ? reachOf(poseWorld, side) : null,
    };
  };
  const arms = { l: armOf('l'), r: armOf('r') };
  const body = pose && poseWorld && visible(pose[L_SHOULDER]) && visible(pose[R_SHOULDER]) ? bodyDepth(pose, poseWorld, aspect) : null;
  const handObsList = hands.map((lm, i) => handObs(lm, handsWorld[i] ?? lm, handsWorld[i] && body ? { body, aspect } : null, aspect));
  matchHandsToArms(handObsList, arms);
  return {
    t, head, shoulderL, shoulderR, hands: handObsList, arms, face: pose ? faceOf(pose) : null,
    body: body && { span3: body.span3, span2: body.span2 },
  };
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

/** Distance to the shoulders (from their apparent vs real width) and where their centre is in the picture. */
function bodyDepth(pose: Landmark[], world: Landmark[], aspect: number): { distance: number; mid: Vec2; span3: number; span2: number } | null {
  const a = pose[L_SHOULDER], b = pose[R_SHOULDER];
  const real = d3(world[L_SHOULDER], world[R_SHOULDER]), seen = Math.hypot((a.x - b.x) * aspect, a.y - b.y);
  if (real < 1e-4 || seen < 1e-4) return null;
  return { distance: (FOCAL_H * real) / seen, mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, span3: real, span2: seen };
}

function handObs(lm: Landmark[], world: Landmark[], depth: { body: { distance: number; mid: Vec2 }; aspect: number } | null, aspect: number): HandObs {
  const c = { x: 0, y: 0 };
  for (const i of PALM) { c.x += lm[i].x / PALM.length; c.y += lm[i].y / PALM.length; }
  const open = openness(world);
  let body3: HandObs['body3'] = null, handDepth: number | null = null;
  // a fist is a rigid 3D shape: fit all of it; a flat open hand is better measured by its palm
  const scale = depth && (open < 0.5
    ? handScale(lm, world, depth.aspect) ?? palmScale(lm, world, depth.aspect)
    : palmScale(lm, world, depth.aspect));
  if (depth && scale) {
    // back-project the palm and the shoulder centre to metres, then take the difference (mirrored x)
    const dHand = FOCAL_H / scale, { distance: dBody, mid } = depth.body;
    const toM = (p: Vec2, d: number) => ({ x: ((p.x - 0.5) * depth.aspect * d) / FOCAL_H, y: ((p.y - 0.5) * d) / FOCAL_H });
    const h = toM(c, dHand), s = toM(mid, dBody);
    body3 = { x: -(h.x - s.x), y: h.y - s.y, z: dBody - dHand };
    handDepth = dHand;
  }
  return {
    center: mirror(c),
    size: dist(lm[WRIST], lm[MIDDLE_KNUCKLE]),
    open,
    facing: palmFacing(world),
    normal: palmNormal(lm, aspect),
    body3,
    depth: handDepth,
  };
}
