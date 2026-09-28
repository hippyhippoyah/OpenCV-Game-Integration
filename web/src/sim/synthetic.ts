/**
 * A synthetic person in front of a simulated webcam, producing MediaPipe-style landmarks — with
 * noise, the pose model's shallow depth for arms pointing at the camera, and the hand tracker
 * dropping fast-moving (blurred) hands — so detection can be tested against realistic motion.
 *
 * Camera frame (metres): x right in the un-mirrored picture, y down, z away from the camera.
 * The person faces the camera, so their right side appears on the picture's left.
 */
import type { RawLandmarks } from '../debug/recorder';
import { FOCAL_H, toFrame, type Landmark } from '../input/landmarks';
import type { Side, TrackingFrame } from '../input/types';
import { mulberry32 } from '../math';

export interface V3 { x: number; y: number; z: number }
const v3 = (x: number, y: number, z: number): V3 => ({ x, y, z });
const add = (a: V3, b: V3) => v3(a.x + b.x, a.y + b.y, a.z + b.z);
const sub = (a: V3, b: V3) => v3(a.x - b.x, a.y - b.y, a.z - b.z);
const mul = (a: V3, k: number) => v3(a.x * k, a.y * k, a.z * k);
const len = (a: V3) => Math.hypot(a.x, a.y, a.z);
const norm = (a: V3) => mul(a, 1 / (len(a) || 1));
const dot = (a: V3, b: V3) => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: V3, b: V3) => v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);

export const ASPECT = 4 / 3;
const UPPER_ARM = 0.29, FOREARM = 0.27, SHOULDER_HALF = 0.19, SHOULDER_Y = -0.05;

/** Picture coordinates (normalized, un-mirrored) of a camera-frame point. */
export function project(p: V3): Landmark {
  return { x: 0.5 + ((p.x / p.z) * FOCAL_H) / ASPECT, y: 0.5 + (p.y / p.z) * FOCAL_H };
}

/** A wrist relative to its own shoulder: out = away from the body's midline, up, fwd = toward the camera. */
export interface Reach { out: number; up: number; fwd: number }

export const POSES = {
  guard: { out: -0.08, up: 0.12, fwd: 0.25 },
  jab: { out: -0.05, up: 0.1, fwd: 0.52 },
  cross: { out: -0.24, up: 0.15, fwd: 0.46 },
  hook: { out: -0.22, up: 0.12, fwd: 0.33 },
  uppercut: { out: -0.05, up: 0.26, fwd: 0.4 },
  rest: { out: 0.02, up: -0.45, fwd: 0.08 },
  shield: { out: 0.08, up: 0.12, fwd: 0.32 },
  xblock: { out: -0.26, up: 0.2, fwd: 0.22 },
} satisfies Record<string, Reach>;

export interface HandKey {
  reach: Reach; open: boolean;
  /** An open hand's palm: 0 = toward the camera (default) … 1 = turned in, facing the other hand. */
  turn?: number;
}
export interface BodyState {
  /** Distance from the camera to the shoulders, metres. */
  distance: number;
  hands: Record<Side, HandKey>;
}

const smoothstep = (k: number) => (k <= 0 ? 0 : k >= 1 ? 1 : k * k * (3 - 2 * k));
export const lerpReach = (a: Reach, b: Reach, k: number): Reach => {
  const e = smoothstep(k);
  return { out: a.out + (b.out - a.out) * e, up: a.up + (b.up - a.up) * e, fwd: a.fwd + (b.fwd - a.fwd) * e };
};

/** A punch: guard → target (outS) → hold (holdS) → back to guard (backS), starting at t0. */
export function punchReach(t: number, t0: number, target: Reach, outS = 0.12, holdS = 0.12, backS = 0.2): Reach {
  const g = POSES.guard;
  if (t < t0) return g;
  if (t < t0 + outS) return lerpReach(g, target, (t - t0) / outS);
  if (t < t0 + outS + holdS) return target;
  return lerpReach(target, g, (t - t0 - outS - holdS) / backS);
}

/** Person's own left is the picture's right (+x). */
const outward = (side: Side) => (side === 'l' ? 1 : -1);

function shoulderAt(side: Side, distance: number): V3 {
  return v3(outward(side) * SHOULDER_HALF, SHOULDER_Y, distance);
}

function wristAt(side: Side, distance: number, r: Reach): V3 {
  return add(shoulderAt(side, distance), v3(outward(side) * r.out, -r.up, -r.fwd));
}

/** Two-bone IK: the elbow hangs down, out and back. */
function elbowAt(side: Side, s: V3, w: V3): V3 {
  const toW = sub(w, s), d = Math.min(len(toW), UPPER_ARM + FOREARM - 1e-3), dir = norm(toW);
  const cosA = (UPPER_ARM * UPPER_ARM + d * d - FOREARM * FOREARM) / (2 * UPPER_ARM * d);
  const a = Math.acos(Math.max(-1, Math.min(1, cosA)));
  const pole = v3(outward(side) * 0.4, 1, 0.3);
  const perp = norm(sub(pole, mul(dir, dot(pole, dir))));
  return add(s, add(mul(dir, UPPER_ARM * Math.cos(a)), mul(perp, UPPER_ARM * Math.sin(a))));
}

/**
 * 21 hand landmarks (camera frame). A fist points along the forearm with the knuckle row
 * horizontal; an open hand has its fingers up and palm toward the camera.
 */
function handPoints(side: Side, wrist: V3, elbow: V3, open: boolean, turn = 0): V3[] {
  let f: V3, lat: V3;
  if (open) {
    f = v3(0, -1, 0);
    // thumb side toward the midline; turned in (palms facing each other), the thumb points back at you
    const a = (turn * Math.PI) / 2;
    lat = v3(-outward(side) * Math.cos(a), 0, Math.sin(a));
  } else {
    // a fist keeps its knuckle row across the body: the picture's x, made perpendicular to the forearm
    f = norm(sub(wrist, elbow));
    lat = norm(sub(v3(1, 0, 0), mul(f, f.x)));
  }
  const n = norm(cross(lat, f));
  const pts: V3[] = new Array(21);
  pts[0] = wrist;
  const at = (base: V3, df: number, dl: number, dn: number) => add(base, add(mul(f, df), add(mul(lat, dl), mul(n, dn))));
  [-0.03, -0.01, 0.01, 0.03].forEach((across, i) => {
    const k = 5 + i * 4, mcp = at(wrist, 0.085, -across, 0);
    pts[k] = mcp;
    if (open) {
      pts[k + 1] = at(mcp, 0.04, 0, 0); pts[k + 2] = at(mcp, 0.07, 0, 0); pts[k + 3] = at(mcp, 0.095, 0, 0);
    } else {
      pts[k + 1] = at(mcp, 0.02, 0, 0.025); pts[k + 2] = at(mcp, 0, 0, 0.035); pts[k + 3] = at(mcp, -0.02, 0, 0.03);
    }
  });
  const thumbBase = at(wrist, 0.02, -0.035, 0);
  for (let i = 1; i <= 4; i++) pts[i] = open ? at(thumbBase, 0.02 * i, -0.015 * i, 0) : at(thumbBase, 0.022 * i, 0.01 * i, 0.015);
  return pts;
}

export interface SimOptions {
  seed?: number;
  /** Picture-coordinate noise (hand / pose landmarks). */
  handNoise?: number;
  poseNoise?: number;
  /** How much of the arm's true depth the pose model reports (MediaPipe underestimates it). */
  poseDepthScale?: number;
  /** Noise on MediaPipe's 3D landmarks, metres (hand / pose). */
  handWorldNoise?: number;
  poseWorldNoise?: number;
  /** Hands moving faster than this across the picture (heights/s) may be lost to blur. */
  blurSpeed?: number;
  blurDropChance?: number;
}

/** Produces MediaPipe-style raw landmarks for body states, frame by frame. */
export class SyntheticCamera {
  private rand: () => number;
  private prevWrist: Partial<Record<Side, Landmark>> = {};
  private o: Required<SimOptions>;

  constructor(opts: SimOptions = {}) {
    this.o = {
      seed: 1, handNoise: 0.0015, poseNoise: 0.003, poseDepthScale: 0.5, handWorldNoise: 0.002, poseWorldNoise: 0.012,
      blurSpeed: 1.4, blurDropChance: 0.6, ...opts,
    };
    this.rand = mulberry32(this.o.seed);
  }

  private gauss(sigma: number): number {
    const u = Math.max(1e-9, this.rand()), w = this.rand();
    return sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w);
  }

  private img(p: V3, sigma: number, visible = 1): Landmark {
    const q = project(p);
    return { x: q.x + this.gauss(sigma) / ASPECT, y: q.y + this.gauss(sigma), visibility: visible };
  }

  raw(state: BodyState, dt: number): RawLandmarks {
    const D = state.distance, o = this.o;
    const hands: Landmark[][] = [], handsWorld: Landmark[][] = [];
    const pose: Landmark[] = [], poseWorld: Landmark[] = [];
    const hip = v3(0, SHOULDER_Y + 0.5, D);
    const put = (i: number, p: V3, depthFrom?: V3) => {
      const q = project(p);
      const inPic = q.x >= 0 && q.x <= 1 && q.y >= 0 && q.y <= 1;
      pose[i] = this.img(p, o.poseNoise, inPic ? 0.99 : 0.2);
      // pose depth: arms pointing at the camera come out much shallower than they are
      const z = depthFrom ? (depthFrom.z - hip.z) + (p.z - depthFrom.z) * o.poseDepthScale : p.z - hip.z;
      const n = o.poseWorldNoise;
      poseWorld[i] = { x: p.x - hip.x + this.gauss(n), y: p.y - hip.y + this.gauss(n), z: z + this.gauss(n * 3), visibility: pose[i].visibility };
    };
    put(0, v3(0, SHOULDER_Y - 0.22, D - 0.1));
    put(2, v3(0.032, SHOULDER_Y - 0.25, D - 0.08)); put(5, v3(-0.032, SHOULDER_Y - 0.25, D - 0.08));
    put(7, v3(0.075, SHOULDER_Y - 0.23, D)); put(8, v3(-0.075, SHOULDER_Y - 0.23, D));
    put(23, v3(0.12, SHOULDER_Y + 0.5, D)); put(24, v3(-0.12, SHOULDER_Y + 0.5, D));
    const order: Side[] = this.rand() < 0.5 ? ['l', 'r'] : ['r', 'l'];
    for (const side of order) {
      const idx = side === 'l' ? [11, 13, 15] : [12, 14, 16];
      const s = shoulderAt(side, D), w = wristAt(side, D, state.hands[side].reach), e = elbowAt(side, s, w);
      put(idx[0], s); put(idx[1], e, s); put(idx[2], w, s);
      // blur: a fast-moving hand is sometimes lost by the hand tracker
      const wp = project(w), prev = this.prevWrist[side];
      this.prevWrist[side] = wp;
      const speed = prev && dt > 0 ? Math.hypot((wp.x - prev.x) * ASPECT, wp.y - prev.y) / dt : 0;
      const inPic = wp.x > 0.02 && wp.x < 0.98 && wp.y > 0.02 && wp.y < 0.98;
      if (!inPic || (speed > o.blurSpeed && this.rand() < o.blurDropChance)) continue;
      const pts = handPoints(side, w, e, state.hands[side].open, state.hands[side].turn);
      const c = mul(pts.reduce(add, v3(0, 0, 0)), 1 / pts.length);
      // MediaPipe's picture z: depth relative to the wrist, on the same scale as x
      hands.push(pts.map(p => ({ ...this.img(p, o.handNoise), z: ((p.z - w.z) * FOCAL_H) / (w.z * ASPECT) })));
      const n = o.handWorldNoise;
      handsWorld.push(pts.map(p => ({ x: p.x - c.x + this.gauss(n), y: p.y - c.y + this.gauss(n), z: p.z - c.z + this.gauss(n) })));
    }
    for (let i = 0; i < 33; i++) {
      pose[i] ??= { ...pose[0] };
      poseWorld[i] ??= { ...poseWorld[0] };
    }
    return { hands, handsWorld, pose, poseWorld };
  }
}

/** Run a scripted performance through the synthetic camera and `toFrame`. */
export function simulate(script: (t: number) => BodyState, seconds: number, opts: SimOptions & { fps?: number } = {}): TrackingFrame[] {
  const fps = opts.fps ?? 30, cam = new SyntheticCamera(opts), frames: TrackingFrame[] = [];
  for (let i = 0; i <= seconds * fps; i++) {
    const t = i / fps, raw = cam.raw(script(t), 1 / fps);
    frames.push(toFrame(t, raw.hands, raw.pose, raw.handsWorld, raw.poseWorld, ASPECT));
  }
  return frames;
}

/** Both hands in guard at `distance`, with `overrides` per side. */
export function guardState(distance: number, overrides: Partial<Record<Side, Partial<HandKey>>> = {}): BodyState {
  return {
    distance,
    hands: {
      l: { reach: POSES.guard, open: false, ...overrides.l },
      r: { reach: POSES.guard, open: false, ...overrides.r },
    },
  };
}
