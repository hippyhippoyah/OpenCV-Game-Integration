import type { Vec2, Vec3 } from '../math';

/** Mirrored, normalized video coordinates: (0,0) top-left, (1,1) bottom-right, as seen in a mirror. */
export type Point = Vec2;

/** Left/right as seen on screen — in the mirrored view that is also your own left/right. */
export type Side = 'l' | 'r';

/** A pose point. May lie outside 0…1 when the model estimates it beyond the picture. */
export interface BodyPoint extends Point {
  /** How sure the model is that the point is visible, 0…1. */
  vis: number;
}

export interface ArmObs {
  shoulder: BodyPoint;
  elbow: BodyPoint;
  wrist: BodyPoint;
  /** 0 = elbow fully bent … 1 = straight arm, from the 3D pose; null if unknown. */
  extension: number | null;
  /**
   * Shoulder → wrist in metres from the 3D pose, mirrored like the picture (x right, y down,
   * z negative toward the camera); null if unknown.
   */
  reach: { x: number; y: number; z: number } | null;
}

export interface HandObs {
  /** Palm centre. */
  center: Point;
  /** Wrist → middle-knuckle distance; grows as the hand moves toward the camera. */
  size: number;
  /** 0 = fist … 1 = open hand. */
  open: number;
  /** How straight each finger is (index, middle, ring, pinky), 0 = curled … 1 = straight; missing from mock hands. */
  fingers?: number[];
  /** 1 = palm faces the camera, 0 = palm edge-on (e.g. palms facing each other). */
  facing: number;
  /**
   * The palm's direction if this is a right hand (x right on screen, y down, z toward the camera);
   * a left hand's palm faces the opposite way. Undefined when not measured (mock hands may leave it out).
   */
  normal?: Vec3 | null;
  /** The arm this hand belongs to (nearest pose wrist), when a body is visible. */
  side?: Side;
  /**
   * Palm centre in metres relative to the shoulder centre, from apparent sizes: x right on screen,
   * y down, z = how far in front of the shoulders. Null without 3D landmarks.
   */
  body3?: { x: number; y: number; z: number } | null;
  /** Distance from the camera to the hand, metres (from its apparent vs real size); null without 3D landmarks. */
  depth?: number | null;
}

/** One tracked camera frame. Camera and mock trackers both produce these. */
export interface TrackingFrame {
  /** Seconds. */
  t: number;
  head: Point | null;
  shoulderL: Point | null;
  shoulderR: Point | null;
  hands: HandObs[];
  /** Pose-tracked arms; wrists are still estimated when the hand itself isn't found. */
  arms: Record<Side, ArmObs | null>;
  /** Head turn (+ = nose toward screen right, in ear-widths) and tilt (radians); null if unsure. */
  face: { yaw: number; roll: number } | null;
  /**
   * Raw shoulder measurements for the body's distance: real width (m, from the 3D pose) and
   * apparent width (picture heights). Null without 3D pose landmarks.
   */
  body?: { span3: number; span2: number } | null;
}

export interface Tracker {
  /** Newest frame, or null if nothing new since the last call. `now` is in milliseconds. */
  poll(now: number): TrackingFrame | null;
  dispose(): void;
}
