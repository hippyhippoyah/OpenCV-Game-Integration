import type { Vec2 } from '../math';

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
}

export interface HandObs {
  /** Palm centre. */
  center: Point;
  /** Wrist → middle-knuckle distance; grows as the hand moves toward the camera. */
  size: number;
  /** 0 = fist … 1 = open hand. */
  open: number;
  /** 1 = palm faces the camera, 0 = palm edge-on (e.g. palms facing each other). */
  facing: number;
  /** The arm this hand belongs to (nearest pose wrist), when a body is visible. */
  side?: Side;
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
}

export interface Tracker {
  /** Newest frame, or null if nothing new since the last call. `now` is in milliseconds. */
  poll(now: number): TrackingFrame | null;
  dispose(): void;
}
