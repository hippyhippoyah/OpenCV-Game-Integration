import type { Vec2 } from '../math';

/** Mirrored, normalized video coordinates: (0,0) top-left, (1,1) bottom-right, as seen in a mirror. */
export type Point = Vec2;

export interface HandObs {
  /** Palm centre. */
  center: Point;
  /** Wrist → middle-knuckle distance; grows as the hand moves toward the camera. */
  size: number;
}

/** One tracked camera frame. Camera and mock trackers both produce these. */
export interface TrackingFrame {
  /** Seconds. */
  t: number;
  head: Point | null;
  shoulderL: Point | null;
  shoulderR: Point | null;
  hands: HandObs[];
}

export interface Tracker {
  /** Newest frame, or null if nothing new since the last call. `now` is in milliseconds. */
  poll(now: number): TrackingFrame | null;
  dispose(): void;
}
