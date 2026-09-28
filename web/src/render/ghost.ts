/**
 * Ghost hands: translucent hands that show a move in a slow loop, over where your own hands are.
 * Poses are keyframes in view units (as HandState.pos: x right, y down, guard near y 22), eased.
 */
export interface GhostHand { pos: { x: number; y: number }; open: boolean; scale: number }
export interface GhostPose { l: GhostHand; r: GhostHand }

export const GHOST_LOOP_S = 2.4;

type Key = [t: number, lx: number, ly: number, lOpen: boolean, lScale: number, rx: number, ry: number, rOpen: boolean, rScale: number];
const G = { l: [-12, 22], r: [12, 22] } as const;
/** Keyframes per lesson: t in 0..1 of the loop; the last key must equal the first (it loops). */
const KEYS: Record<string, Key[]> = {
  move: [[0, -12, 22, false, 1, 12, 22, false, 1], [0.3, -30, 22, false, 1, -6, 22, false, 1], [0.6, 6, 22, false, 1, 30, 22, false, 1], [0.8, -12, 36, false, 1, 12, 36, false, 1], [1, -12, 22, false, 1, 12, 22, false, 1]],
  punch: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.15, ...G.l, false, 1, 4, 8, false, 1.6], [0.35, ...G.l, false, 1, ...G.r, false, 1], [0.5, -4, 8, false, 1.6, ...G.r, false, 1], [0.7, ...G.l, false, 1, ...G.r, false, 1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  flurry: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.1, ...G.l, false, 1, 4, 8, false, 1.6], [0.2, -4, 8, false, 1.6, ...G.r, false, 1], [0.3, ...G.l, false, 1, 4, 8, false, 1.6], [0.45, ...G.l, false, 1, ...G.r, false, 1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  shield: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.25, -16, 14, true, 1.1, 16, 14, true, 1.1], [0.85, -16, 14, true, 1.1, 16, 14, true, 1.1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  pillar: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.3, -40, 22, false, 1, -16, 22, false, 1], [0.7, -40, 22, false, 1, -16, 22, false, 1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  palm: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.2, ...G.l, false, 1, ...G.r, true, 1], [0.45, ...G.l, false, 1, 6, 10, true, 1.6], [0.7, ...G.l, false, 1, ...G.r, true, 1], [0.85, ...G.l, false, 1, ...G.r, false, 1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  charge: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.2, ...G.l, false, 1, 20, 58, false, 0.9], [0.6, ...G.l, false, 1, 20, 58, false, 0.9], [0.75, ...G.l, false, 1, 4, 8, false, 1.7], [0.9, ...G.l, false, 1, ...G.r, false, 1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  wall: [[0, -14, 44, true, 1, 14, 44, true, 1], [0.35, -14, 4, true, 1.1, 14, 4, true, 1.1], [0.7, -14, 4, true, 1.1, 14, 4, true, 1.1], [1, -14, 44, true, 1, 14, 44, true, 1]],
  // forearms crossed in front of the chest, fists up
  xblock: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.25, 9, 12, false, 1.1, -9, 12, false, 1.1], [0.8, 9, 12, false, 1.1, -9, 12, false, 1.1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  // jab, jab, then the right palm shoved forward
  onetwo: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.1, ...G.l, false, 1, 4, 8, false, 1.6], [0.2, ...G.l, false, 1, ...G.r, false, 1], [0.3, -4, 8, false, 1.6, ...G.r, false, 1], [0.4, ...G.l, false, 1, ...G.r, true, 1], [0.55, ...G.l, false, 1, 6, 10, true, 1.6], [0.75, ...G.l, false, 1, ...G.r, false, 1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  // right palm pushed, then straight away the left
  volley: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.12, ...G.l, false, 1, ...G.r, true, 1], [0.28, ...G.l, true, 1, 6, 10, true, 1.6], [0.44, -6, 10, true, 1.6, ...G.r, true, 1], [0.6, ...G.l, true, 1, ...G.r, true, 1], [0.75, ...G.l, false, 1, ...G.r, false, 1], [1, ...G.l, false, 1, ...G.r, false, 1]],
  // sweep both open hands up (a wall), then shove both palms forward
  wallbreaker: [[0, -14, 44, true, 1, 14, 44, true, 1], [0.25, -14, 4, true, 1.1, 14, 4, true, 1.1], [0.45, -14, 4, true, 1.1, 14, 4, true, 1.1], [0.6, -12, 8, true, 1.6, 12, 8, true, 1.6], [0.8, -12, 8, true, 1.6, 12, 8, true, 1.6], [1, -14, 44, true, 1, 14, 44, true, 1]],
  // hands together as if to catch a ball, held (they catch fire), then spread wide
  // hands together over the head, held (they burn blue), slammed down, then spread apart
  inferno: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.15, -6, -18, false, 1.1, 6, -18, false, 1.1], [0.5, -6, -18, false, 1.1, 6, -18, false, 1.1], [0.58, -8, 38, false, 1.2, 8, 38, false, 1.2], [0.7, -8, 38, false, 1.2, 8, 38, false, 1.2], [0.8, -42, 34, true, 1.2, 42, 34, true, 1.2], [0.92, -42, 34, true, 1.2, 42, 34, true, 1.2], [1, ...G.l, false, 1, ...G.r, false, 1]],
  ultimate: [[0, ...G.l, false, 1, ...G.r, false, 1], [0.15, -8, 18, true, 1, 8, 18, true, 1], [0.55, -8, 18, true, 1.05, 8, 18, true, 1.05], [0.7, -40, 16, true, 1.1, 40, 16, true, 1.1], [0.88, -40, 16, true, 1.1, 40, 16, true, 1.1], [1, ...G.l, false, 1, ...G.r, false, 1]],
};

const ease = (k: number) => k * k * (3 - 2 * k);

export function ghostPose(lessonId: string, t: number): GhostPose | null {
  const keys = KEYS[lessonId];
  if (!keys) return null;
  const u = ((t % GHOST_LOOP_S) + GHOST_LOOP_S) % GHOST_LOOP_S / GHOST_LOOP_S;
  let i = 1;
  while (i < keys.length - 1 && keys[i][0] < u) i++;
  const a = keys[i - 1], b = keys[i], k = ease(Math.min(1, Math.max(0, (u - a[0]) / (b[0] - a[0] || 1))));
  const mix = (p: number, q: number) => p + (q - p) * k;
  return {
    l: { pos: { x: mix(a[1], b[1]), y: mix(a[2], b[2]) }, open: k < 0.5 ? a[3] : b[3], scale: mix(a[4], b[4]) },
    r: { pos: { x: mix(a[5], b[5]), y: mix(a[6], b[6]) }, open: k < 0.5 ? a[7] : b[7], scale: mix(a[8], b[8]) },
  };
}
