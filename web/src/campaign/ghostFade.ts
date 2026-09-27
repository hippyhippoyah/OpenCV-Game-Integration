/** Seconds the ghost hands take to fade in or out. */
export const GHOST_FADE_S = 0.6;
/** Seconds of no progress on the lesson before the ghost fades back in to help. */
export const GHOST_STALL_S = 4;

/**
 * Ghost hands' target strength, per the design: full strength until the lesson's first success
 * (`doneCount > 0`), then faded out — reappearing only if you stall (no progress) for
 * `GHOST_STALL_S` seconds — and gone once the lesson is complete. `prev` is last frame's alpha,
 * `dt` the frame time, and `sinceProgressS` seconds since `doneCount` last increased.
 */
export function ghostAlpha(prev: number, dt: number, doneCount: number, sinceProgressS: number, complete: boolean): number {
  if (complete) return Math.max(0, prev - dt / GHOST_FADE_S);
  const target = doneCount <= 0 || sinceProgressS >= GHOST_STALL_S ? 1 : 0;
  const step = dt / GHOST_FADE_S;
  if (target > prev) return Math.min(target, prev + step);
  if (target < prev) return Math.max(target, prev - step);
  return prev;
}
