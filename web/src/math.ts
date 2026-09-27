export interface Vec2 { x: number; y: number }

export const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));
export const lerp = (a: number, b: number, k: number): number => a + (b - a) * k;
export const dist = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);

/** Distance from p to the segment a–b. */
export function distToSeg(p: Vec2, a: Vec2, b: Vec2): number {
  const vx = b.x - a.x, vy = b.y - a.y, l2 = vx * vx + vy * vy || 1;
  const k = clamp(((p.x - a.x) * vx + (p.y - a.y) * vy) / l2, 0, 1);
  return Math.hypot(p.x - a.x - vx * k, p.y - a.y - vy * k);
}

/** Small seeded PRNG so game tests are deterministic. */
export function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let r = Math.imul(a ^ (a >>> 15), 1 | a);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
