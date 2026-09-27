import { describe, expect, it } from 'vitest';
import { GHOST_LOOP_S, ghostPose } from './ghost';

describe('ghost hands', () => {
  it('loop smoothly for each move taught in the campaign', () => {
    for (const id of ['move', 'punch', 'flurry', 'shield', 'pillar', 'palm', 'charge', 'wall', 'ultimate']) {
      const a = ghostPose(id, 0), b = ghostPose(id, GHOST_LOOP_S);
      expect(a, id).not.toBeNull();
      expect(b!.r.pos.x).toBeCloseTo(a!.r.pos.x, 5);
      expect(b!.r.pos.y).toBeCloseTo(a!.r.pos.y, 5);
    }
  });

  it('show the move: a palm push opens the right hand and brings it forward', () => {
    const rest = ghostPose('palm', 0)!, out = ghostPose('palm', GHOST_LOOP_S * 0.45)!;
    expect(out.r.open).toBe(true);
    expect(out.r.scale).toBeGreaterThan(rest.r.scale);
  });

  it('a charge drops a fist to the hip and holds it', () => {
    const held = ghostPose('charge', GHOST_LOOP_S * 0.4)!;
    expect(held.r.open).toBe(false);
    expect(held.r.pos.y).toBeGreaterThan(40);
  });

  it('no ghost for moves not in the campaign', () => {
    expect(ghostPose('xblock', 0)).toBeNull();
  });
});
