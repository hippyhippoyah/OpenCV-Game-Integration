import { describe, expect, it } from 'vitest';
import { openness, palmFacing, toFrame, type Landmark } from './landmarks';

const pose = (): Landmark[] => Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 1 }));

/** 3D hand in metres: wrist at the origin, fingers pointing up (−y). 'zy' turns the palm edge-on to the camera. */
function worldHand(curled: boolean, plane: 'xy' | 'zy' = 'xy'): Landmark[] {
  const lm: Landmark[] = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
  const put = (i: number, across: number, y: number, depth: number) => {
    lm[i] = plane === 'xy' ? { x: across, y, z: depth } : { x: depth, y, z: across };
  };
  [-0.02, -0.007, 0.007, 0.02].forEach((across, f) => {
    const k = 5 + f * 4;
    put(k, across, -0.04, 0);
    put(k + 1, across, -0.065, 0);
    if (curled) {
      put(k + 2, across, -0.065, 0.02);
      put(k + 3, across, -0.045, 0.02);
    } else {
      put(k + 2, across, -0.085, 0);
      put(k + 3, across, -0.1, 0);
    }
  });
  return lm;
}

describe('toFrame', () => {
  it('mirrors x, orders shoulders left-to-right on screen and summarises hands', () => {
    const p = pose();
    p[0] = { x: 0.4, y: 0.3, visibility: 1 };
    p[11] = { x: 0.6, y: 0.5, visibility: 1 };
    p[12] = { x: 0.4, y: 0.5, visibility: 1 };
    const hand: Landmark[] = Array.from({ length: 21 }, () => ({ x: 0.3, y: 0.6 }));
    hand[0] = { x: 0.3, y: 0.7 };
    const f = toFrame(1, [hand], p, [worldHand(false)]);
    expect(f.head!.x).toBeCloseTo(0.6);
    expect(f.shoulderL!.x).toBeCloseTo(0.4);
    expect(f.shoulderR!.x).toBeCloseTo(0.6);
    expect(f.hands[0].center.x).toBeCloseTo(0.7);
    expect(f.hands[0].center.y).toBeCloseTo(0.62);
    expect(f.hands[0].size).toBeCloseTo(0.1);
    expect(f.hands[0].open).toBeGreaterThan(0.9);
    expect(f.hands[0].facing).toBeGreaterThan(0.9);
  });

  it('drops points the model is unsure about', () => {
    const p = pose();
    p[0].visibility = 0.2;
    p[11].visibility = 0.2;
    const f = toFrame(0, [], p);
    expect(f.head).toBeNull();
    expect(f.shoulderL).toBeNull();
  });

  it('handles no person', () => {
    expect(toFrame(0, [], undefined)).toEqual({ t: 0, head: null, shoulderL: null, shoulderR: null, hands: [], arms: { l: null, r: null }, face: null, body: null });
  });
});

describe('hand shape', () => {
  it('tells an open hand from a fist in 3D', () => {
    expect(openness(worldHand(false))).toBeGreaterThan(0.9);
    expect(openness(worldHand(true))).toBeLessThan(0.1);
  });

  it('gives the same answer when the hand points another way', () => {
    expect(openness(worldHand(false, 'zy'))).toBeGreaterThan(0.9);
    expect(openness(worldHand(true, 'zy'))).toBeLessThan(0.1);
  });

  it('measures whether the palm faces the camera or is edge-on', () => {
    expect(palmFacing(worldHand(false, 'xy'))).toBeGreaterThan(0.9);
    expect(palmFacing(worldHand(false, 'zy'))).toBeLessThan(0.1);
  });
});

describe('body', () => {
  /** Raw (un-mirrored) pose: the person's left side (11/13/15) sits on the image's right. */
  const fullPose = (): Landmark[] => {
    const p = pose();
    p[11] = { x: 0.6, y: 0.5, visibility: 1 };
    p[12] = { x: 0.4, y: 0.5, visibility: 1 };
    p[13] = { x: 0.65, y: 0.65, visibility: 0.9 };
    p[14] = { x: 0.35, y: 0.65, visibility: 0.9 };
    p[15] = { x: 0.7, y: 0.55, visibility: 0.9 };
    p[16] = { x: -0.2, y: 0.8, visibility: 0.1 }; // right wrist estimated beyond the picture
    return p;
  };
  const flatHand = (x: number, y: number): Landmark[] => Array.from({ length: 21 }, () => ({ x, y }));

  it('labels arms by the body and keeps off-screen wrist estimates', () => {
    const f = toFrame(0, [], fullPose());
    expect(f.arms.l!.shoulder.x).toBeCloseTo(0.4);
    expect(f.arms.l!.wrist.x).toBeCloseTo(0.3);
    expect(f.arms.r!.wrist.x).toBeCloseTo(1.2);
    expect(f.arms.r!.wrist.vis).toBeCloseTo(0.1);
    expect(f.arms.l!.extension).toBeNull();
  });

  it('has no arms without visible shoulders', () => {
    const p = fullPose();
    p[11].visibility = 0.1;
    expect(toFrame(0, [], p).arms.l).toBeNull();
  });

  it('measures arm straightness in 3D', () => {
    const world: Landmark[] = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0 }));
    world[11] = { x: 0.2, y: 0, z: 0 }; world[13] = { x: 0.2, y: 0.3, z: 0 }; world[15] = { x: 0.2, y: 0.3, z: -0.3 }; // bent 90°
    world[12] = { x: -0.2, y: 0, z: 0 }; world[14] = { x: -0.2, y: 0, z: -0.3 }; world[16] = { x: -0.2, y: 0, z: -0.6 }; // straight at the camera
    const f = toFrame(0, [], fullPose(), [], world);
    expect(f.arms.r!.extension).toBeGreaterThan(0.9);
    expect(f.arms.l!.extension).toBeLessThan(0.3);
  });

  it('gives each arm a 3D reach direction, mirrored like the picture', () => {
    const world: Landmark[] = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0 }));
    world[12] = { x: -0.2, y: 0, z: 0 }; world[16] = { x: -0.4, y: 0.1, z: -0.5 }; // right arm: toward the camera, off to its right
    const reach = toFrame(0, [], fullPose(), [], world).arms.r!.reach!;
    expect(reach.z).toBeLessThan(0);
    expect(reach.x).toBeCloseTo(0.2); // mirrored: the image's left is the screen's right
    expect(reach.y).toBeCloseTo(0.1);
    expect(toFrame(0, [], fullPose()).arms.r!.reach).toBeNull();
  });

  it('matches each hand to the nearest wrist, whatever order the hands come in', () => {
    const p = fullPose();
    p[16] = { x: 0.25, y: 0.55, visibility: 0.9 };
    const f = toFrame(0, [flatHand(0.27, 0.55), flatHand(0.72, 0.55)], p);
    expect(f.hands.map(h => h.side)).toEqual(['r', 'l']);
  });

  it('reads head turn and tilt', () => {
    const p = fullPose();
    p[0] = { x: 0.5, y: 0.3, visibility: 1 };
    p[2] = { x: 0.53, y: 0.28, visibility: 1 }; p[5] = { x: 0.47, y: 0.28, visibility: 1 };
    p[7] = { x: 0.58, y: 0.3, visibility: 1 }; p[8] = { x: 0.42, y: 0.3, visibility: 1 };
    const straight = toFrame(0, [], p).face!;
    expect(straight.yaw).toBeCloseTo(0);
    expect(straight.roll).toBeCloseTo(0);
    p[0] = { x: 0.45, y: 0.3, visibility: 1 };
    expect(toFrame(0, [], p).face!.yaw).toBeGreaterThan(0.2);
  });
});

describe('how far each hand is in front of the body', () => {
  // imported lazily so the rest of this file doesn't depend on the simulator
  const sim = () => import('../sim/synthetic');

  it("reads each fist's reach from palm size vs shoulder size, at any distance and fist angle", async () => {
    const { ASPECT, POSES, SyntheticCamera, guardState } = await sim();
    const cam = new SyntheticCamera({ handNoise: 0, poseNoise: 0, handWorldNoise: 0, poseWorldNoise: 0 });
    for (const distance of [1.5, 2.5]) {
      const reachOf = (reach: typeof POSES.guard) => {
        const raw = cam.raw(guardState(distance, { r: { reach } }), 0);
        const f = toFrame(0, raw.hands, raw.pose, raw.handsWorld, raw.poseWorld, ASPECT);
        return f.hands.find(h => h.side === 'r')!.body3!;
      };
      const guard = reachOf(POSES.guard), jab = reachOf(POSES.jab), cross = reachOf(POSES.cross);
      for (const [pos, reach] of [[guard, POSES.guard], [jab, POSES.jab], [cross, POSES.cross]] as const) {
        // measured at the palm, a few cm past the wrist the pose is defined by
        expect(pos.z, `${distance} m`).toBeGreaterThan(reach.fwd - 0.03);
        expect(pos.z, `${distance} m`).toBeLessThan(reach.fwd + 0.12);
      }
      expect(jab.z - guard.z, `${distance} m`).toBeGreaterThan(0.2);
      expect(cross.x, 'a right cross ends up left of centre').toBeLessThan(0);
      expect(guard.x, 'the right fist guards on the right').toBeGreaterThan(0);
    }
  });

  it('has no 3D position without MediaPipe world landmarks', () => {
    const f = toFrame(0, [Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5 }))], pose());
    expect(f.hands[0].body3).toBeNull();
  });
});
