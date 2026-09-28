import type { MoveName } from '../game/game';
import type { Scene } from '../render/scenes';
import type { ScrollId } from './progress';
import type { FightScript } from './scripts';

export interface ScrollDef { id: ScrollId; name: string; moves: MoveName[]; lessonId: string }

/** A stop on the path: arrive, maybe find a scroll, practise, fight. Distances are metres along the path. */
export interface StopDef {
  id: string;
  place: string;
  /** Ren's lines on arrival. */
  ren: string[];
  /** The scroll found on the way into this stop (at scrollAt). */
  scroll?: ScrollId;
  scrollAt?: number;
  /** Tutorial lesson ids practised (with ghost hands) before the fight. */
  practice: string[];
  fight: FightScript;
  /** The move that earns this stop's third flame. */
  newMove: MoveName | null;
  /** The backdrop of the real fight here (practice is always in the training yard). */
  scene: Scene;
  pathAt: number;
  /** A scroll found after winning here (the boss). */
  reward?: ScrollId;
}

export const SCROLLS: Record<ScrollId, ScrollDef> = {
  flameShield: { id: 'flameShield', name: 'Flame Shield', moves: ['shield'], lessonId: 'shield' },
  risingPillar: { id: 'risingPillar', name: 'Rising Pillar', moves: ['palm'], lessonId: 'palm' },
  heldBreath: { id: 'heldBreath', name: 'Held Breath', moves: ['charge'], lessonId: 'charge' },
  burningWall: { id: 'burningWall', name: 'Burning Wall', moves: ['wall'], lessonId: 'wall' },
  finalFlame: { id: 'finalFlame', name: 'Final Flame', moves: ['finisher'], lessonId: 'ultimate' },
};

/** What you can do before finding any scroll. */
export const START_MOVES: MoveName[] = ['punch', 'flurry'];

export function movesFor(scrolls: ScrollId[]): Set<MoveName> {
  return new Set<MoveName>([...START_MOVES, ...scrolls.flatMap(id => SCROLLS[id].moves)]);
}

export const PATH_LENGTH = 300;

const spirit = (x: number, z: number, only: 'orb' | 'slab', pace: number) => ({ kind: 'spirit' as const, x, z, only, pace, cd: 1.5, hp: 2 });
const earth = (x: number, z: number, pace: number) => ({ kind: 'earth' as const, x, z, only: 'pillar' as const, pace, cd: 1.5, hp: 2 });

export const STOPS: StopDef[] = [
  {
    id: 'courtyard', place: 'Temple Courtyard', pathAt: 30, newMove: 'flurry',
    ren: ['The Spirit Moon rises, and I am far away. You must keep the Flame.', 'Fists up. Let the fire come from your breath.'],
    practice: ['move', 'punch', 'flurry'],
    fight: { goal: { type: 'defeat' }, groups: [
      { at: 0, enemies: [spirit(-20, 9, 'orb', 3.5)] },
      { at: 4, enemies: [spirit(15, 10, 'orb', 3.5), spirit(-5, 11, 'orb', 3.5)] },
    ] },
    scene: 'courtyard',
  },
  {
    id: 'stairs', place: 'The Long Stairs', pathAt: 80, scroll: 'flameShield', scrollAt: 68, newMove: 'shield',
    ren: ['Spirits on the stairs. When you cannot step aside, stand your ground.'],
    practice: ['shield'],
    fight: { goal: { type: 'defeat' }, groups: [
      { at: 0, enemies: [spirit(-15, 8, 'orb', 1.6), spirit(15, 8, 'orb', 1.9)] },
      { at: 8, enemies: [spirit(0, 9, 'slab', 3), spirit(-20, 10, 'orb', 1.8)] },
    ] },
    scene: 'stairs',
  },
  {
    id: 'bridge', place: 'Bamboo Bridge', pathAt: 135, scroll: 'risingPillar', scrollAt: 122, newMove: 'palm',
    ren: ["Daro's scouts. Earth moves slowly — read it, then answer with fire."],
    practice: ['pillar', 'palm'],
    fight: { goal: { type: 'defeat' }, groups: [
      { at: 0, enemies: [earth(0, 9, 3.2)] },
      { at: 5, enemies: [spirit(-10, 7, 'orb', 2.5), spirit(0, 9, 'orb', 2.5), spirit(10, 11, 'orb', 2.5)] },
      { at: 12, enemies: [earth(-15, 10, 3), earth(15, 9, 3.4)] },
    ] },
    scene: 'bridge',
  },
  {
    id: 'garden', place: 'Stone Garden', pathAt: 190, scroll: 'heldBreath', scrollAt: 176, newMove: 'charge',
    ren: ['Some spirits are old and hard. Hold your breath, gather your fire, then strike once.'],
    practice: ['charge'],
    fight: { goal: { type: 'defeat' }, groups: [
      { at: 0, enemies: [{ ...spirit(0, 8, 'orb', 2.2), hp: 3 }, { ...spirit(-20, 10, 'orb', 2.6), hp: 3 }] },
      { at: 10, enemies: [earth(15, 9, 3), { ...spirit(-10, 9, 'slab', 3), hp: 3 }] },
    ] },
    scene: 'garden',
  },
  {
    id: 'gate', place: 'The Village Gate', pathAt: 245, scroll: 'burningWall', scrollAt: 232, newMove: 'wall',
    ren: ['They are at the gate. Raise a wall and let nothing through.'],
    practice: ['wall'],
    fight: { goal: { type: 'survive', seconds: 45 }, groups: [
      { at: 0, enemies: [earth(-15, 9, 2.8), spirit(15, 8, 'orb', 2)] },
      { at: 15, enemies: [earth(15, 10, 2.6), spirit(-10, 9, 'orb', 1.8)] },
      { at: 30, enemies: [earth(0, 9, 2.4), spirit(-20, 8, 'slab', 3), spirit(20, 8, 'orb', 1.8)] },
    ] },
    scene: 'gate',
  },
  {
    id: 'daro', place: 'Daro Stonefist', pathAt: 262, newMove: null, reward: 'finalFlame',
    ren: ['Daro Stonefist. Strong, and slow to anger — and slower to tire. Break his wall. Stay out of his lanes.'],
    practice: [],
    fight: { goal: { type: 'defeat' }, boss: true, groups: [] },
    scene: 'boss',
  },
];

/** After Daro falls: the Final Flame scroll and a first try of the finisher. */
export const EPILOGUE = {
  lessonId: 'ultimate',
  ren: ['He falls back — but Kuzan will come himself now.', 'Take my last scroll. Bring your hands together and hold the fire between them… then let it fly.'],
};
