import type { FightScript, ScriptEnemy } from './scripts';

/** A raid on the temple: a fight you pick from the ladder, defended by what you've built. */
export interface RaidDef { id: string; name: string; fight: FightScript }

const spirit = (x: number, z: number, only: 'orb' | 'slab', pace: number, hp = 2): ScriptEnemy => ({ kind: 'spirit', x, z, only, pace, cd: 1.5, hp });
const earth = (x: number, z: number, pace: number, hp = 2): ScriptEnemy => ({ kind: 'earth', x, z, only: 'pillar', pace, cd: 1.5, hp });

/** The raid ladder: each one opens when the one before it is won, and each is harder. */
export const RAIDS: RaidDef[] = [
  {
    id: 'r1', name: 'First Night',
    fight: { goal: { type: 'defeat' }, groups: [
      { at: 0, enemies: [spirit(-15, 9, 'orb', 3), spirit(15, 9, 'orb', 3.2)] },
      { at: 6, enemies: [spirit(0, 10, 'orb', 2.8), spirit(-25, 11, 'slab', 3.5)] },
    ] },
  },
  {
    id: 'r2', name: 'Stone and Water',
    fight: { goal: { type: 'defeat' }, groups: [
      { at: 0, enemies: [earth(0, 9, 3), spirit(-20, 8, 'orb', 2.6)] },
      { at: 7, enemies: [spirit(20, 9, 'slab', 3), earth(-15, 10, 2.8)] },
      { at: 14, enemies: [spirit(0, 8, 'orb', 2.4), spirit(25, 10, 'orb', 2.4)] },
    ] },
  },
  {
    id: 'r3', name: 'The Long Night',
    fight: { goal: { type: 'survive', seconds: 50 }, groups: [
      { at: 0, enemies: [spirit(-20, 9, 'orb', 2.4), earth(15, 9, 2.8)] },
      { at: 15, enemies: [spirit(10, 8, 'slab', 2.8), spirit(-5, 11, 'orb', 2.2)] },
      { at: 30, enemies: [earth(-15, 10, 2.5), spirit(20, 9, 'orb', 2.2)] },
    ] },
  },
  {
    id: 'r4', name: 'Rising Tide',
    fight: { goal: { type: 'defeat' }, groups: [
      { at: 0, enemies: [spirit(-15, 8, 'orb', 1.8, 4), spirit(15, 8, 'slab', 2.2, 4), spirit(0, 11, 'orb', 1.9, 4)] },
      { at: 6, enemies: [spirit(-25, 9, 'slab', 2, 4), spirit(25, 9, 'orb', 1.8, 4)] },
      { at: 12, enemies: [spirit(-10, 9, 'orb', 1.7, 4), spirit(10, 9, 'orb', 1.7, 4), spirit(0, 10, 'slab', 2, 4)] },
    ] },
  },
  {
    id: 'r5', name: 'Earth Clan',
    fight: { goal: { type: 'defeat' }, groups: [
      { at: 0, enemies: [earth(-15, 9, 1.9, 4), earth(15, 10, 2, 4), spirit(0, 8, 'orb', 1.8, 4)] },
      { at: 6, enemies: [earth(-25, 11, 1.8, 4), spirit(20, 8, 'orb', 1.7, 4)] },
      { at: 12, enemies: [earth(20, 9, 1.8, 4), spirit(-10, 10, 'slab', 1.9, 4), spirit(10, 8, 'orb', 1.6, 4), earth(0, 11, 1.9, 4)] },
    ] },
  },
  {
    id: 'r6', name: "Kuzan's Vanguard",
    fight: { goal: { type: 'survive', seconds: 75 }, groups: [
      { at: 0, enemies: [earth(-15, 9, 1.8, 5), spirit(15, 8, 'orb', 1.6, 5), spirit(0, 11, 'slab', 1.9, 5)] },
      { at: 10, enemies: [earth(20, 10, 1.7, 5), spirit(-25, 9, 'orb', 1.5, 5)] },
      { at: 20, enemies: [spirit(-10, 8, 'orb', 1.5, 5), earth(0, 10, 1.7, 5), spirit(25, 9, 'slab', 1.8, 5)] },
      { at: 32, enemies: [earth(-20, 10, 1.6, 5), spirit(10, 8, 'orb', 1.4, 5), spirit(-5, 11, 'orb', 1.5, 5)] },
      { at: 45, enemies: [earth(15, 9, 1.6, 5), spirit(-15, 8, 'slab', 1.7, 5), spirit(20, 10, 'orb', 1.4, 5)] },
      { at: 58, enemies: [spirit(0, 8, 'orb', 1.3, 5), earth(-10, 10, 1.5, 5), spirit(15, 9, 'orb', 1.3, 5)] },
    ] },
  },
];
