import { STOPS } from './chapter1';
import type { Progress } from './progress';

/**
 * The temple you defend after Chapter 1: three kinds of defense, each built up to level 3 with
 * embers. The campaign pays exactly what maxing it costs (see EMBERS_PER_FLAME, CHAPTER_BONUS).
 */
export type BuildingId = 'brazierL' | 'brazierR' | 'wall' | 'shrine';

export interface BuildingDef {
  id: BuildingId;
  name: string;
  /** What it does, in a few words. */
  does: string;
  /** Cost of level 1, 2, 3. */
  costs: [number, number, number];
}

export const BUILDINGS: BuildingDef[] = [
  { id: 'brazierL', name: 'Left Brazier', does: 'Shoots fire at raiders', costs: [400, 800, 1400] },
  { id: 'brazierR', name: 'Right Brazier', does: 'Shoots fire at raiders', costs: [400, 800, 1400] },
  { id: 'wall', name: 'Temple Wall', does: 'Blocks some hits', costs: [500, 1000, 1700] },
  { id: 'shrine', name: 'Shrine of Ren', does: 'Hits hurt less', costs: [300, 500, 800] },
];
export const MAX_LEVEL = 3;

/** Embers each 🔥 flame earned in the campaign pays (once), and finishing Chapter 1. */
export const EMBERS_PER_FLAME = 400;
export const CHAPTER_BONUS = 2800;

/** What the whole temple costs to max out — and what a three-flame campaign pays. */
export const TEMPLE_MAX_COST = BUILDINGS.reduce((n, b) => n + b.costs.reduce((a, c) => a + c, 0), 0);

/** Embers the campaign has paid so far: its flames, and the chapter bonus. */
export function embersEarned(p: Progress): number {
  const flames = STOPS.reduce((n, s) => n + p.flames(s.id), 0);
  return flames * EMBERS_PER_FLAME + (p.data.chapterDone ? CHAPTER_BONUS : 0);
}

/** Embers spent on buildings. */
export function embersSpent(p: Progress): number {
  return BUILDINGS.reduce((n, b) => n + b.costs.slice(0, p.level(b.id)).reduce((a, c) => a + c, 0), 0);
}

/** Embers you can spend (never stored: always worked out from flames and buildings). */
export function embersAvailable(p: Progress): number {
  return Math.max(0, embersEarned(p) - embersSpent(p));
}

/** The temple opens once Chapter 1 is finished. */
export const templeOpen = (p: Progress): boolean => p.data.chapterDone;

/** What the next level of a building costs, or null when it's maxed. */
export function nextCost(p: Progress, id: BuildingId): number | null {
  const lvl = p.level(id), b = BUILDINGS.find(x => x.id === id)!;
  return lvl >= MAX_LEVEL ? null : b.costs[lvl];
}

/** Build the next level if you can afford it. Returns true if it was built. */
export function upgrade(p: Progress, id: BuildingId): boolean {
  const cost = nextCost(p, id);
  if (cost === null || cost > embersAvailable(p)) return false;
  p.setLevel(id, p.level(id) + 1);
  return true;
}

/** How the temple helps in a raid (see Game.temple). */
export interface TempleEffects {
  /** Braziers: where each stands (world x, depth), seconds between shots and damage per shot. */
  braziers: { x: number; z: number; cd: number; damage: number }[];
  /** Chance each hit on you is blocked outright by the temple wall. */
  blockChance: number;
  /** Damage you take per hit is multiplied by this (the shrine). */
  damageMult: number;
}

const BRAZIER = [null, { cd: 5, damage: 1 }, { cd: 4, damage: 1 }, { cd: 3, damage: 1 }] as const;
const BLOCK = [0, 0.15, 0.3, 0.45];
const SHRINE = [1, 0.85, 0.7, 0.55];
/** Where the braziers stand: by the gate on either side, just in front of you. */
export const BRAZIER_SPOTS: Record<'brazierL' | 'brazierR', { x: number; z: number }> = {
  brazierL: { x: -42, z: 3 },
  brazierR: { x: 42, z: 3 },
};

/** What the temple as built does in a raid. */
export function templeEffects(p: Progress): TempleEffects {
  const braziers = (['brazierL', 'brazierR'] as const).flatMap(id => {
    const b = BRAZIER[p.level(id)];
    return b ? [{ ...BRAZIER_SPOTS[id], cd: b.cd, damage: b.damage }] : [];
  });
  return { braziers, blockChance: BLOCK[p.level('wall')], damageMult: SHRINE[p.level('shrine')] };
}
