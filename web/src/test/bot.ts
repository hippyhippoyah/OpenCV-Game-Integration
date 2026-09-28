import { Game, type Enemy, type MoveName } from '../game/game';
import type { HandState, Intent } from '../intent/interpret';
import { FightRunner, type FightScript } from '../campaign/scripts';

const base: Intent = { present: true, head: { x: 0, y: 0 }, hands: { l: null, r: null }, shoulders: { l: { x: -20, y: 20 }, r: { x: 20, y: 20 } }, punches: [], palms: [], shield: false, xBlock: false, casts: [], face: null, bodyTilt: 0 };

/** Where an enemy appears on screen (view units), to aim at it. */
const onScreen = (g: Game, e: Enemy) => { const s = 3 / (3 + e.z); return { x: (e.x - g.cam.x) * s, y: (e.y - g.cam.y) * s }; };

/** A raised, open hand for the shield: only the fields the shield check reads matter. */
const shieldHand = (x: number, y: number): HandState =>
  ({ pos: { x, y }, vel: { x: 0, y: 0 }, openness: 1, open: true, facing: 1, palm: null, source: 'hand', inView: true, elbow: null, extension: null, punchReady: true, punchRise: null, reach: null, reachBase: null, reachNoise: null, aimDir: null, charge: 0 });

/**
 * A bot plays a fight on a scripted game `g` (already set up: allowed moves, temple, boss).
 * `true` (smart) dodges, punches, shields and walls as a decent player would; 'attack' only
 * attacks (never dodges or shields), like a player still learning; `false` stands still.
 */
export function botFight(g: Game, script: FightScript, moves: Set<MoveName>, smart: boolean | 'attack', seconds = 240): 'won' | 'lost' {
  const f = new FightRunner(g, script);
  let head = { x: 0, y: 0 };
  for (let t = 0; t < seconds; t += 1 / 60) {
    const i: Intent = { ...base, head, punches: [], palms: [], casts: [] };
    if (smart) {
      const dodges = smart === true, next = g.incoming()[0];
      if (dodges) {
        head = !next || next.safe ? head : next.kind === 'slab' ? { x: head.x, y: 20 } : { x: next.away * 30, y: 0 };
        if (!next) head = { x: head.x * 0.98, y: head.y * 0.9 };
      }
      const target = g.enemies.find(e => e.hp > 0);
      const frame = Math.round(t * 60);
      if (target && frame % 30 === 0) {
        const at = onScreen(g, target);
        const charged = moves.has('charge') && (target.hp >= 3 || !!target.boss) && frame % 60 === 0;
        i.punches = [{ hand: frame % 60 ? 'l' : 'r', at, shoulder: { x: 20, y: 20 }, dir: null, charged }];
      }
      if (moves.has('wall') && frame % 300 === 0) i.casts = [{ kind: 'wall', at: { x: 0, y: 10 } }];
      if (moves.has('palm') && target && frame % 90 === 45) i.palms = [{ kind: 'push', hand: 'r', at: onScreen(g, target), shoulder: { x: 20, y: 20 }, dir: null }];
      // Raise the shield when an orb is about to arrive: it needs both hands, and the bot's punches
      // don't use `hands` for anything, so this is free to hold up whenever it's useful.
      if (dodges && moves.has('shield')) {
        const soon = g.projs.some(p => p.kind === 'enemy' && p.z > 0 && p.z / -p.vz < 0.35);
        if (soon) { i.shield = true; i.hands = { l: shieldHand(head.x - 15, head.y), r: shieldHand(head.x + 15, head.y) }; }
      }
    }
    g.step(1 / 60, i);
    g.drainEvents();
    const out = script.boss ? (g.state === 'over' ? 'lost' : g.boss ? 'fighting' : 'won') : f.update(1 / 60);
    if (out !== 'fighting') return out;
  }
  return 'lost';
}
