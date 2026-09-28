# The Temple: base building + raids — Design

**Status:** approved for planning (chat design, 2026-09-28). This doc is the spec; an implementation
plan follows from `superpowers:writing-plans`.

## Summary

After finishing Campaign Chapter 1, a new **Temple** opens from the main menu. Finishing the
campaign (and earning its 🔥 flames) pays a fixed total of **embers**. You spend embers building up
the temple's three defenses. A **raid ladder** (6 raids, harder than the last) lets you test your
build in a real fight, at the temple, defended by whatever you've built. Raids don't pay embers yet
— that, and an ember-earning loop, is future work; this slice is deliberately closed: earn from the
campaign, spend once, done.

## Embers

- **Flat value per flame: 400.** Each of the 6 campaign stops can earn up to 3 flames, so 18 flames
  total = 7,200 embers, paid the first time each flame is earned (going from 🔥🔥 to 🔥🔥🔥 on a
  replay pays the difference; re-earning a flame you already have pays nothing).
- **Chapter completion bonus: 2,800**, paid once, when `progress.chapterDone` is first true.
- **Total available: 10,000** — exactly what maxing every building costs (see below). Three-flame
  the whole campaign and you can afford everything; nothing to grind, nothing left over.
- **No stored balance.** `embersEarned(progress)` is computed from `progress.data` (flames +
  completion) each time it's needed; `embersSpent(temple)` from the temple's building levels.
  `available = earned - spent`. This can't desync and doesn't need its own save-version migration
  logic beyond what `Progress`/`Temple` already have.
- **Losing a raid costs nothing.** No repairs, no upkeep — a fixed 10,000-ember budget can't support
  a repair cost without risk of getting stuck, unable to afford the thing you need.

## Buildings

Fixed spots at the temple (matching the "campaign is a placeholder Ember Temple" flavor already in
`chapter1.ts`), each with 3 levels, each level a flat one-time cost:

| Building | L1 | L2 | L3 | Total | Effect (raids only) |
|---|---:|---:|---:|---:|---|
| Left Brazier | 400 | 800 | 1400 | 2,600 | Auto-fires at the nearest enemy every `cd`s for `dmg` (both scale with level) |
| Right Brazier | 400 | 800 | 1400 | 2,600 | Same, mirrored |
| Temple Wall | 500 | 1,000 | 1,700 | 3,200 | Chance per incoming attack to block it outright |
| Shrine of Ren | 300 | 500 | 800 | 1,600 | Reduces damage taken per hit |

**Total to max everything: 10,000** (matches the embers total above by construction).

Levels are bought in order (can't buy L2 before L1); a level, once bought, is permanent (no selling,
no repairs — nothing to desync with the no-stored-balance design above).

### Effect numbers

- **Brazier:** L1 `cd=4s, dmg=1`; L2 `cd=3s, dmg=1`; L3 `cd=2s, dmg=2`. Targets the nearest living,
  non-dummy enemy within range (the whole field — the courtyard isn't large). Fires a small ember
  bolt (reuse the existing pillar/burn damage path) and emits an event for a sound and a flash at
  the brazier so it reads as "the temple is helping."
- **Temple Wall:** block chance per hit `L1 15%, L2 30%, L3 45%`, checked once per incoming hit
  (`hurt()`), independently of the player's own shield/X-block/wall. A blocked hit fires the
  existing `blocked` event instead of `playerHit` — no new event type needed.
- **Shrine:** damage-taken multiplier `L1 ×0.85, L2 ×0.7, L3 ×0.55`, applied to `TUNE.hitDamage` in
  `hurt()`. (Not extra max HP — HP is drawn as a raw 0–100 percentage in the HUD today, so changing
  the cap would need a second HUD change; a damage multiplier keeps that invariant untouched and is
  simpler to reason about.)

All three default to "no effect" (brazier level 0 = no brazier, wall/shrine level 0 = 0% / ×1) and
only apply when a `TempleEffects` config is passed into a `Game` — a normal campaign or tutorial
`Game` behaves exactly as it does today.

## Raids

A ladder of 6 raids (`RAID_1` … `RAID_6`), reusing the existing generic `FightRunner`/`FightScript`
machinery from `campaign/scripts.ts` as-is (it already takes a `Game` and a script and doesn't know
about the campaign specifically). Each raid:

- Is a `FightScript` (`groups`, `goal: { type: 'defeat' }`), harder than the last: more enemies,
  earlier earthbenders, tighter timing — same shape as the campaign's own fights, just without a
  practice phase, scroll, or Ren dialogue.
- Is fought first-person, at the temple, with whatever's currently built applied as a
  `TempleEffects` config on the `Game` (built live from the player's current `Temple` levels each
  time a raid starts — buildings aren't locked in per-raid, so building more between attempts helps
  immediately).
- Uses the existing `night` scene (already commented as "the waves keep the night temple" in
  `scenes.ts` — the temple-at-night backdrop already exists and is otherwise unused once Waves is
  gone).
- Is scored in 🔥 flames (0–3, same `reasons`-based scheme as campaign stops: finished / took little
  damage / — a raid has no "new move" reason, so its third flame is "beat it without the shield
  ever dropping" or similar, chosen per raid) — kept for feedback and future-proofing (a later patch
  can pay embers per raid flame without a data-model change), but **raids pay no embers now**.
- Progress (`raidsCleared: Record<string, { flames: number }>`) is a new, separate field alongside
  campaign `Progress`, structurally identical to `stops` — best flames kept, like campaign stops.

Losing a raid: no penalty, try again immediately (same as a campaign fight's "lost" card).

## Screens

- **Main menu:** a new "Temple" entry, shown only once `progress.data.chapterDone`. Its detail panel
  shows embers available, and a one-line reminder if the campaign has more flames to earn.
- **The Temple screen** (`?mode=temple`): a single 2D scene (parchment-map style, reusing
  `explore/map2d.ts`'s rendering conventions but static, no path) showing the temple with its three
  building spots. Clicking a spot with available embers and the next level affordable buys it
  (confirmed inline, not a separate dialog — it's cheap and reversible only in the sense that it's a
  one-way permanent purchase, which is stated on the spot). A "Raids" button opens the raid ladder
  (a vertical list like the tutorial's lesson chips: locked/cleared/flames per raid), matching the
  existing lit/unlit lantern language from the campaign map.
- **A raid fight** uses the same fight HUD as any other fight; the only new visual is the two
  braziers (static glowing shapes at their fixed screen positions, flaring when they fire) and, if
  built, a temple wall silhouette. No new HUD chrome.

## What this explicitly doesn't do (future work, not this slice)

- Raids paying embers, or any other post-campaign ember source.
- Building choice/placement (buildings are fixed slots, per the approved answer).
- Repairs, upkeep, or losing levels.
- An endless/night-mode raid after the ladder.
- Multiplayer/async raids.

## Self-review

- **Placeholders:** none — every number above is concrete.
- **Consistency:** embers-earned (10,000) and embers-to-max (10,000) match by construction; the
  ember/flame accounting has no stored balance, so no desync path.
- **Scope:** one coherent slice — earn once, build, fight a fixed ladder. Decomposition not needed.
- **Ambiguity resolved:** "max the temple" requires every campaign flame (my call per the open
  question in chat, confirmed acceptable); HP-bar-as-percentage stays untouched by using a damage
  multiplier instead of extra max HP for the shrine.
