# Campaign — Chapter 1: The Ember Path

## Goal

Replace "a list of lessons" with the start of a story. Chapter 1 is where a new player learns to
bend: each stop along a journey teaches one move (by finding its **scroll**) and then tests it in a
real fight, ending in the first boss. It's also the first step of the long-term loop:

1. **Campaign** (this doc) — story, moves unlocked as scrolls.
2. Embers and mastery — fights reward embers; used moves level up.
3. The temple — build and upgrade a home base between fights; waves become raids on it.
4. Asynchronous multiplayer — raid other players' temples, leaderboards.

Chapter 1 has to stand on its own: playable start to finish, fun in 15–20 minutes, and short
bursts (each stop is 1–3 minutes, since the game is physical).

## Story

Avatar-flavoured, our own world.

You are the last acolyte of the **Ember Temple**, high on a mountain. Your master, **Ren**, is away.
The Spirit Moon is rising — water spirits grow restless — and **General Kuzan** of the earth clans
uses the chaos to march on the temple for its Eternal Flame. Kuzan's lieutenant, **Daro Stonefist**,
leads the vanguard up the mountain path.

Chapter 1 follows the path down from the temple courtyard to the village gate, where Daro waits.
Ren speaks through a spirit ember you carry (a short line of text at each stop); scrolls are
Ren's old teachings, hidden along the path.

Better story ideas are welcome; the structure below doesn't depend on the details.

## Scrolls (moves)

A **scroll** is a move. You have a move only once you've found its scroll. Scrolls are collected
in the campaign and kept (saved on the device). Waves and Training use all moves for now (they
become part of the temple later).

| Scroll | Moves it teaches | Found in Chapter 1 |
|---|---|---|
| (start) | Punch, leaning & ducking | — you start with these |
| Scroll of the Flame Shield | Flame shield | Stop 2 |
| Scroll of the Rising Pillar | Palm push | Stop 3 |
| Scroll of the Held Breath | Charged punch (hip / ear) | Stop 4 |
| Scroll of the Burning Wall | Fire wall | Stop 5 |
| Scroll of the Final Flame | Finisher (jab, jab, gather & fling) | Boss reward |
| Flurry | Flurry | Stop 1 (it's just punching fast — taught, no scroll) |
| — | One-two push, Pillar volley, Wall breaker, X block, counter | Not in Chapter 1 (later chapters / temple) |

A move you haven't learned does nothing (no effect, no hint), so nothing fires by accident while
you're learning. When a scroll is found, the game shows it opening, and its lesson (the existing
lesson card with its animation) plays as a short practice before the fight goes on.

## Structure of a stop

Every stop runs the same steps:

1. **Travel** (5–8 s, skippable): the view glides down the path (on rails) through scenery to the
   next place; nothing to do but watch and rest.
2. **Arrival**: a one- or two-line message from Ren; the place's name as a banner.
3. **Scroll** (if the stop has one): the scroll is found and opens; the lesson card teaches the
   move with a practice goal against a dummy or a harmless spirit (as in the tutorial).
4. **Fight**: a scripted fight that needs the new move (and earlier ones). You can lose: at 0
   health the stop restarts from the fight (not the travel or the scroll).
5. **Result**: up to 3 flames (stars) — finished; took little damage; used the new move / a combo
   — then back to the map.

## The stops

| # | Place | Scroll | Fight |
|---|---|---|---|
| 1 | **Temple courtyard** (dawn) | — (teaches punch, leaning, flurry) | Dummies, then 3 slow water spirits that throw orbs you lean away from |
| 2 | **The long stairs** | Flame Shield | Spirits in pairs throwing orbs faster than you can only dodge; waves you duck |
| 3 | **Bamboo bridge** | Rising Pillar | Daro's earthbender scouts: pillars down the bridge's sides; a line of spirits to burn through with a pillar |
| 4 | **Stone garden** | Held Breath | Tougher spirits (3 hits) that charged punches drop in one; mixed with earthbenders |
| 5 | **Village gate** | Burning Wall | Hold the gate: a wave from both earthbenders and spirits; pillars and orbs stop at your wall |
| 6 | **The gate — Daro Stonefist** | reward: Final Flame | Boss fight (below) |

Difficulty ramps across the stops (more enemies, faster attacks), and each fight is small enough
to finish in 1–3 minutes.

## Boss: Daro Stonefist

A big earthbender with a health bar across the top of the screen and three phases.

- **Phase 1 (full → 60%)**: pillars down one side, then the other; every so often he raises a
  **stone wall** in front of himself that blocks fireballs — break it with a palm push or a charged
  punch.
- **Phase 2 (60 → 25%)**: **twin pillars** down both sides at once (stay in the middle), and a
  spinning **boulder at head height** (duck). Spirits join now and then.
- **Phase 3 (under 25%)**: faster, and after each big attack he's **winded** for 2 s (he glows and
  sags) — hits land double, and a finisher (if you have it) ends him. In Chapter 1 you don't have
  the finisher yet, so you win by damage; beating him *gives* the Scroll of the Final Flame, and a
  short epilogue lets you try it on him as he retreats.

Each attack has a clear tell (the existing wind-ups, furrow and dodge cue), and every one can be
dodged or blocked with the moves you have by then.

## Screens

- **Mode menu**: Campaign (new), Waves, Training, and **Training grounds** (the current tutorial
  lessons, for replaying any move you've unlocked).
- **Map**: the mountain path from the temple down to the gate, stops as lanterns — lit when done
  (with their flames), the next one glowing, later ones dark. Pick a stop to play (or replay).
  Keyboard/mouse (it's a rest moment), and a raised open palm held over a lantern selects it too.
- **Scroll**: the scroll unrolls with the move's name and animation.
- **Result**: flames earned, and "Continue" to the map.
- **Boss health bar** at the top during the boss fight.

## How it works (code)

- `src/campaign/chapter1.ts` — data: stops (place, message, scroll, fight script, flame rules).
- `src/campaign/runner.ts` — the stop's steps as a state machine: travel → arrival → scroll →
  fight → result; restart on loss. Pure logic, tested like the tutorial.
- `src/campaign/progress.ts` — scrolls found, stops done and flames; saved in `localStorage`
  (wrapped in try/catch; a fresh start if unavailable).
- `src/campaign/scripts.ts` — fight scripts: timed groups of enemies (reusing `addEnemy`, `only`,
  `pace`), and "hold out until…" / "defeat all" goals.
- **Game**: `Game.allowed` — the set of moves you have; punches, palms, casts and combos not in it
  are ignored. A `Boss` (an enemy with phases, a health bar, the stone wall and twin pillars, the
  boulder, the winded window).
- **Render**: travel scenery (a camera glide along the path, reusing the parallax layers with
  per-place tints and props), the map, scroll and result cards, the boss and health bar.
- The tutorial's lessons become the scroll lessons and Training grounds (their `LESSONS` data is
  reused; the lesson card and animations are unchanged).

## Testing

- Runner: every step order; losing restarts the fight; results and flames; unlocks saved.
- Gating: a locked move never reaches the game; unlocking lets it through.
- Scripts: each stop's fight can be won with only the moves you have by then (a scripted
  "player" in tests), and loses if you stand still in the ones meant to make you move.
- Boss: phase changes at 60% / 25%; the stone wall blocks fireballs but not palm pushes or charged
  punches; twin pillars leave the middle safe; winded doubles damage.

## Not in Chapter 1

Embers, mastery, the temple, other scrolls (one-two push, pillar volley, wall breaker, X block),
voice or audio, multiplayer. Waves and Training keep every move.

## Open questions

1. Should Waves and Training also be limited to scrolls you've found, from the start? (Proposed:
   not yet — they stay "everything unlocked" until the temple arrives.)
2. The X block isn't in the campaign (you may remove it) — keep it in Waves/Training for now?
3. Map selection by raising a palm, or mouse/keys only?
