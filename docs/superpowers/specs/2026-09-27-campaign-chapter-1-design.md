# Campaign — Chapter 1: The Ember Path

## Goal

Replace "a list of lessons" with the start of a story. Chapter 1 is where a new player learns to
bend: they walk a mountain path, find **scrolls** (moves), and use each new move in a real fight,
ending in the first boss. It's the first step of the long-term loop:

1. **Campaign** (this doc) — story; moves unlocked as scrolls.
2. Embers and mastery — fights reward embers; used moves level up.
3. The temple — a home base you walk around (WASD + mouse), build and upgrade; waves become raids.
4. Asynchronous multiplayer — raid other players' temples, leaderboards.

Chapter 1 stands on its own: playable start to finish in 15–20 minutes, in short physical bursts
(each fight 1–3 minutes) with rests in between.

## The two halves: explore and fight

The game alternates between two ways of playing, and the switch between them is a designed moment:

- **Explore — keyboard and mouse, first person, in a 3D world.** In the campaign there's one way
  to go, so you **auto-walk** the path: the view glides along it on its own while you look around
  with the mouse. It stops where there's something to do — a scroll (E or click to pick it up), an
  arena (E or click to step in) — and walks on after. M for the map, Tab for scrolls, Space to skip
  ahead to the next stop. You sit at the computer; this is the rest between fights. (Free WASD
  walking comes with the temple, on the same 3D world code.)
- **Fight — webcam.** You stand back from the screen and bend. The view is the existing
  first-person fight (lean and duck to move; no WASD).

**Explore → fight.** Stepping into an arena's glowing circle shows a card: *"Step back and raise
your fists"*, with a live check — head and shoulders seen, hands up, at a good distance (too close
/ too far said plainly). Once the camera has you for about a second, a short countdown ("3, 2, 1")
starts the fight. Esc backs out to explore. If the camera loses you mid-fight, the fight pauses
(as now) with the same card.

**Fight → explore.** Winning shows the result card, then *"Take a seat — walk on when you're ready"*
and hands control back to keyboard and mouse. Losing offers "Try again" (the fight only).

A small **controls strip** in a corner always shows the keys for the current half (explore:
`Mouse` look · `E` interact · `Space` skip ahead · `M` map · `Tab` scrolls · `Esc` menu; fight:
`Esc` pause).

## Story

Avatar-flavoured, our own world.

You are the last acolyte of the **Ember Temple**, high on a mountain. Your master, **Ren**, is
away. The Spirit Moon is rising and the water spirits grow restless — and **General Kuzan** of the
earth clans uses the chaos to march on the temple for its Eternal Flame. His lieutenant, **Daro
Stonefist**, leads the vanguard up the mountain path.

Chapter 1 follows the path down from the temple courtyard to the village gate, where Daro waits.
Ren speaks through a spirit ember you carry (a line or two of text at key moments); the scrolls are
Ren's old teachings, left along the path for you.

## Scrolls (moves)

A **scroll** is a move: you have a move only once you've found its scroll. Scrolls are kept
(saved on the device) and listed in the **Scrolls** inventory (Tab), each with its name, how to do
it, and its animation (the lesson card's), so you can check any move you've learned.

| Scroll | Move | Found |
|---|---|---|
| (start) | Punch; leaning and ducking; flurry (just punching fast) | — |
| Flame Shield | Flame shield | Stop 2 |
| Rising Pillar | Palm push | Stop 3 |
| Held Breath | Charged punch (fist at the hip or at head level) | Stop 4 |
| Burning Wall | Fire wall | Stop 5 |
| Final Flame | Finisher (jab, jab, gather & fling) | Boss reward |
| — | One-two push, pillar volley, wall breaker | Later chapters |

Not in the campaign at all: the X block and the shield counter.

A move you haven't learned does nothing in a fight (no effect, no hint), so nothing fires by
accident while learning.

**Finding a scroll.** The scroll glows on a stand or ledge in the explore world, with a soft light
visible from a distance. Walking up to it and pressing E: the scroll unrolls (its move's name and
animation), and a notification slides in — *"New move learned: Flame Shield — scroll added to your
Scrolls (Tab)"*. The next fight is where you practise it.

## Learning a move in a fight: ghost hands

The first fight after finding a scroll starts with a short **practice**: a harmless target (a
dummy, or a spirit that doesn't attack), the lesson card at the top with its goal counter, and
**ghost hands** — translucent, faintly glowing hands drawn over your own that perform the move in
a slow loop, in the same place your real hands are. The ghost shows *where* and *how*: a fist
dropping to the hip and glowing, a palm shoved forward, both hands sweeping up. They fade as you
succeed (after the first success they appear only if you stall for a few seconds) and are gone
when the practice goal is met; then the real fight begins ("Now for real"). Moves you already know
can still show a ghost if a fight needs one and you haven't used it for a while.

The same ghost hands can later teach anything (the tutorial's lesson demos stay as the small
animation in the card and in the Scrolls inventory).

## The stops

The path is one continuous 3D place you walk down; each stop is an area with an arena circle.

**The world** (three.js, stylised and moody rather than realistic): a mountain at night under the
rising Spirit Moon, the path lit by lanterns (warm point lights) with fog in the valleys, drifting
embers and fireflies. The temple courtyard (red pillars, a brazier, training dummies), the long
stone stairs down the cliff, a bamboo bridge over mist, a stone garden (raked sand, boulders), and
the village gate (wooden palisade, torches, Daro's banners). Scrolls glow gold on stands, visible
from a distance; arena circles are rings of flame on the ground. Fights keep the existing fight
view, tinted to the place (its sky and light colours).

| # | Place | Scroll | Fight |
|---|---|---|---|
| 1 | **Temple courtyard** (dawn) | — | Practice: lean, duck, punch the dummies, a flurry. Then 3 slow water spirits whose orbs you lean away from |
| 2 | **The long stairs** | Flame Shield | Practice: block orbs. Then spirits in pairs throwing orbs too fast to only dodge; waves to duck |
| 3 | **Bamboo bridge** | Rising Pillar | Practice: pillar through two dummies. Then Daro's scouts: pillars down the bridge's sides, and a line of spirits to burn through |
| 4 | **Stone garden** | Held Breath | Practice: charged punches at a dummy. Then tougher spirits (3 hits) a charged punch drops in one, with earthbenders |
| 5 | **Village gate** | Burning Wall | Practice: raise walls. Then hold the gate: earthbenders and spirits together; pillars and orbs stop at your wall |
| 6 | **The gate — Daro Stonefist** | reward: Final Flame | Boss fight (below) |

Difficulty ramps across the stops (more enemies, faster attacks); each fight is 1–3 minutes.
You can replay a finished stop from the map.

## Boss: Daro Stonefist

A big earthbender with a health bar across the top of the screen and three phases.

- **Phase 1 (full → 60%)**: pillars down one side, then the other; every so often he raises a
  **stone wall** in front of himself that blocks fireballs — break it with a palm push or a
  charged punch.
- **Phase 2 (60 → 25%)**: **twin pillars** down both sides at once (stay in the middle), and a
  spinning **boulder at head height** (duck). Spirits join now and then.
- **Phase 3 (under 25%)**: faster, and after each big attack he's **winded** for 2 s (he glows and
  sags): hits land double.

You win by damage (you don't have the finisher yet). He staggers back to the gate; you find the
**Final Flame** scroll where he stood, and the epilogue is a short practice of the finisher — ghost
hands and all — on the retreating Daro, who flees down the valley. Ren: "Kuzan will come himself
now." End of Chapter 1.

Every attack has a clear tell (the existing wind-ups, furrow and dodge cue) and can be dodged or
blocked with the moves you have by then.

## Screens and feedback

- **Mode menu**: Campaign (new; "Continue" if started), Waves, Training, Tutorial (unchanged).
- **Explore view**: first person on the path; the controls strip; Ren's lines as subtitles at the
  bottom; a prompt when near something (`E` Pick up scroll · `E` Enter the arena).
- **Map (M)**: the mountain path from above, stops as lanterns — lit when done (with their flames),
  the next glowing, later ones dark; where you are. Selecting a finished stop's lantern offers a
  replay; the map is also shown briefly at the start of the campaign.
- **Scrolls (Tab)**: the scrolls you have, each opening to its move's card; locked ones as dark
  silhouettes ("Found later on the path").
- **Notifications**: slide in at the top right for a few seconds — new move learned, stop
  complete, flames earned.
- **Result card**: flames (finished; took little damage; used the new move or a combo), then
  "Walk on" (back to explore) or "Try again".
- **Boss health bar** at the top during the boss fight.

## How it works (code)

- `src/explore/` — the keyboard-and-mouse half: the path as a curve through the world with its
  stops (`path.ts`: position and heading at a distance along it, where it pauses), mouse look
  (pointer lock, yaw/pitch limits), and the three.js scene (`world3d.ts`: terrain, the places,
  lights, fog, sky, particles, scroll stands, arena rings; the camera rides the path).
  `three` is added as a dependency.
- `src/campaign/chapter1.ts` — data: the path's areas, Ren's lines, scroll positions, each stop's
  practice and fight scripts, flame rules.
- `src/campaign/runner.ts` — the campaign as a state machine: explore ⇄ (arena → camera check →
  countdown → practice → fight → result) with restarts, scroll pickups and unlocks. Pure logic,
  tested like the tutorial.
- `src/campaign/progress.ts` — scrolls found, stops done, flames, where you stood; saved in
  `localStorage` (wrapped in try/catch; a fresh start if unavailable).
- `src/campaign/scripts.ts` — fight scripts: timed groups of enemies (reusing `addEnemy`, `only`,
  `pace`) and goals ("defeat all", "hold out for…").
- **Game**: `Game.allowed` — the moves you have; punches, palms, casts and combos not in it are
  ignored. A `Boss` (an enemy with phases, a health bar, stone wall, twin pillars, boulder, winded).
- **Render**: ghost hands (the hand renderer, translucent, driven by a per-move keyframe loop), the
  explore view, map, scrolls, notifications, result card, boss and health bar.
- `main.ts` — a new phase for explore, the camera handoff card, and input switching (keyboard and
  mouse listeners only while exploring; the webcam tracker only needs to run in fights, but keeps
  running so the handoff check is instant).

## Testing

- Runner: explore → arena → check → countdown → practice → fight → result → explore; losing
  restarts only the fight; scroll pickups unlock and save; replays.
- Gating: a locked move never reaches the game; unlocking lets it through.
- Scripts: each stop's fight can be won with only the moves you have by then (a scripted "player"
  in tests), and is lost by standing still where it's meant to make you move.
- Boss: phase changes at 60% / 25%; the stone wall blocks fireballs but not palm pushes or charged
  punches; twin pillars leave the middle safe; winded doubles damage.
- Explore: position along the path, pausing at scrolls and arenas, skipping ahead, look limits
  (logic only, no WebGL).

## Not in Chapter 1

Embers, mastery, the temple (built on the same explore walker later), other scrolls, voice or
audio, multiplayer. Waves, Training and the Tutorial are unchanged.
