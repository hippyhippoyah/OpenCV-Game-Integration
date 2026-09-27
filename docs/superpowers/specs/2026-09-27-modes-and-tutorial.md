# Modes and tutorial

After calibrating (or starting with mouse & keys) you choose a mode; `Esc` in game returns to the
choice. `?mode=tutorial|waves|training` skips it (`?dummies` = training).

- **Tutorial** — ten lessons, one move each, in teaching order. Each has a how-to, a goal with a
  counter, and its own scripted enemies (tutorial enemies are sturdy and come back if knocked
  down; dummies respawn). You can't lose: attacks land and show, but cost no health. A lesson
  completes when its goal is met, celebrates for 2 s, then the next begins. `N` skips, `B` goes
  back; the menu can also jump straight to any lesson.
- **Waves** — the endless spirits and earthbenders (unchanged).
- **Training** — practice dummies that never fight back (unchanged). `T` still swaps between waves
  and training in game.

| # | Lesson | Enemies | Goal |
|---|--------|---------|------|
| 1 | Move | — | lean left, lean right, duck (view offset ≥ 20 sideways / ≥ 15 down) |
| 2 | Punch | a dummy | 5 hits |
| 3 | Dodge a pillar | an earthbender (pillars every 2.5 s) | 2 dodges |
| 4 | Duck the wave | a spirit (waves every 2 s) | 2 dodges |
| 5 | Flame shield | a spirit (orbs every 1.2 s) | 3 blocks with the shield up |
| 6 | X block | a spirit (orbs) | 3 blocks with the X block |
| 7 | Palm push | two dummies | 3 pillars |
| 8 | Fire wall | a spirit (orbs) | 2 walls |
| 9 | Wall push | two dummies | 2 wall pushes |
| 10 | Ultimate | three dummies (ultimate charged) | 1 ultimate |

Implementation: `src/game/tutorial.ts` (`LESSONS`, `Tutorial`), driven from `main.ts` with each
frame's game events; `Game.scripted()`, `addEnemy()`, `noDamage`, per-enemy `only` (attack) and
`pace` (seconds between attacks) let a script run the field.

## Revision: lesson card with move animations

The lesson card sits at the top centre (it replaces the move-hint pill during the tutorial) with
larger text, and shows a small looping animation of the move (`src/render/lessonDemo.ts`):
stylised hands doing the gesture with arrows — a fist snapping out, a palm shoved forward, palms
sweeping up, gathering and flinging — or, for the dodges, a figure leaning away from a pillar or
ducking under a wave. Lessons made of separate steps (Move) show each as a chip that turns green
with a tick when done.
