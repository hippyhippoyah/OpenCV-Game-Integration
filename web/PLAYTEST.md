# Playtest checklist

Run `npm run dev`, play with the camera in Chrome. Note the result of each item.

- [ ] Frame rate (corner panel) stays ≥ 30 fps while playing.
- [ ] After calibrating, the mode menu offers Tutorial, Waves and Training; Esc gets back to it.
- [ ] The tutorial: each lesson's instructions make sense on their own, its goal is reachable, and it moves on by itself.
- [ ] Fire follows your hands without noticeable lag.
- [ ] Fists read as FIST and open hands as OPEN in the corner panel at 1 m, 1.5 m and 2.5 m.
- [ ] At about 1.5 m, fist punches (default) fire every time — jabs straight at the camera and crosses —
      and the corner bar for that fist crosses its orange mark. Note the ±cm wobble shown there.
- [ ] Standing, weaving and leaning in guard never fire. The on-screen fists don't jitter.
- [ ] The aim reticle sits on the enemy you'd hit, and punches go there.
- [ ] Standing ~2.5 m away shows "step closer" rather than firing phantom punches.
- [ ] Crossing your forearms raises the X block; a single cross punch doesn't.
- [ ] Reaching slowly, stretching, or holding an arm out does not fire; pulling back re-arms (dot lights).
- [ ] With `P` (open-hand style), a punch that opens at the end fires every time.
- [ ] Opening a still fist does not fire; the shot goes roughly where your hand was when it opened.
- [ ] Opening both hands raises the shield without also firing a punch; it blocks an attack you cover.
- [ ] Sweeping both open hands up quickly raises a fire wall where they are; slow raises don't.
- [ ] Gathering both open hands together and flinging them apart fires the ultimate; opening while already apart doesn't.
- [ ] Pushing both open palms at the camera (or pushing out of the shield) rolls a fire wall — never the ultimate.
- [ ] An earthbender's pillar clearly comes down your left or right (its furrow shows the track), and leaning away dodges it with time to spare.
- [ ] The dodge cue turns green (✓ CLEAR) as soon as you're out of the way, for both pillars and the water wave.
- [ ] Nothing flashes or shakes enough to startle you.
- [ ] Holding a fist at your hip (or cocked by your ear) turns it blue within about half a second; the next punch is a blue fireball that flies at an enemy. It never charges on its own in guard, while punching, swaying or with arms hanging.
- [ ] Three quick punches make a flurry; jab-jab-push makes a wide pillar; push-push (one hand each) makes a pillar volley.
- [ ] Raising a wall then pushing both palms breaks it forward; pushing both palms otherwise does nothing (with a hint).
- [ ] Blocking with the shield then punching throws a counter that homes in.
- [ ] Jab, jab, gather & fling fires the finisher when the ultimate bar is full; the fling alone says how.
- [ ] Enemies stay near the middle of the screen, where they're easy to aim at.
- [ ] A high sweep: a small duck gets under it.
- [ ] The view's lean and tilt feel big enough to play with, but not nauseating.
- [ ] Pushing one open palm at the camera (other hand a fist) rolls a pillar of fire forward — every time,
      including short pushes and pushes where the hand opens on the way. Holding a palm open fires nothing.
- [ ] Open hands at rest show no fire; fire appears only on punches, shield and casts.
- [ ] Palm "face" reads high with palms toward the camera and low with palms facing each other.
- [ ] Leaning and ducking dodge attacks aimed at your head.
- [ ] Walking out of frame pauses with "Step into frame".
- [ ] A 3-minute run is fun and doesn't wear out your arms.

Tuning knobs: `TUNING` in `src/intent/interpret.ts` (tracking feel), `TUNE` in `src/game/game.ts` (gameplay). Press `` ` `` in game for live numbers.

- [ ] Campaign: the walk down the path looks good and runs smoothly; mouse look feels natural; Space skips ahead.
- [ ] Scrolls: picking one up shows the note; Tab lists it with its animation.
- [ ] The camera handoff is clear: the checklist tells you what's missing; the countdown starts once you're ready.
- [ ] Ghost hands make each new move obvious; they fade once you've got it.
- [ ] Every fight is winnable with the moves you have, and the boss's attacks are readable.
- [ ] M map shows your progress and replays finished stops.
