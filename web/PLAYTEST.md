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
- [ ] Finisher: open hands held together (like catching a football) catch fire in about half a second, then spreading them wide casts it — reliably, even slowly. While recharging it says how long.
- [ ] Hands held close together never raise the shield; the shield still comes up with hands about shoulder width apart.
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

- [ ] Campaign: the map looks good, your token walks the path smoothly, Space skips ahead.
- [ ] Practice is always in the training yard; each real fight is in its own place (courtyard, stairs, bridge, garden, gate, Daro's gate).
- [ ] Scrolls: picking one up shows the note; Tab lists it with its animation.
- [ ] The camera handoff is clear: the checklist tells you what's missing; the countdown starts once you're ready.
- [ ] Ghost hands make each new move obvious; they fade once you've got it.
- [ ] Every fight is winnable with the moves you have, and the boss's attacks are readable.
- [ ] Clicking a lit stop on the map replays it.
- [ ] Resting fists look close to your body; they only reach out when you punch.
- [ ] Breath: jabbing steadily never runs out; spamming walls or a long flurry does, and the attack fizzles with a hint. The balance feels fair.
- [ ] Vitality reads as health (green); Breath (orange, top centre) is the first thing you look at; the finisher icon beside it is clear when ready.
- [ ] Blue inferno: hands together over the head swell a blue fireball; slamming them down sends a line of blue flame; spreading them after sets the ground ablaze. Holding them up there doesn't gather the finisher or punch.
- [ ] The one-two push's pillar is blue.
- [ ] The title and menu look like a finished game; keyboard and mouse both work; the camera is only asked for when you start a mode.
- [ ] Sound: jabs feel punchy and vary; big moves feel big; nothing is too loud or grating over a long session. Volume in Settings works.
- [ ] Holding the shield: it crackles, and you can see enemies and attacks through its flames. Charging either ultimate has a warm-up swell.
