# Playtest checklist

Run `npm run dev`, play with the camera in Chrome. Note the result of each item.

- [ ] Frame rate (corner panel) stays ≥ 30 fps while playing.
- [ ] Fire follows your hands without noticeable lag.
- [ ] Palms together makes fire at 1 m, 1.5 m and 2.5 m from the camera.
- [ ] A quick push throws; slowly moving hands toward the camera does not.
- [ ] Spreading hands raises the shield; it blocks an attack you cover.
- [ ] Leaning and ducking dodge attacks aimed at your head.
- [ ] Walking out of frame pauses with "Step into frame".
- [ ] A 3-minute run is fun and doesn't wear out your arms.

Tuning knobs: `TUNING` in `src/intent/interpret.ts` (tracking feel), `TUNE` in `src/game/game.ts` (gameplay). Press `` ` `` in game for live numbers.
