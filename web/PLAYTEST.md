# Playtest checklist

Run `npm run dev`, play with the camera in Chrome. Note the result of each item.

- [ ] Frame rate (corner panel) stays ≥ 30 fps while playing.
- [ ] Fire follows your hands without noticeable lag.
- [ ] Fists read as FIST and open hands as OPEN in the corner panel at 1 m, 1.5 m and 2.5 m.
- [ ] Fist punches (default) fire every time — straight at the camera, a cross, and a hook — and the
      arm meter in the corner crosses the orange mark on each one.
- [ ] Reaching slowly, stretching, or holding an arm out does not fire; pulling back re-arms (dot lights).
- [ ] With `P` (open-hand style), a punch that opens at the end fires every time.
- [ ] Opening a still fist does not fire; the shot goes roughly where your hand was when it opened.
- [ ] Opening both hands raises the shield without also firing a punch; it blocks an attack you cover.
- [ ] Sweeping both open hands up quickly raises a fire wall where they are; slow raises don't.
- [ ] Spreading both open hands apart quickly fires the ultimate; opening while already apart doesn't.
- [ ] Open hands at rest show no fire; fire appears only on punches, shield and casts.
- [ ] Palm "face" reads high with palms toward the camera and low with palms facing each other.
- [ ] Leaning and ducking dodge attacks aimed at your head.
- [ ] Walking out of frame pauses with "Step into frame".
- [ ] A 3-minute run is fun and doesn't wear out your arms.

Tuning knobs: `TUNING` in `src/intent/interpret.ts` (tracking feel), `TUNE` in `src/game/game.ts` (gameplay). Press `` ` `` in game for live numbers.
