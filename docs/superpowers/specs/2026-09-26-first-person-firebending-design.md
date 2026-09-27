# First-Person Firebending (webcam) — Design

Date: 2026-09-26 · Status: approved in conversation · Visual reference: `mockup/index.html`

## Goal

A browser game playable with only a laptop webcam. You see through your own eyes: your real
head position moves the camera, your real hands appear as fire-hands, and water-spirits throw
attacks at you. The first playable version answers one question: **does making, throwing and
shielding with fire using your real hands feel great?**

## Scope (v1)

In:
- **Fireball** — raise hands, palms close together → fire appears between them. Push toward the
  screen → it flies at the spirit nearest to where you aimed.
- **Flame shield** — hands spread wide → a wall of flame between them blocks attacks. Drains an
  energy bar while held; breaks at zero and is unusable for 1.2 s; recharges when lowered.
  Spreading while holding a fireball converts it into the shield.
- **Dodge** — spirits aim at your head/chest at throw time. Leaning or ducking moves the camera
  (and your hitbox). Landing rings show where attacks will arrive; red = will hit you.
- **Drop** — hands low → fire goes out (rest pose).
- Endless waves of spirit enemies, health, score, wave banner, game over + restart.
- Calibration screen, mock input mode (mouse/keys), debug overlay.

Out (later): whip, breath/mic charging, launch/dash mobility, crits, other elements, bosses,
multiplayer, saves, mobile, segmentation-mask hitbox, input recording.

Success criteria:
- ≥ 30 fps in Chrome on the developer's MacBook with tracking on.
- Fire visibly follows the hands without noticeable lag.
- Summon, throw and shield trigger reliably at 1–2.5 m from the camera.
- A 3-minute run is fun and doesn't exhaust your arms.

## Architecture

Vite + TypeScript in `web/`. Canvas 2D rendering (ported from the mockup).
`@mediapipe/tasks-vision` on GPU: **PoseLandmarker (lite)** for head + shoulders,
**HandLandmarker** (2 hands) for hands. No segmentation in v1.

```
CameraTracker ─┐                         
               ├─► TrackingFrame ─► interpretIntent() ─► Game.step() ─► Renderer
MockTracker ───┘   (normalized)        (pure)            (pure)         (canvas + HUD)
```

### Units
All intent geometry is expressed in **shoulder widths (sw)** relative to the calibrated body,
so it works at any distance from the camera. Screen mapping happens only at the edge.

### Modules
- `src/input/types.ts` — `TrackingFrame { t, head?, shoulderL?, shoulderR?, hands: Hand[] }`,
  points in mirrored normalized video coords (0..1, x flipped so moving right moves right on screen).
  `Hand { side, center, wrist, size, openness }`.
- `src/input/camera.ts` — getUserMedia + MediaPipe → `TrackingFrame` per video frame.
- `src/input/mock.ts` — mouse/keys → `TrackingFrame` (same shape). Used with `?input=mock` or when
  the camera is denied.
- `src/intent/calibration.ts` — capture neutral head position and shoulder width (hold still 1.5 s).
- `src/intent/interpret.ts` — pure: `(frame, calibration, prevState) → Intent`:
  `{ head: {lean, duck}, hands: {l, r, center, spread, vel} (screen-normalized), raised,
  events: {throw} }` plus filtered internal state (One-Euro-style smoothing).
  - lean = (head.x − neutral.x) / sw; duck = (head.y − neutral.y) / sw (clamped).
  - hand screen pos = shoulder-relative position scaled so a comfortable reach fills the screen.
  - throw = hands raised and together, and (hand size grows > X%/s  **or** hand speed > Y sw/s);
    150 ms refractory period.
  - hands lost → keep last position for 0.5 s, then report "no hands".
- `src/game/` — pure fixed-timestep simulation (fire, shield, enemies, projectiles, waves,
  scoring) consuming `Intent`. Exposes a read-only state for the renderer.
- `src/render/` — background layers with parallax, enemies, particles, first-person hands, HUD
  (DOM), debug overlay (video + landmarks + intent numbers, toggle with backtick).
- `src/main.ts` — screens: start → permissions → calibration → play → game over.

### Error handling
- Camera denied / unavailable → message + automatic mock mode.
- No person detected → pause with "step into frame".
- Model load failure → message with retry; mock mode still available.

### Testing
- Vitest unit tests for `interpret` (synthetic frames: distance-invariance, summon, throw,
  shield spread, lost-hands grace) and `game` (summon/throw/hit/shield/damage/waves).
- Manual playtest checklist against the success criteria.
