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

## Revision 2026-09-26: fist/open controls

Replaces the palms-together summon and push-throw, which depended on weak depth sensing.

- **Guard (rest):** both fists up at chest height. Fists smoulder with embers.
- **Punch:** a fist that moves fast (or toward the camera), then opens → fire leaves that hand.
  Aim = where the hand opened on screen, bent along shoulder→hand (x 0.5, y 0.2), snapping onto a
  target within 22 view units. Each hand punches independently (0.2 s cooldown per hand).
  Not fired if the other hand is open (or opens within 80 ms), since both open = shield.
- **Shield:** both hands open for 0.15 s → flame wall between them. Unlimited while testing
  (`TUNE.shieldInfinite`); drain/break logic kept for later.
- **Detection:** open/fist from 3D finger straightness (MediaPipe world landmarks), with hysteresis
  (open > 0.65, fist < 0.35). Palm facing (1 = toward camera, 0 = edge-on) is measured and shown in
  the debug panel; `TUNING.shieldNeedsEdgeOnPalms` can require palms facing each other for a shield.
  Hands are followed frame to frame so crossing punches keep their left/right labels.
- **Practice mode:** `T` or `?dummies` swaps spirits for still, respawning training dummies.

## Revision 2026-09-26: body tracking

- **Arms** (pose): shoulder, elbow and wrist per side with confidence, labelled by the person's own
  left/right. Wrists are still estimated when outside the picture.
- **Hands belong to arms:** each detected hand is labelled by the nearest pose wrist (both matched
  jointly); frame-to-frame continuity is only the fallback when no body is visible.
- **Fallback:** when the hand tracker loses a hand, its position follows the pose wrist (palm placed
  25% of a forearm past the wrist). `source` = `hand` / `arm` / `estimate`; `inView` false when the
  wrist is outside the picture or low-confidence. Shield needs both hands in view.
- **Arm extension:** elbow straightness in 3D (70° bent → 165° straight). A rise of 0.3 within 0.35 s
  counts as punch motion, so punches register even when the hand barely moves on screen.
- **Also recorded:** head turn/tilt (from nose, eyes, ears) and shoulder tilt.
- **Feedback:** first-person arms bend at the tracked elbow; an edge marker shows where an
  out-of-view hand is; the debug panel draws the arm skeleton and lists source/extension.

## Revision 2026-09-26: fist punches

- **Default punch = fist punch** (`TUNING.punchTrigger = 'extend'`): fires when an arm that is a
  fist straightens past 0.75 after rising by 0.3 within 0.35 s (hand raised, in view or tracked by
  its arm). The arm must drop below 0.5 to re-arm. 50 ms confirm window; cancelled if either hand
  opens (shield).
- **Open-hand punch** kept as the alternative (`'open'`): toggle with `P` in game or `?punch=open`.
- **Aim** also uses the 3D shoulder→wrist direction (tangent of the punch angle × 40 view units,
  blended 60% with the 2D aim), then snaps to a nearby target.
- **Debug panel** shows per-arm extension bars with the fire (orange) and re-arm (green) marks and
  a ready dot, plus the current punch style.

## Revision 2026-09-27: back to open-hand punches; fire wall and ultimate

- **Default punch = open-hand release** again (`punchTrigger: 'open'`). Fist punches stay behind `P`
  until a tracking recording (`K`) shows why arm extension didn't fire on a real camera.
- **Fire wall:** both hands open, then both rise ≥ 14 view units within 0.4 s (measured only after
  both are open) → a wall at depth 2.5 where the hands are, ±55 world units wide, for 4 s. It
  blocks enemy attacks crossing it; your fireballs pass through. 1 s cooldown.
- **Ultimate:** both hands open, then spread apart ≥ 24 view units within 0.4 s → every enemy
  destroyed (+100 each) and every incoming attack cleared. 12 s recharge (HUD meter).
- **Shield** now needs both open hands held still (< 35 view units/s) for 0.15 s to come up, so a
  sweep or spread doesn't raise it; once up it stays while both hands are open.
- **Fire only when doing something:** idle fists and open hands stay dark (faint rim only). Hands
  burn for 0.35 s after a punch, while shielding, and after a cast.
