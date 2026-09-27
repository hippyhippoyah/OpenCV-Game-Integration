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
- **Ultimate:** both hands open, then spread apart ≥ 24 view units within 0.4 s → a flat, spinning
  disc of fire (a saw blade) spreads out from the hands, dropped 45% of the way to the floor so it
  reads as a layer rather than an edge-on line. It grows at 16 depth units/s (30 world units
  sideways per depth unit) and cuts down every enemy (+100) and incoming attack its rim reaches,
  near ones first. 12 s recharge (HUD meter).
- **Shield** now needs both open hands held still (< 35 view units/s) for 0.15 s to come up, so a
  sweep or spread doesn't raise it; once up it stays while both hands are open.
- **Fire only when doing something:** idle fists and open hands stay dark (faint rim only). Hands
  burn for 0.35 s after a punch, while shielding, and after a cast.

## Revision 2026-09-27: fist punches rebuilt on reach-from-size; X block

**Why the old fist punch failed:** the 3D pose depth for an arm pointing at the camera is shallow
(a straight arm reads bent); the "hand size" cue used the wrist→knuckle length, which shrinks when
the knuckles face the camera; positions jittered; and aim came from that same shallow depth.

**How far each fist is in front of the body (metres):**
- Hand distance from the camera = its apparent size ÷ real size (MediaPipe 3D hand landmarks),
  from a least-squares fit of all 21 points for a fist (rigid 3D shape) or of the palm plate for
  an open hand — exact at any hand angle. Body distance = the same with the shoulders, using a
  learned (constant) shoulder width. Assumed focal length 1.05 picture heights; comparisons don't
  depend on it.
- reach = filtered body distance − filtered hand distance (One Euro filters; bodies slow, fists fast).

**Fist punch** (default; `P` switches to open-hand): a closed fist that is past its learned guard
baseline, came forward quickly, and leads the other fist. Thresholds scale with the reading's
wobble, modelled as (camera coefficient, learned from fists at rest) × distance², so a far,
noisy fist can't misfire; re-arm by pulling back. Aim = the fist's sideways/vertical offset from
its own shoulder over its reach (≥ 0.45 m), shared with an on-screen reticle per ready fist.

**Tested against a synthetic webcam** (`src/sim/`: a 3D body, camera projection, landmark noise,
shallow pose depth, blur dropouts): jabs and crosses caught ~100% up to 1.8 m (uppercuts to
1.5 m), no misfires from standing, weaving, leaning, slow reaches, two-handed pushes, the shield or
the X block at any distance up to 2.5 m. Beyond ~1.8 m punches are missed rather than misfiring,
and the HUD asks the player to step closer. Hooks (little forward motion) aren't detected.

**Also:** hand positions use One Euro filtering and no longer jump when falling back to the pose
wrist; arm labels are overridden by frame-to-frame continuity when they'd teleport a hand.

**X block:** each wrist crossed past the body's centre line (a cross punch moves only one), wrists
raised, held 0.08 s → blocks every attack that reaches you; draws a fiery X.

## Revision 2026-09-27: fist punches relaxed ("mostly distance")

Real-camera punches were often missed: the noise-scaled thresholds could climb out of reach on a
noisy webcam. Now: past guard ≥ 0.12 m, leading the other fist by ≥ 0.05 m, and ≥ 0.10 m of
forward movement within 0.5 s (only so drift doesn't count); noise scaling is capped (0.20 / 0.10 /
0.16 m), a fist only blocks a punch once it is clearly open (openness ≥ 0.8), and `[` / `]` change
sensitivity live. Simulated: 100% of punches caught at 1.2–1.8 m with no misfires there (was
80–98%); ~90% at 2.5 m with rare misfires (was 27%). A deliberate slow reach now counts.

## Revision 2026-09-27: fist punches are a quick jolt (rapid fire, partial punches)

A punch is now any quick movement of a fist toward the camera, not reaching a set distance: the
fist came ≥ max(0.065 m, 8 × wobble) closer within 0.2 s (capped at 0.2 m), more than the other
fist did by ≥ max(0.05 m, 4 × wobble). It re-arms once it comes back ≥ max(0.05 m, 4 × wobble)
from the punch's peak (no need to return to guard), at most one punch per 0.12 s per fist; the
game cooldown is 0.1 s. A fist's reach reading warms up 0.6 s before it can punch. The hand-depth
filter releases faster (One Euro 2 Hz, β 4) so rapid snaps aren't averaged away.

Simulated, tuned sensitivity-first: rapid snaps (4/s, 13–24 cm) and jabs 100% caught at 1.2–1.8 m;
misfires from standing/weaving/leaning/two-handed pushes: none at 1.2 m, ~1 per 50 s at 1.5–1.8 m;
trigger-happy at 2.5 m (HUD: step closer). `[` / `]` trade sensitivity for strictness live.

## Revision 2026-09-27: palm moves (pillar and eruption)

Two heavy single-hand attacks with an open palm, fist-punch mode only (the open-hand punch style
already turns opening hands into punches). The palm must have been open ≥ 0.1 s (so a fist opening
at the end of a punch isn't one) and the other hand not open. Each waits 0.08 s and is dropped if the
other hand opens meanwhile (shield or cast), then that hand rests 0.5 s in detection, 0.8 s in game.

- **Push → rolling pillar:** the palm shoved toward the camera: its reach, averaged over 3 frames,
  rises ≥ max(0.12 m, 9 × wobble) (capped at 0.17 m) within 0.3 s, leading the other hand. An open
  palm's reach reading wobbles ~1.7× a fist's, hence the bigger, smoothed threshold. A column of
  fire rolls forward from the hand toward the aim at 9 depth/s, burning each enemy it passes
  (2 damage, a punch does 1) and every attack it meets.
- **Rise → eruption:** the palm swept up ≥ 16 view units (½ shoulder width) within 0.3 s, 1.5× more
  up than sideways. A glowing mark appears under the aimed target (same aim and assist as punches);
  0.25 s later a pillar bursts up there, burning enemies and attacks within 14 units, for 1 s.

Opening or closing a hand switches how its distance is measured (palm plane vs whole-fist fit), so
the reach reading restarts on a shape change instead of reading the jump as movement.

Simulated at 1.2–1.8 m: every push and rise caught, with no fist punch; nothing fires from jabs, a
still or slowly moving open palm, raising the shield, or a two-hand wall sweep. Mock: E = push,
Q = rise (right hand). A rise-then-push combo (a bigger pillar) is a possible follow-up.

## Revision 2026-09-27: eruption removed; palm push more forgiving

The rising-palm eruption was too unreliable on camera and is gone; the push stays, and was missing
real pushes. Now:

- The palm no longer has to be open beforehand: it may open on the way out. A fist punch whose hand
  opens while it is confirming becomes a push (one attack either way).
- Opening/closing no longer wipes the reach history: the history is shifted by the measurement
  jump, so the forward movement before the hand opened still counts.
- Threshold (3-frame-averaged reach, within 0.35 s, counting only movement since the hand's last
  attack): ≥ max(0.08 m, 8 × wobble), capped at 0.15 m, ÷ punch sensitivity (`[` / `]` adjust
  both). Frames lost to blur are skipped, and a push can continue on the pose wrist.

Simulated: full, slow (0.3 s) and short (15 cm) pushes, and pushes that open mid-way, all caught at
1.2–1.5 m; at 1.8 m a 20 cm push. No pushes from a still or slowly moving open palm, the shield, jabs
or both palms pushed together.

## Revision 2026-09-27: default sensitivity 1.4

By preference, punches and pushes default to sensitivity ×1.4 (all thresholds ÷ 1.4) to catch
more real attempts. The simulated suites still test detection at ×1.0; at ×1.4 the simulator shows
occasional misfires (weaving or leaning in guard, a still open palm), most at 1.5 m and beyond.
`[` / `]` still adjust it live.

## Revision 2026-09-27: movement you can see; wall push; stricter ultimate

**Movement.** Leaning, stepping and ducking are the only movement, so they are exaggerated: 70
view units per shoulder width of head movement sideways (was 40, max ±60) and 55 down (was 40, max
35); the view also tilts against the lean (0.0014 rad per unit).

**Attacks you have to move out of.** Spirits now wind up one of three attacks (orb 50%, quake 30%,
high sweep 20%), each with its own tell:
- *Quake* (earth; the ground cracks at the spirit's feet): rock spikes rip toward you along the
  ground on the side the spirit stands on, covering from 10 units past where you stood outward; the
  danger side of the floor glows red. Lean or step ≥ ~22 units the other way. Toast names the side.
- *High sweep* (a disc spins up above the spirit): a sheet of water crosses the whole field at the
  height your eyes were; a dashed red line marks it. Duck ≥ 14 units.
Shield and X block don't stop either; a fire wall (standing or rolling) does.

**Wall push.** Both open palms shoved toward the camera (each ≥ 0.8 × a single push's threshold, not
scaled by punch sensitivity) roll a fire wall (80 wide) forward at 7 depth/s, burning each enemy it
passes (2 damage) and blocking attacks; 2.5 s cooldown. Pushing out of a held shield works.

**Ultimate vs wall push.** Pushing both palms at the camera makes them look further apart, which
read as the ultimate's spread. The ultimate is now "gather and fling": hands start together (≤ 0.4 m
apart in 3D, or ≤ 0.9 shoulder widths on screen without 3D data) and spread ≥ 0.3 m in 3D (which a
push doesn't change); the push is checked first. Simulated at 1.2–1.8 m (sensitivity ×1 and ×1.4):
pushes, pushes out of the shield, gather-and-fling and sweeps each fire only their own cast; a held
shield casts nothing. Mock: F = wall push.

## Revision 2026-09-27: earthbenders instead of quakes; enemies stay central

The quake didn't fit the world and came too fast. It is replaced by an **earthbender** (green and
brown robes; about a third of enemies, at least one per wave). His opening move: over a 1.4 s
wind-up he stomps and lifts his arms, raising a 70-unit stone pillar beside himself, then drives
both palms forward and the pillar slides at you at 3.5 depth/s (~2 s to arrive), curving into a
lane just to one side of where you stood (centre 10 units off yours, 26 wide); that lane glows red
on the floor in front of you and a toast names the side. Lean or step the other way. Knocking him
down while he raises it crumbles it; a fire wall (standing or rolling) stops it; shield and X
block don't. Afterwards he throws rocks (blockable like orbs), raising another pillar 30% of the
time. Water spirits keep orbs and the high sweep (30%).

**Centred enemies.** Aiming at the screen edges is hard, so enemies spawn and drift only within
0.4 of the screen's half-width of its centre (spread apart from each other), and dummies stand at
±36 instead of ±45.

**Pillar warning (revised):** instead of a red lane on the floor, the edge of the screen on the
pillar's side flashes red (the inner 45% of the width fades out) — faintly while it rises, then
stronger and faster as it closes in — and only while you're still in its path: once you've moved
out of the way, it stops.

## Revision 2026-09-27: clearer, calmer dodging

- **Nothing startling:** the pillar warning is a soft, slowly breathing red glow on the edge of the
  screen on its side (30% wide, max ~0.28 alpha), only while you're in its path; being hit is a
  small shake and a soft red edge (was a full shake and flash); a shoved pillar barely shakes.
- **Pillars clearly go to one side:** a pillar erupts straight up in its lane (centre 22 units off
  yours, 40 wide, so its near edge only just reaches past your centre) and slides straight down it,
  ploughing a furrow on the ground from the pillar to you so you can see the track it will take.
- **Earthbenders only raise pillars;** rocks are gone (too hard to see).
- **High sweep looks like water:** a wave rolling at you with a foamy crest at your eye height; its
  underside sits just above where your eyes need to duck to, so ducking visibly takes you under it.
- **Always know if you're dodging:** while a pillar or sweep is coming, a cue below the centre
  says what to do in red (← MOVE LEFT / MOVE RIGHT → / ↓ DUCK) with a bar filling as it arrives,
  and turns green (✓ CLEAR) the moment you're out of its way; a green ✓ DODGED confirms it.

**Wave height (revised):** the high sweep always comes at standing eye height (`slabY` 0), no
longer at wherever your eyes were when it was sent — a wave sent while you were ducking came in
too low to duck under.
