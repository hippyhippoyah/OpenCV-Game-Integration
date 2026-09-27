import './style.css';
import { CampaignRunner } from './campaign/runner';
import { CampaignUI, type HandoffCheck } from './campaign/ui';
import { Progress } from './campaign/progress';
import { STOPS } from './campaign/chapter1';
import { downloadRecording, Recorder } from './debug/recorder';
import { World3D } from './explore/world3d';
import { Look } from './explore/path';
import { Game, type GameEvent } from './game/game';
import { LESSONS, Tutorial } from './game/tutorial';
import { CameraError, CameraTracker } from './input/camera';
import { bindMockControls, MOCK_CALIBRATION, MockTracker } from './input/mock';
import type { Tracker, TrackingFrame } from './input/types';
import { Calibrator, type Calibration } from './intent/calibration';
import { initialState, interpret, TUNING, type Cast, type Intent, type InterpretState, type Palm, type Punch } from './intent/interpret';
import { DebugView } from './render/debug';
import { Hud } from './render/hud';
import { LessonDemo } from './render/lessonDemo';
import { Renderer } from './render/renderer';

const STEP = 1 / 60;
const $ = (id: string) => document.getElementById(id)!;
const show = (id: string, on = true) => $(id).classList.toggle('hidden', !on);

const renderer = new Renderer($('game') as HTMLCanvasElement);
const hud = new Hud();
const debug = new DebugView($('pip') as HTMLCanvasElement, $('debugText'));
const lessonDemo = new LessonDemo($('lessonDemo') as HTMLCanvasElement);

type Mode = 'tutorial' | 'waves' | 'training' | 'campaign';
let phase: 'menu' | 'loading' | 'calibrating' | 'modes' | 'play' = 'menu';
let mode: Mode = 'waves';
let tutorial: Tutorial | null = null;
/** The mouse & keys help is shown once, the first time you play with them. */
let mockHelpShown = false;
let tracker: Tracker | null = null;
let camera: CameraTracker | null = null;
let calibrator = new Calibrator();
let calibration: Calibration | null = null;
let istate: InterpretState = initialState();
let intent: Intent | null = null;
/** Punches and casts seen since the last fixed game step. */
let pendingPunches: Punch[] = [];
let pendingCasts: Cast[] = [];
let pendingPalms: Palm[] = [];
let lastFrame: TrackingFrame | null = null;
let game: Game | null = null;
const storage = (() => { try { return localStorage; } catch { return null; } })();
const progress = Progress.load(storage);
let campaign: CampaignRunner | null = null;
let campUI: CampaignUI | null = null;
let world: World3D | null = null;
const look = new Look();
const params = new URLSearchParams(location.search);
/** Skip the mode menu with ?mode=tutorial|waves|training|campaign (?dummies = training). */
const startMode: Mode | null = params.has('dummies') ? 'training'
  : (['tutorial', 'waves', 'training', 'campaign'] as const).find(m => m === params.get('mode')) ?? null;
/** Fist punches by arm extension (default) or open-hand punches; toggled with P, or start with ?punch=open. */
if (params.get('punch') === 'open') TUNING.punchTrigger = 'open';
let acc = 0, last = performance.now(), fpsTime = 0, fpsFrames = 0;
const RECORD_SECONDS = 10;
/** K records RECORD_SECONDS of tracking numbers and downloads them, for debugging detection offline. */
const recorder = new Recorder(r => {
  downloadRecording(r);
  hud.toast(`SAVED ${r.samples.length} FRAMES`, 'cool');
  show('recording', false);
});

function startMock(): void {
  const mock = new MockTracker(renderer);
  mock.setMouse(innerWidth / 2, innerHeight * 0.7);
  bindMockControls(mock, $('game'));
  tracker = mock;
  calibration = MOCK_CALIBRATION;
  show('start', false);
  show('status', false);
  if (startMode) beginPlay(startMode);
  else showModes();
}

async function startCamera(): Promise<void> {
  phase = 'loading';
  show('start', false);
  show('status');
  show('statusFallback', false);
  try {
    camera = await CameraTracker.create(message => { $('statusText').textContent = message; });
    tracker = camera;
    show('status', false);
    beginCalibration();
  } catch (e) {
    const why = e instanceof CameraError ? e.message : 'Something went wrong starting the camera.';
    $('statusText').textContent = `${why} You can still play with mouse and keys.`;
    show('statusFallback');
    phase = 'menu';
  }
}

function beginCalibration(): void {
  calibrator = new Calibrator();
  phase = 'calibrating';
  game = null;
  show('over', false);
  show('away', false);
  show('calib');
  $('calibFill').style.width = '0%';
}

/** The mode menu (after calibrating, or Esc in game). `note` is shown above the choices. */
function showModes(note = ''): void {
  phase = 'modes';
  game = null;
  tutorial = null;
  campUI?.hideAll();
  show('world', false);
  renderer.ghost = null;
  renderer.tint = null;
  document.exitPointerLock?.();
  for (const id of ['calib', 'over', 'away', 'lesson', 'dodge', 'mockHelp']) show(id, false);
  document.body.classList.remove('tutorial');
  $('campaignLabel').textContent = Object.keys(progress.data.stops).length > 0 ? 'Continue' : 'Campaign';
  $('modesNote').textContent = note;
  show('modesNote', !!note);
  show('modes');
}

function beginPlay(m: Mode = mode, lesson = 0): void {
  mode = m;
  istate = initialState();
  intent = null;
  pendingPunches = [];
  pendingCasts = [];
  pendingPalms = [];
  acc = 0;
  phase = 'play';
  for (const id of ['calib', 'over', 'modes']) show(id, false);
  if (m === 'campaign') {
    world ??= new World3D($('world') as HTMLCanvasElement);
    campUI ??= new CampaignUI(progress, stop => campaign?.replay(stop));
    campaign = new CampaignRunner(progress, () => new Game(Math.random, renderer.viewHalfW, true));
    game = null;
    tutorial = null;
    show('lesson', false);
    document.body.classList.remove('tutorial');
    if (tracker instanceof MockTracker && !mockHelpShown) {
      mockHelpShown = true;
      show('mockHelp');
    }
    return;
  }
  // training (and the tutorial, which then takes the field over) start with dummies, not a wave
  game = new Game(Math.random, renderer.viewHalfW, m !== 'waves');
  tutorial = m === 'tutorial' ? new Tutorial(game, lesson) : null;
  if (tutorial) game.label = 'Tutorial';
  show('lesson', !!tutorial);
  document.body.classList.toggle('tutorial', !!tutorial);
  drawLesson();
  if (tracker instanceof MockTracker && !mockHelpShown) {
    mockHelpShown = true;
    show('mockHelp');
  }
}

/** The lesson panel: which lesson, how to do it, and how far along the goal you are. */
function drawLesson(): void {
  if (!tutorial) return;
  const l = tutorial.lesson, done = tutorial.completedFor !== null;
  $('lessonStep').textContent = `Lesson ${tutorial.index + 1} of ${LESSONS.length}`;
  $('lessonTitle').textContent = l.title;
  $('lessonHow').textContent = l.how;
  $('lessonGoal').textContent = done ? '✓ Done — next lesson…' : l.goal;
  $('lessonCount').textContent = `${tutorial.done} / ${l.need}`;
  $('lessonFill').style.width = `${Math.round((tutorial.done / l.need) * 100)}%`;
  const steps = $('lessonSteps');
  steps.replaceChildren(...(l.steps ?? []).map(st => {
    const el = document.createElement('span');
    el.textContent = tutorial!.marksDone.has(st.mark) ? `✓ ${st.label}` : st.label;
    el.classList.toggle('done', tutorial!.marksDone.has(st.mark));
    return el;
  }));
  show('lessonSteps', !!l.steps);
  $('lesson').classList.toggle('done', done);
}

function onFrame(f: TrackingFrame): void {
  lastFrame = f;
  if (phase === 'calibrating') {
    $('calibFill').style.width = `${Math.round(calibrator.add(f) * 100)}%`;
    const result = calibrator.result();
    if (result) {
      calibration = result;
      if (startMode) beginPlay(startMode);
      else showModes();
    }
  } else if (phase === 'play' && calibration) {
    intent = interpret(f, calibration, istate);
    pendingPunches.push(...intent.punches); // held until the next fixed step consumes them
    pendingCasts.push(...intent.casts);
    pendingPalms.push(...intent.palms);
    recorder.push(f, camera?.lastRaw ?? null, intent);
  }
}

function stepGame(dt: number): void {
  if (!game || !intent) return;
  hud.update(game, intent.hands);
  show('away', !intent.present);
  const paused = !$('mockHelp').classList.contains('hidden');
  if (!intent.present || paused) { acc = 0; return; }
  acc += dt;
  while (acc >= STEP) {
    game.step(STEP, { ...intent, punches: pendingPunches, casts: pendingCasts, palms: pendingPalms });
    pendingPunches = [];
    pendingCasts = [];
    pendingPalms = [];
    acc -= STEP;
  }
  const events = game.drainEvents();
  for (const e of events) {
    renderer.onEvent(e);
    hud.onEvent(e);
    if (e.type === 'gameOver') {
      $('overScore').textContent = String(game.score);
      show('over');
    }
  }
  if (tutorial) {
    if (tutorial.update(dt, events)) hud.toast('✓ LESSON COMPLETE', 'good');
    if (tutorial.finished) { showModes('Tutorial complete — you know every move. Try the waves!'); return; }
    drawLesson();
  }
  hud.update(game, intent.hands);
}

function stepCampaign(dt: number, now: number): void {
  const r = campaign!;
  const exploring = r.state === 'walk' || r.state === 'scroll' || r.state === 'arena' || r.state === 'end';
  const check = handoffCheck();
  // the fight's game is the runner's
  game = r.game;
  let events: GameEvent[] = [];
  if (game && intent && (r.state === 'practice' || r.state === 'fight')) {
    acc += dt;
    while (acc >= STEP) { game.step(STEP, { ...intent, punches: pendingPunches, casts: pendingCasts, palms: pendingPalms }); pendingPunches = []; pendingCasts = []; pendingPalms = []; acc -= STEP; }
    events = game.drainEvents();
    for (const e of events) { renderer.onEvent(e); hud.onEvent(e); }
    hud.update(game, intent.hands);
  }
  if (!campUI!.overlayOpen) r.update(dt, check.seen && check.handsUp && check.distance === 'ok', events);
  renderer.ghost = r.ghostMove ? { lessonId: r.ghostMove, alpha: 1 } : null;
  renderer.tint = STOPS[r.stop].tint;
  show('world', exploring || r.state === 'handoff' || r.state === 'countdown');
  show('game', !(exploring || r.state === 'handoff' || r.state === 'countdown'));
  if (exploring || r.state === 'handoff' || r.state === 'countdown') {
    look.relax(dt);
    world!.setTaken(STOPS.flatMap((s, i) => (s.scroll && progress.hasScroll(s.scroll) ? [i] : [])));
    world!.render(r.rail, look, dt, r.state === 'scroll' ? 'scroll' : r.state === 'arena' ? 'arena' : null);
  }
  campUI!.update(r, check, now);
}

/** Is the camera ready for a fight: you're seen, fists up, at a good distance? (Mouse & keys: always.) */
function handoffCheck(): HandoffCheck {
  if (tracker instanceof MockTracker) return { seen: true, handsUp: true, distance: 'ok' };
  const f = lastFrame, i = intent;
  if (!f || !i?.present) return { seen: false, handsUp: false, distance: 'unknown' };
  const up = (h: typeof i.hands.l) => !!h && h.inView && h.pos.y < 40;
  const d = f.body ? (1.05 * f.body.span3) / f.body.span2 : null;
  return { seen: true, handsUp: up(i.hands.l) && up(i.hands.r), distance: d === null ? 'unknown' : d < 0.9 ? 'close' : d > 2.2 ? 'far' : 'ok' };
}

function headLabel(i: Intent | null): string {
  if (!i) return '—';
  if (i.head.y > 10) return 'duck';
  if (i.head.x < -10) return 'left';
  if (i.head.x > 10) return 'right';
  return 'center';
}

function loop(now: number): void {
  const elapsed = (now - last) / 1000, dt = Math.min(0.05, elapsed); // dt is capped for the simulation only
  last = now;
  fpsTime += elapsed;
  fpsFrames++;
  if (fpsTime > 0.5) {
    $('fps').textContent = `${Math.round(fpsFrames / fpsTime)} fps`;
    fpsTime = 0;
    fpsFrames = 0;
  }
  const f = tracker?.poll(now);
  if (f) onFrame(f);
  if (phase === 'play') {
    if (mode === 'campaign') stepCampaign(dt, now);
    else stepGame(dt);
  }
  renderer.render(phase === 'play' ? game : null, dt);
  if (tutorial && phase === 'play') lessonDemo.draw(tutorial.lesson.id, now / 1000);
  debug.draw(lastFrame, intent, camera?.video ?? null);
  $('handsN').textContent = String(lastFrame?.hands.length ?? 0);
  $('headTag').textContent = headLabel(intent);
  requestAnimationFrame(loop);
}

$('camBtn').addEventListener('click', () => void startCamera());
$('mockBtn').addEventListener('click', startMock);
$('statusFallback').addEventListener('click', startMock);
$('againBtn').addEventListener('click', () => beginPlay());
$('menuBtn').addEventListener('click', () => showModes());
for (const b of document.querySelectorAll<HTMLButtonElement>('button.mode')) {
  b.addEventListener('click', () => beginPlay(b.dataset.mode as Mode));
}
LESSONS.forEach((l, i) => {
  const b = document.createElement('button');
  b.textContent = `${i + 1}. ${l.title}`;
  b.addEventListener('click', () => beginPlay('tutorial', i));
  $('lessonChips').appendChild(b);
});
$('mockHelpClose').addEventListener('click', () => show('mockHelp', false));
$('campWalk').addEventListener('click', () => campaign?.walkOn());
$('campAgain').addEventListener('click', () => {
  if (campaign?.state === 'result') { campaign.state = 'lost'; campaign.retry(); }
});
$('campRetry').addEventListener('click', () => campaign?.retry());
$('campMenu').addEventListener('click', () => showModes());

/** Is the campaign currently exploring the 3D path (walk/scroll/arena/end), where mouse look & keys apply? */
function exploringCampaign(): boolean {
  return mode === 'campaign' && phase === 'play' && !!campaign
    && (campaign.state === 'walk' || campaign.state === 'scroll' || campaign.state === 'arena' || campaign.state === 'end');
}

// Mouse look while exploring the campaign path: pointer lock when available, else a plain drag.
let dragging = false, lastMouse: { x: number; y: number } | null = null;
$('world').addEventListener('mousedown', e => {
  if (!exploringCampaign() || !campaign) return;
  const r = campaign;
  if (document.pointerLockElement === $('world')) {
    if (r.state === 'scroll' || r.state === 'arena') r.interact();
  } else if ($('world').requestPointerLock) {
    try {
      const p = $('world').requestPointerLock() as unknown;
      if (p && typeof (p as Promise<void>).catch === 'function') (p as Promise<void>).catch(() => { /* fall back to plain drag */ });
    } catch { /* fall back to plain drag */ }
  }
  dragging = true;
  lastMouse = { x: e.clientX, y: e.clientY };
});
addEventListener('mouseup', () => { dragging = false; lastMouse = null; });
addEventListener('mousemove', e => {
  if (!exploringCampaign()) return;
  if (document.pointerLockElement === $('world')) {
    look.move(e.movementX, e.movementY);
  } else if (dragging && lastMouse) {
    look.move(e.clientX - lastMouse.x, e.clientY - lastMouse.y);
    lastMouse = { x: e.clientX, y: e.clientY };
  }
});
addEventListener('resize', () => {
  renderer.resize();
  debug.resize();
  world?.resize();
  if (game) game.viewHalfW = renderer.viewHalfW;
});
addEventListener('keydown', e => {
  if (e.repeat) return;
  const k = e.key.toLowerCase();
  if (k === '`') debug.toggle();
  if (k === 'k' && calibration && phase === 'play' && !recorder.active) {
    recorder.start(RECORD_SECONDS, calibration);
    show('recording');
  }
  if (k === 'p') {
    TUNING.punchTrigger = TUNING.punchTrigger === 'extend' ? 'open' : 'extend';
    istate.pending = [];
    hud.toast(TUNING.punchTrigger === 'extend' ? 'PUNCH: FIST' : 'PUNCH: OPEN HAND', 'cool');
  }
  if ((k === '[' || k === ']') && TUNING.punchTrigger === 'extend') {
    // live punch sensitivity: ] = easier to trigger, [ = stricter
    TUNING.punchSensitivity = Math.round(Math.min(2.5, Math.max(0.5, TUNING.punchSensitivity + (k === ']' ? 0.1 : -0.1))) * 10) / 10;
    hud.toast(`PUNCH SENSITIVITY ×${TUNING.punchSensitivity.toFixed(1)}`, 'cool');
  }
  if (k === 't' && game?.state === 'play' && mode !== 'tutorial' && mode !== 'campaign') {
    game.setPractice(!game.practice);
    mode = game.practice ? 'training' : 'waves';
  }
  if (mode === 'campaign' && phase === 'play' && campaign) {
    const r = campaign;
    if (exploringCampaign()) {
      if (k === 'e') r.interact();
      if (k === ' ') { e.preventDefault(); r.skip(); }
      if (k === 'm') campUI?.toggleMap();
      if (k === 'tab') { e.preventDefault(); campUI?.toggleScrolls(); }
    }
    if (k === 'escape') {
      if (r.state === 'handoff' || r.state === 'countdown') r.back();
      else if (campUI?.overlayOpen) { campUI.toggleMap(false); campUI.toggleScrolls(false); }
      else showModes();
      return;
    }
  }
  if (k === 'escape' && phase === 'play' && mode !== 'campaign') showModes();
  if (tutorial && phase === 'play' && (k === 'n' || k === 'b')) {
    if (k === 'n') tutorial.next(); else tutorial.back();
    if (tutorial.finished) showModes('Tutorial complete — you know every move. Try the waves!');
    else drawLesson();
  }
  if (k === 'r' && game?.state === 'over' && mode !== 'campaign') beginPlay();
  if (k === 'c' && camera && phase === 'play') beginCalibration();
  if ((k === '?' || k === '/') && tracker instanceof MockTracker) $('mockHelp').classList.toggle('hidden');
});

if (params.get('input') === 'mock') startMock();
requestAnimationFrame(loop);
