import './style.css';
import { downloadRecording, Recorder } from './debug/recorder';
import { Game } from './game/game';
import { CameraError, CameraTracker } from './input/camera';
import { bindMockControls, MOCK_CALIBRATION, MockTracker } from './input/mock';
import type { Tracker, TrackingFrame } from './input/types';
import { Calibrator, type Calibration } from './intent/calibration';
import { initialState, interpret, TUNING, type Cast, type Intent, type InterpretState, type Punch } from './intent/interpret';
import { DebugView } from './render/debug';
import { Hud } from './render/hud';
import { Renderer } from './render/renderer';

const STEP = 1 / 60;
const $ = (id: string) => document.getElementById(id)!;
const show = (id: string, on = true) => $(id).classList.toggle('hidden', !on);

const renderer = new Renderer($('game') as HTMLCanvasElement);
const hud = new Hud();
const debug = new DebugView($('pip') as HTMLCanvasElement, $('debugText'));

let phase: 'menu' | 'loading' | 'calibrating' | 'play' = 'menu';
let tracker: Tracker | null = null;
let camera: CameraTracker | null = null;
let calibrator = new Calibrator();
let calibration: Calibration | null = null;
let istate: InterpretState = initialState();
let intent: Intent | null = null;
/** Punches and casts seen since the last fixed game step. */
let pendingPunches: Punch[] = [];
let pendingCasts: Cast[] = [];
let lastFrame: TrackingFrame | null = null;
let game: Game | null = null;
const params = new URLSearchParams(location.search);
/** Dummies instead of spirits; toggled with T, or start with ?dummies. */
let practice = params.has('dummies');
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
  show('mockHelp');
  beginPlay();
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

function beginPlay(): void {
  istate = initialState();
  intent = null;
  pendingPunches = [];
  pendingCasts = [];
  acc = 0;
  game = new Game(Math.random, renderer.viewHalfW, practice);
  phase = 'play';
  show('calib', false);
  show('over', false);
}

function onFrame(f: TrackingFrame): void {
  lastFrame = f;
  if (phase === 'calibrating') {
    $('calibFill').style.width = `${Math.round(calibrator.add(f) * 100)}%`;
    const result = calibrator.result();
    if (result) {
      calibration = result;
      beginPlay();
    }
  } else if (phase === 'play' && calibration) {
    intent = interpret(f, calibration, istate);
    pendingPunches.push(...intent.punches); // held until the next fixed step consumes them
    pendingCasts.push(...intent.casts);
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
    game.step(STEP, { ...intent, punches: pendingPunches, casts: pendingCasts });
    pendingPunches = [];
    pendingCasts = [];
    acc -= STEP;
  }
  for (const e of game.drainEvents()) {
    renderer.onEvent(e);
    hud.onEvent(e);
    if (e.type === 'gameOver') {
      $('overScore').textContent = String(game.score);
      show('over');
    }
  }
  hud.update(game, intent.hands);
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
  if (phase === 'play') stepGame(dt);
  renderer.render(phase === 'play' ? game : null, dt);
  debug.draw(lastFrame, intent, camera?.video ?? null);
  $('handsN').textContent = String(lastFrame?.hands.length ?? 0);
  $('headTag').textContent = headLabel(intent);
  requestAnimationFrame(loop);
}

$('camBtn').addEventListener('click', () => void startCamera());
$('mockBtn').addEventListener('click', startMock);
$('statusFallback').addEventListener('click', startMock);
$('againBtn').addEventListener('click', beginPlay);
$('mockHelpClose').addEventListener('click', () => show('mockHelp', false));
addEventListener('resize', () => {
  renderer.resize();
  debug.resize();
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
  if (k === 't' && game?.state === 'play') {
    practice = !game.practice;
    game.setPractice(practice);
  }
  if (k === 'r' && game?.state === 'over') beginPlay();
  if (k === 'c' && camera && phase === 'play') beginCalibration();
  if ((k === '?' || k === '/') && tracker instanceof MockTracker) $('mockHelp').classList.toggle('hidden');
});

if (params.get('input') === 'mock') startMock();
requestAnimationFrame(loop);
