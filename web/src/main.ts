import './style.css';
import { Game } from './game/game';
import { CameraError, CameraTracker } from './input/camera';
import { bindMockControls, MOCK_CALIBRATION, MockTracker } from './input/mock';
import type { Tracker, TrackingFrame } from './input/types';
import { Calibrator, type Calibration } from './intent/calibration';
import { initialState, interpret, type Intent, type InterpretState } from './intent/interpret';
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
let pendingThrow = false;
let lastFrame: TrackingFrame | null = null;
let game: Game | null = null;
let acc = 0, last = performance.now(), fpsTime = 0, fpsFrames = 0;

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
  pendingThrow = false;
  acc = 0;
  game = new Game(Math.random, renderer.viewHalfW);
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
    if (intent.throwNow) pendingThrow = true; // held until the next fixed step consumes it
  }
}

function stepGame(dt: number): void {
  if (!game || !intent) return;
  show('away', !intent.present);
  const paused = !$('mockHelp').classList.contains('hidden');
  if (!intent.present || paused) { acc = 0; return; }
  acc += dt;
  while (acc >= STEP) {
    game.step(STEP, { ...intent, throwNow: pendingThrow });
    pendingThrow = false;
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
  hud.update(game);
}

function headLabel(i: Intent | null): string {
  if (!i) return '—';
  if (i.head.y > 10) return 'duck';
  if (i.head.x < -10) return 'left';
  if (i.head.x > 10) return 'right';
  return 'center';
}

function loop(now: number): void {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  fpsTime += dt;
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
  if (k === 'r' && game?.state === 'over') beginPlay();
  if (k === 'c' && camera && phase === 'play') beginCalibration();
  if ((k === '?' || k === '/') && tracker instanceof MockTracker) $('mockHelp').classList.toggle('hidden');
});

if (new URLSearchParams(location.search).get('input') === 'mock') startMock();
requestAnimationFrame(loop);
