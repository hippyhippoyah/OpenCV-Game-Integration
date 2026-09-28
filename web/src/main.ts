import './style.css';
import { ghostAlpha } from './campaign/ghostFade';
import { CampaignRunner } from './campaign/runner';
import { CampaignUI, type HandoffCheck } from './campaign/ui';
import { Progress } from './campaign/progress';
import { STOPS } from './campaign/chapter1';
import { downloadRecording, Recorder } from './debug/recorder';
import { PathMap } from './explore/map2d';
import { Game, type GameEvent } from './game/game';
import { Tutorial } from './game/tutorial';
import { CameraError, CameraTracker } from './input/camera';
import { bindMockControls, MOCK_CALIBRATION, MockTracker } from './input/mock';
import type { Tracker, TrackingFrame } from './input/types';
import { Calibrator, type Calibration } from './intent/calibration';
import { initialState, interpret, TUNING, type Cast, type Intent, type InterpretState, type Palm, type Punch } from './intent/interpret';
import { DebugView } from './render/debug';
import { Hud } from './render/hud';
import { Sfx } from './audio/sfx';
import { Menu } from './ui/menu';
import { Settings, type InputKind } from './ui/settings';
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
/** front: title, menu or settings (see ui/menu.ts); loading/calibrating: camera setup before play. */
let phase: 'front' | 'loading' | 'calibrating' | 'play' = 'front';
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
let pathMap: PathMap | null = null;
/** Ghost hands' fade state for the campaign's current practice lesson (see stepCampaign). */
let ghostPractice: Tutorial | null = null;
let ghostLessonIndex = -1;
let ghostDone = 0;
let ghostSinceProgress = 0;
let ghostAlphaNow = 1;
/** So "NOW FOR REAL" toasts only when practice just ended (the runner's previous state). */
let prevCampaignState: string | null = null;
/** True while the campaign's practice/fight is paused via Esc (see stepCampaign & togglePause). */
let campPaused = false;
/** When R was first pressed on the campaign map (a second R soon after resets progress). */
let resetAskedAt = -Infinity;
const params = new URLSearchParams(location.search);
/** Skip the mode menu with ?mode=tutorial|waves|training|campaign (?dummies = training). */
const startMode: Mode | null = params.has('dummies') ? 'training'
  : (['tutorial', 'waves', 'training', 'campaign'] as const).find(m => m === params.get('mode')) ?? null;
/** Fist punches by arm extension (default); open-hand punches only with ?punch=open (no key switches it). */
if (params.get('punch') === 'open') TUNING.punchTrigger = 'open';
const settings = Settings.load(storage);
/** ?input=mock plays with mouse and keys this time (not saved). */
if (params.get('input') === 'mock') settings.data.input = 'mock';
TUNING.punchSensitivity = settings.data.sensitivity;
let mockTracker: MockTracker | null = null;
/** The camera's own calibration (mouse & keys use MOCK_CALIBRATION). */
let camCalibration: Calibration | null = null;
/** What to start once the camera is set up. */
let pendingPlay: { mode: Mode; lesson: number } | null = null;
const sfx = new Sfx();
sfx.setVolume(settings.data.volume);
/** Last frame's charge and gather, so their "ready" sounds play once. */
const wasCharged = { l: false, r: false };
let wasGathered = false;
let wasPrimed = false;
const menu = new Menu(settings, progress, {
  sound: s => sfx.ui(s),
  onVolume: v => sfx.setVolume(v),
  onPlay: (m, lesson) => play(m, lesson ?? 0),
  onInput: () => { /* takes effect at the next play */ },
  onSensitivity: v => { TUNING.punchSensitivity = v; },
  onResetCampaign: () => progress.reset(),
});
let acc = 0, last = performance.now(), fpsTime = 0, fpsFrames = 0;
const RECORD_SECONDS = 10;
/** K records RECORD_SECONDS of tracking numbers and downloads them, for debugging detection offline. */
const recorder = new Recorder(r => {
  downloadRecording(r);
  hud.toast(`SAVED ${r.samples.length} FRAMES`, 'cool');
  show('recording', false);
});

/** Switch to an input; returns true if it's ready to play (the camera needs setting up first). */
function useInput(kind: InputKind): boolean {
  if (kind === 'mock') {
    if (!mockTracker) {
      mockTracker = new MockTracker(renderer);
      mockTracker.setMouse(innerWidth / 2, innerHeight * 0.7);
      bindMockControls(mockTracker, $('game'));
    }
    tracker = mockTracker;
    calibration = MOCK_CALIBRATION;
    return true;
  }
  if (!camera) return false;
  tracker = camera;
  calibration = camCalibration;
  return calibration !== null;
}

/** Start a mode from the menu: straight in, or via camera setup the first time. */
function play(m: Mode, lesson = 0): void {
  if (useInput(settings.data.input)) { beginPlay(m, lesson); return; }
  pendingPlay = { mode: m, lesson };
  if (camera) beginCalibration();
  else void startCamera();
}

async function startCamera(): Promise<void> {
  phase = 'loading';
  menu.hide();
  document.body.classList.add('setup');
  show('status');
  show('statusFallback', false);
  show('statusBack', false);
  $('statusText').textContent = 'Allow camera access when your browser asks.';
  try {
    camera = await CameraTracker.create(message => { $('statusText').textContent = message; });
    tracker = camera;
    show('status', false);
    beginCalibration();
  } catch (e) {
    const why = e instanceof CameraError ? e.message : 'Something went wrong starting the camera.';
    $('statusText').textContent = `${why} You can still play with mouse and keys.`;
    show('statusFallback');
    show('statusBack');
  }
}

function beginCalibration(): void {
  calibrator = new Calibrator();
  phase = 'calibrating';
  menu.hide();
  document.body.classList.add('setup');
  game = null;
  show('over', false);
  show('away', false);
  show('calib');
  $('calibFill').style.width = '0%';
}

/** Back to the main menu (Esc in game, or after a mode ends). `note` is shown above the choices. */
function showMenu(note = ''): void {
  phase = 'front';
  game = null;
  tutorial = null;
  campPaused = false;
  pendingPlay = null;
  campUI?.hideAll();
  show('world', false);
  show('status', false);
  renderer.ghost = null;
  renderer.scene = 'courtyard';
  for (const id of ['calib', 'over', 'away', 'lesson', 'dodge', 'mockHelp']) show(id, false);
  document.body.classList.remove('tutorial', 'exploring', 'setup');
  menu.showMenu(note);
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
  for (const id of ['calib', 'over', 'status']) show(id, false);
  menu.hide();
  document.body.classList.remove('setup');
  if (m === 'campaign') {
    pathMap ??= new PathMap($('world') as HTMLCanvasElement);
    campUI ??= new CampaignUI(progress);
    campaign = new CampaignRunner(progress, () => new Game(Math.random, renderer.viewHalfW, true));
    game = null;
    tutorial = null;
    show('lesson', false);
    document.body.classList.remove('tutorial');
    ghostPractice = null;
    ghostLessonIndex = -1;
    ghostAlphaNow = 1;
    prevCampaignState = null;
    campPaused = false;
    if (tracker instanceof MockTracker && !mockHelpShown) {
      mockHelpShown = true;
      show('mockHelp');
    }
    return;
  }
  // training (and the tutorial, which then takes the field over) start with dummies, not a wave
  game = new Game(Math.random, renderer.viewHalfW, m !== 'waves');
  // the waves are fought at the night temple; practice of any kind in the training yard
  renderer.scene = m === 'waves' ? 'night' : 'training';
  tutorial = m === 'tutorial' ? new Tutorial(game, lesson) : null;
  if (tutorial) game.label = 'Tutorial';
  show('lesson', !!tutorial);
  document.body.classList.toggle('tutorial', !!tutorial);
  if (tutorial) drawLesson(tutorial);
  if (tracker instanceof MockTracker && !mockHelpShown) {
    mockHelpShown = true;
    show('mockHelp');
  }
}

/**
 * The lesson panel: which lesson, how to do it, and how far along the goal you are. Used for the
 * tutorial's own lessons and (with a different `Tutorial` instance) for a campaign practice.
 */
function drawLesson(t: Tutorial): void {
  const l = t.lesson, done = t.completedFor !== null;
  $('lessonStep').textContent = `Lesson ${t.index + 1} of ${t.lessons.length}`;
  $('lessonTitle').textContent = l.title;
  $('lessonHow').textContent = l.how;
  $('lessonGoal').textContent = done ? '✓ Done — next lesson…' : l.goal;
  $('lessonCount').textContent = `${t.done} / ${l.need}`;
  $('lessonFill').style.width = `${Math.round((t.done / l.need) * 100)}%`;
  const steps = $('lessonSteps');
  steps.replaceChildren(...(l.steps ?? []).map(st => {
    const el = document.createElement('span');
    el.textContent = t.marksDone.has(st.mark) ? `✓ ${st.label}` : st.label;
    el.classList.toggle('done', t.marksDone.has(st.mark));
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
      calibration = camCalibration = result;
      const next = pendingPlay ?? { mode, lesson: 0 };
      pendingPlay = null;
      beginPlay(next.mode, next.lesson);
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
  cueSounds(game, events);
  for (const e of events) {
    renderer.onEvent(e);
    hud.onEvent(e);
    if (e.type === 'gameOver') {
      $('overScore').textContent = String(game.score);
      const best = mode === 'waves' && settings.recordScore(game.score);
      $('overBest').textContent = best ? 'A new best!' : mode === 'waves' ? `Best: ${settings.data.best}` : '';
      show('overBest', mode === 'waves');
      show('over');
    }
  }
  if (tutorial) {
    if (tutorial.update(dt, events)) hud.toast('✓ LESSON COMPLETE', 'good');
    if (tutorial.finished) { showMenu('Tutorial complete — you know every move. Try the waves!'); return; }
    drawLesson(tutorial);
  }
  hud.update(game, intent.hands);
}

/** Play the game's sounds, and cue a hand charging blue or the finisher's gather catching fire. */
function cueSounds(g: Game, events: GameEvent[]): void {
  for (const e of events) sfx.onEvent(e, g.cam.x);
  for (const side of ['l', 'r'] as const) {
    const full = (g.hands[side]?.charge ?? 0) >= 1;
    if (full && !wasCharged[side]) sfx.ui('charged');
    wasCharged[side] = full;
  }
  if (g.gather >= 1 && !wasGathered) sfx.ui('gathered');
  wasGathered = g.gather >= 1;
  if (g.infernoPrep >= 1 && !wasPrimed) sfx.ui('blueReady');
  wasPrimed = g.infernoPrep >= 1;
}

function stepCampaign(dt: number, now: number): void {
  const r = campaign!;
  const exploring = r.state === 'walk' || r.state === 'scroll' || r.state === 'arena' || r.state === 'end';
  const check = handoffCheck();
  // the fight's game is the runner's
  game = r.game;
  let events: GameEvent[] = [];
  if (campPaused) {
    // dropped so nothing fires on resume
    pendingPunches = [];
    pendingCasts = [];
    pendingPalms = [];
  } else {
    if (game && intent && (r.state === 'practice' || r.state === 'fight')) {
      acc += dt;
      while (acc >= STEP) { game.step(STEP, { ...intent, punches: pendingPunches, casts: pendingCasts, palms: pendingPalms }); pendingPunches = []; pendingCasts = []; pendingPalms = []; acc -= STEP; }
      events = game.drainEvents();
      for (const e of events) { renderer.onEvent(e); hud.onEvent(e); }
      cueSounds(game, events);
      hud.update(game, intent.hands);
    }
    const before = r.state, count = Math.ceil(r.countdown);
    if (!campUI!.overlayOpen) r.update(dt, check.seen && check.handsUp && check.distance === 'ok', events);
    // countdown ticks, then a rush of fire as the fight starts; a chime per scroll and per flame won
    if (r.state === 'countdown' && (before !== 'countdown' || Math.ceil(r.countdown) !== count)) sfx.ui('tick');
    if (before === 'countdown' && r.state !== 'countdown' && r.state !== 'arena') sfx.ui('go');
    if (r.notes.some(n => n.kind === 'scroll')) sfx.ui('scroll');
    if (before !== 'result' && r.state === 'result' && r.result) {
      for (let i = 0; i < r.result.flames; i++) setTimeout(() => sfx.ui('flame'), 250 + i * 280);
    }
  }
  if (prevCampaignState === 'practice' && r.state === 'fight') { hud.toast('NOW FOR REAL'); sfx.ui('go'); }
  prevCampaignState = r.state;
  // the lesson card, shared with the tutorial's own: shown only while practising a campaign lesson
  const practice = r.state === 'practice' ? r.practice : null;
  show('lesson', !!practice);
  document.body.classList.toggle('tutorial', !!practice);
  if (practice) {
    if (practice !== ghostPractice || practice.index !== ghostLessonIndex) {
      ghostPractice = practice;
      ghostLessonIndex = practice.index;
      ghostDone = practice.done;
      ghostSinceProgress = 0;
      ghostAlphaNow = 1;
    } else if (practice.done !== ghostDone) {
      ghostDone = practice.done;
      ghostSinceProgress = 0;
    } else {
      ghostSinceProgress += dt;
    }
    ghostAlphaNow = ghostAlpha(ghostAlphaNow, dt, practice.done, ghostSinceProgress, practice.finished);
    drawLesson(practice);
    renderer.ghost = ghostAlphaNow > 0.001 ? { lessonId: practice.lesson.id, alpha: ghostAlphaNow } : null;
  } else {
    ghostPractice = null;
    renderer.ghost = null;
  }
  // practice always in the training yard; the real fight where it happens
  renderer.scene = r.state === 'fight' || r.state === 'lost' || r.state === 'result' ? STOPS[r.stop].scene : 'training';
  // on the map the fight HUD (health, bars, camera view) is put away
  document.body.classList.toggle('exploring', exploring);
  show('world', exploring || r.state === 'handoff' || r.state === 'countdown');
  show('game', !(exploring || r.state === 'handoff' || r.state === 'countdown'));
  if (exploring || r.state === 'handoff' || r.state === 'countdown') {
    pathMap!.render({
      d: r.rail.d,
      walking: r.state === 'walk' && !campUI!.overlayOpen,
      waiting: r.state === 'scroll' || r.state === 'arena',
      isDone: i => progress.isDone(STOPS[i].id),
      flames: i => progress.flames(STOPS[i].id),
      hasScroll: id => progress.hasScroll(id),
      next: STOPS.findIndex(s => !progress.isDone(s.id)),
    }, dt);
  }
  campUI!.update(r, check, now, campPaused);
  if (practice) lessonDemo.draw(practice.lesson.id, now / 1000);
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
  // held sounds (shield fire, ultimates warming up) only while actually fighting
  const helpOpen = !$('mockHelp').classList.contains('hidden');
  sfx.ambient(phase === 'play' && game && game.state === 'play' && !campPaused && !helpOpen && intent?.present ? game : null);
  if (tutorial && phase === 'play') lessonDemo.draw(tutorial.lesson.id, now / 1000);
  debug.draw(lastFrame, intent, camera?.video ?? null);
  $('handsN').textContent = String(lastFrame?.hands.length ?? 0);
  $('headTag').textContent = headLabel(intent);
  requestAnimationFrame(loop);
}

$('statusFallback').addEventListener('click', () => {
  settings.data.input = 'mock';
  settings.save();
  const next = pendingPlay ?? { mode, lesson: 0 };
  pendingPlay = null;
  show('status', false);
  play(next.mode, next.lesson);
});
$('statusBack').addEventListener('click', () => showMenu());
$('againBtn').addEventListener('click', () => beginPlay());
$('menuBtn').addEventListener('click', () => showMenu());
$('mockHelpClose').addEventListener('click', () => show('mockHelp', false));
$('campWalk').addEventListener('click', () => campaign?.walkOn());
/** Fight the stop again from its result card (not saved as done) or after losing. */
function campaignTryAgain(): void {
  if (campaign?.state === 'result') { campaign.state = 'lost'; campaign.retry(); }
  else campaign?.retry();
}
$('campAgain').addEventListener('click', campaignTryAgain);
$('campRetry').addEventListener('click', campaignTryAgain);
$('campMenu').addEventListener('click', () => showMenu());
$('campResume').addEventListener('click', () => { campPaused = false; });
$('campLeave').addEventListener('click', () => { campPaused = false; showMenu(); });

/** Is the campaign currently exploring the path map (walk/scroll/arena/end), where the map clicks & keys apply? */
function exploringCampaign(): boolean {
  return mode === 'campaign' && phase === 'play' && !!campaign
    && (campaign.state === 'walk' || campaign.state === 'scroll' || campaign.state === 'arena' || campaign.state === 'end');
}

// Clicking a finished stop's lantern on the map replays it.
$('world').addEventListener('click', e => {
  if (!exploringCampaign() || !campaign || campaign.state === 'end' || campUI?.overlayOpen) return;
  const stop = pathMap?.stopAt(e.clientX, e.clientY);
  if (stop !== null && stop !== undefined && progress.isDone(STOPS[stop].id)) campaign.replay(stop);
});
addEventListener('resize', () => {
  renderer.resize();
  debug.resize();
  pathMap?.resize();
  if (game) game.viewHalfW = renderer.viewHalfW;
});
addEventListener('keydown', e => {
  if (e.repeat) return;
  // the title, menu and settings take their own keys
  if (phase === 'front' && menu.key(e)) return;
  if (phase === 'loading' && e.key === 'Escape') { showMenu(); return; }
  const k = e.key.toLowerCase();
  if (k === '`') debug.toggle();
  if (k === 'k' && calibration && phase === 'play' && !recorder.active) {
    recorder.start(RECORD_SECONDS, calibration);
    show('recording');
  }
  if ((k === '[' || k === ']') && TUNING.punchTrigger === 'extend') {
    // live punch sensitivity: ] = easier to trigger, [ = stricter
    TUNING.punchSensitivity = Math.round(Math.min(2.5, Math.max(0.5, TUNING.punchSensitivity + (k === ']' ? 0.1 : -0.1))) * 10) / 10;
    hud.toast(`PUNCH SENSITIVITY ×${TUNING.punchSensitivity.toFixed(1)}`, 'cool');
    settings.data.sensitivity = TUNING.punchSensitivity;
    settings.save();
  }
  if (k === 't' && game?.state === 'play' && mode !== 'tutorial' && mode !== 'campaign') {
    game.setPractice(!game.practice);
    mode = game.practice ? 'training' : 'waves';
  }
  if (mode === 'campaign' && phase === 'play' && campaign) {
    const r = campaign;
    // the result and lost cards' buttons, from the keyboard
    if (r.state === 'result' && k === 'enter') { e.preventDefault(); r.walkOn(); return; }
    if ((r.state === 'result' || r.state === 'lost') && k === 'r') { campaignTryAgain(); return; }
    if (exploringCampaign()) {
      if (k === 'e') r.interact();
      if (k === ' ') { e.preventDefault(); r.skip(); }
      if (k === 'tab') { e.preventDefault(); campUI?.toggleScrolls(); }
      // R twice (within a few seconds) erases the campaign and starts Chapter 1 over
      if (k === 'r') {
        if (performance.now() - resetAskedAt < 4000) {
          resetAskedAt = -Infinity;
          progress.reset();
          campaign = new CampaignRunner(progress, () => new Game(Math.random, renderer.viewHalfW, true));
          campaign.notes.push({ kind: 'info', text: 'Campaign progress reset — Chapter 1 starts fresh' });
          sfx.ui('reset');
        } else {
          resetAskedAt = performance.now();
          r.notes.push({ kind: 'info', text: 'Press R again to erase all scrolls, stops and flames' });
        }
      }
    }
    if (k === 'escape') {
      if (r.state === 'handoff' || r.state === 'countdown') r.back();
      else if (campUI?.overlayOpen) campUI.toggleScrolls(false);
      else if (r.state === 'practice' || r.state === 'fight') campPaused = !campPaused;
      else if (r.state === 'lost' || r.state === 'result') { /* the cards' own buttons decide */ }
      else showMenu();
      return;
    }
  }
  if (k === 'escape' && phase === 'play' && mode !== 'campaign') showMenu();
  if (tutorial && phase === 'play' && (k === 'n' || k === 'b')) {
    if (k === 'n') tutorial.next(); else tutorial.back();
    if (tutorial.finished) showMenu('Tutorial complete — you know every move. Try the waves!');
    else drawLesson(tutorial);
  }
  if (k === 'r' && game?.state === 'over' && mode !== 'campaign') beginPlay();
  if (k === 'c' && camera && tracker === camera && phase === 'play') {
    pendingPlay = { mode, lesson: tutorial?.index ?? 0 };
    beginCalibration();
  }
  if ((k === '?' || k === '/') && tracker instanceof MockTracker) $('mockHelp').classList.toggle('hidden');
});

// ?mode=… goes straight into play; otherwise the title screen, over the courtyard at dawn
renderer.scene = 'courtyard';
if (startMode) play(startMode);
else menu.showTitle();
requestAnimationFrame(loop);
