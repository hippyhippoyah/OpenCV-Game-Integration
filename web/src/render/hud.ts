import { TUNE, type ComboName, type Game, type GameEvent } from '../game/game';

const COMBO_NAMES: Record<ComboName, string> = {
  charged: 'CHARGED', flurry: 'FLURRY', counter: 'COUNTER', oneTwo: 'ONE-TWO PUSH', volley: 'PILLAR VOLLEY',
  wallBreaker: 'WALL BREAKER', finisher: 'FINISHER',
};
import { TUNING, type Intent } from '../intent/interpret';

const $ = (id: string) => document.getElementById(id)!;

/** DOM overlay: health, score, current move, shield energy, toasts and wave banners. */
export class Hud {
  /** `hands` is the live tracking, so the HUD is right even before the game has stepped (e.g. paused). */
  update(g: Game, hands: Intent['hands'] = g.hands): void {
    // fist punches read how big each fist looks; from too far away that reading is too shaky
    const tooFar = TUNING.punchTrigger === 'extend'
      && (['l', 'r'] as const).some(side => (hands[side]?.reachNoise ?? 0) > TUNING.reachNoiseMax);
    $('hpFill').style.width = `${g.hp}%`;
    $('hpBar').classList.toggle('low', g.hp <= 30);
    $('breathFill').style.width = `${Math.round((g.breath / TUNE.breathMax) * 100)}%`;
    $('breathBar').classList.toggle('low', g.breath < TUNE.breathPunch * 2);
    $('score').textContent = String(g.score);
    $('wave').textContent = g.label ?? (g.practice ? 'Practice dummies' : `Wave ${g.wave}`);
    const pill = $('pill');
    const noHands = !hands.l && !hands.r;
    pill.classList.toggle('off', noHands);
    pill.classList.toggle('shield', g.shield.on);
    const [name, hint] = g.infernoSpreadIn > 0 ? ['BLUE INFERNO', 'now spread your hands apart!']
      : g.infernoPrep >= 1 ? ['BLUE INFERNO', g.infernoIn <= 0 ? 'hands ablaze — slam them down!' : `recharging — ${Math.ceil(g.infernoIn)}s`]
      : g.infernoPrep > 0 ? ['BLUE INFERNO', 'hold them together over your head…']
      : g.gather >= 1 ? ['FINISHER', g.ultimateIn <= 0 ? 'hands ablaze — spread them wide!' : `recharging — ${Math.ceil(g.ultimateIn)}s`]
      : g.gather > 0 ? ['GATHERING', 'hold your open hands together…']
      : g.xBlock ? ['X BLOCK', 'arms crossed: blocks attacks (not pillars or sweeps: move!)']
      : g.shield.on ? ['FLAME SHIELD', 'cover the red rings with the fire between your hands']
      : noHands ? ['NO HANDS', 'raise your fists into view']
      : tooFar ? ['GUARD', 'step closer (about 1.5 m) so fist punches can see your fists clearly']
        : ['GUARD', TUNING.punchTrigger === 'extend'
          ? 'fist: punch (pull back & hold: charge) · palm push: pillar · both palms: shield · sweep up: wall, then push · hands together, then spread: finisher'
          : 'punch & open: shoot · still open hands: shield · sweep up: wall · hands together, then spread: finisher'];
    const lost = (['l', 'r'] as const).filter(side => hands[side] && !hands[side]!.inView);
    $('modeName').textContent = name;
    $('modeHint').textContent = lost.length
      ? `${lost.map(side => (side === 'l' ? 'left' : 'right')).join(' and ')} hand out of camera view`
      : hint;
    // the attack you most need to move out of: red with what to do, green once you're clear
    const next = g.incoming()[0];
    $('dodge').classList.toggle('hidden', !next);
    if (next) {
      $('dodge').classList.toggle('safe', next.safe);
      $('dodgeText').textContent = next.safe ? '✓ CLEAR'
        : next.kind === 'slab' ? '↓ DUCK' : next.away < 0 ? '← MOVE LEFT' : 'MOVE RIGHT →';
      $('dodgeBar').style.transform = `scaleX(${next.closeness.toFixed(3)})`;
    }
    // the finisher: a small cooldown icon beside the breath bar, once you have it
    const ult = $('ult'), ready = g.ultimateCharge >= 1;
    ult.classList.toggle('hidden', !g.has('finisher'));
    ult.classList.toggle('ready', ready);
    ult.style.setProperty('--p', g.ultimateCharge.toFixed(3));
    $('ultNum').textContent = ready ? '' : String(Math.ceil(g.ultimateIn));
    const inf = $('inferno'), infReady = g.infernoIn <= 0;
    inf.classList.toggle('hidden', !g.has('inferno'));
    inf.classList.toggle('ready', infReady);
    inf.style.setProperty('--p', (1 - g.infernoIn / TUNE.infernoCooldownS).toFixed(3));
    $('infernoNum').textContent = infReady ? '' : String(Math.ceil(g.infernoIn));
  }

  onEvent(e: GameEvent): void {
    switch (e.type) {
      case 'blocked': this.toast('BLOCKED', 'cool'); break;
      case 'dodged': this.toast('✓ DODGED', 'good'); break;
      case 'clash': this.toast('CLASH', 'cool'); break;
      case 'playerHit': this.toast('HIT', 'bad'); break;
      case 'killEnemy': this.toast('+100'); break;
      case 'wall': this.toast('FIRE WALL'); break;
      case 'pillar': this.toast('PILLAR'); break;
      case 'combo': this.toast(COMBO_NAMES[e.name], e.name === 'charged' ? 'charged' : ''); break;
      case 'hint': this.toast(e.text, 'cool'); break;
      case 'ultimate': this.banner('ULTIMATE'); break;
      case 'inferno': this.banner('BLUE INFERNO'); break;
      case 'wave': this.banner(`WAVE ${e.wave}`); break;
    }
  }

  toast(text: string, cls = ''): void {
    const box = $('toasts');
    while (box.children.length > 3) box.firstChild!.remove();
    const d = document.createElement('div');
    d.className = `toast ${cls}`;
    d.textContent = text;
    box.appendChild(d);
    setTimeout(() => d.remove(), 1000);
  }

  private banner(text: string): void {
    const b = $('banner');
    b.textContent = text;
    b.classList.remove('show');
    void b.offsetWidth; // restart the CSS animation
    b.classList.add('show');
  }
}
