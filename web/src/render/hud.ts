import { TUNE, type Game, type GameEvent } from '../game/game';
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
    $('score').textContent = String(g.score);
    $('wave').textContent = g.practice ? 'Practice dummies' : `Wave ${g.wave}`;
    const pill = $('pill');
    const noHands = !hands.l && !hands.r;
    pill.classList.toggle('off', noHands);
    pill.classList.toggle('shield', g.shield.on);
    const [name, hint] = g.xBlock ? ['X BLOCK', 'arms crossed: blocking everything that reaches you']
      : g.shield.on ? ['FLAME SHIELD', 'cover the red rings with the fire between your hands']
      : noHands ? ['NO HANDS', 'raise your fists into view']
      : tooFar ? ['GUARD', 'step closer (about 1.5 m) so fist punches can see your fists clearly']
        : ['GUARD', `${TUNING.punchTrigger === 'extend' ? 'punch' : 'punch & open'}: shoot · still open hands: shield · sweep up: wall · spread: ultimate`];
    const lost = (['l', 'r'] as const).filter(side => hands[side] && !hands[side]!.inView);
    $('modeName').textContent = name;
    $('modeHint').textContent = lost.length
      ? `${lost.map(side => (side === 'l' ? 'left' : 'right')).join(' and ')} hand out of camera view`
      : hint;
    $('shieldFill').style.width = `${Math.round(g.shield.energy * 100)}%`;
    $('shieldBar').classList.toggle('broken', g.shield.broken > 0);
    const ready = g.ultimateCharge >= 1;
    $('ultFill').style.width = `${Math.round(g.ultimateCharge * 100)}%`;
    $('ultBar').classList.toggle('ready', ready);
    $('ultState').textContent = ready ? 'ready' : `${Math.ceil(g.ultimateIn)}s`;
    $('shieldState').textContent = g.shield.broken > 0 ? 'broken' : g.shield.on ? (TUNE.shieldInfinite ? 'holding · unlimited' : 'holding')
      : g.shield.energy < 1 ? 'recharging' : 'ready';
  }

  onEvent(e: GameEvent): void {
    switch (e.type) {
      case 'blocked': this.toast('BLOCKED', 'cool'); break;
      case 'dodged': this.toast('DODGED', 'cool'); break;
      case 'clash': this.toast('CLASH', 'cool'); break;
      case 'playerHit': this.toast('HIT', 'bad'); break;
      case 'shieldBroken': this.toast('SHIELD BROKEN', 'bad'); break;
      case 'killEnemy': this.toast('+100'); break;
      case 'wall': this.toast('FIRE WALL'); break;
      case 'ultimate': this.banner('ULTIMATE'); break;
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
