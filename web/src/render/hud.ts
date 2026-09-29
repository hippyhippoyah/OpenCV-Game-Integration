import { TUNE, type ComboName, type Game, type GameEvent } from '../game/game';

const COMBO_NAMES: Record<ComboName, string> = {
  charged: 'CHARGED', flurry: 'FLURRY', counter: 'COUNTER', oneTwo: 'ONE-TWO PUSH', volley: 'PILLAR VOLLEY',
  wallBreaker: 'WALL BREAKER', finisher: 'FINISHER',
};
import { palmsFaceEachOther, TUNING, type Intent } from '../intent/interpret';

const $ = (id: string) => document.getElementById(id)!;

/** DOM overlay: health, breath, current move, dodge cue, toasts and banners. */
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
    $('place').textContent = g.label ?? (g.practice ? 'Training' : '');
    // both hands open and still-ish, but the palms don't face each other: say how to make a shield
    const openPalmsApart = !!hands.l?.open && !!hands.r?.open && !!hands.l.palm && !!hands.r.palm
      && !palmsFaceEachOther(hands.l, hands.r, TUNING.shieldFacing);
    const pill = $('pill');
    const noHands = !hands.l && !hands.r;
    pill.classList.toggle('off', noHands);
    pill.classList.toggle('shield', g.shield.on);
    // a word or two, never a paragraph: the name of what you're doing, and what to do next
    const [name, hint] = g.lightningCharged ? ['LIGHTNING', `aim and thrust! ${Math.ceil(g.lightningLeft)}`]
      : g.lightningCalling > 0 ? ['LIGHTNING', 'hold them up…']
      : g.infernoSpreadIn > 0 ? ['BLUE INFERNO', 'spread your hands!']
      : g.infernoPrep >= 1 ? ['BLUE INFERNO', g.infernoIn <= 0 ? 'slam them down!' : `recharging ${Math.ceil(g.infernoIn)}s`]
      : g.infernoPrep > 0 ? ['BLUE INFERNO', 'hold them overhead…']
      : g.gather >= 1 ? ['FINISHER', g.ultimateIn <= 0 ? 'spread them wide!' : `recharging ${Math.ceil(g.ultimateIn)}s`]
      : g.gather > 0 ? ['GATHERING', 'hold them together…']
      : g.xBlock ? ['X BLOCK', 'blocks orbs']
      : g.shield.on ? ['FLAME SHIELD', 'cover the red rings']
      : noHands ? ['NO HANDS', 'raise your fists']
      : openPalmsApart ? ['OPEN PALMS', 'face them together: shield']
      : tooFar ? ['GUARD', 'step closer (about 1.5 m)']
      : ['GUARD', ''];
    const lost = (['l', 'r'] as const).filter(side => hands[side] && !hands[side]!.inView);
    $('modeName').textContent = name;
    $('modeHint').textContent = lost.length
      ? `${lost.map(side => (side === 'l' ? 'left' : 'right')).join(' and ')} hand out of view`
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
    // lightning: its own recharge, beside the others
    const bolt = $('lightning'), boltReady = g.lightningIn <= 0;
    bolt.classList.toggle('hidden', !g.has('lightning'));
    bolt.classList.toggle('ready', boltReady);
    bolt.style.setProperty('--p', (1 - g.lightningIn / TUNE.lightningCooldownS).toFixed(3));
    $('lightningNum').textContent = boltReady ? '' : String(Math.ceil(g.lightningIn));
  }

  onEvent(e: GameEvent): void {
    switch (e.type) {
      case 'blocked': this.toast('BLOCKED', 'cool'); break;
      case 'dodged': this.toast('✓ DODGED', 'good'); break;
      case 'clash': this.toast('CLASH', 'cool'); break;
      case 'playerHit': this.toast('HIT', 'bad'); break;
      case 'wall': this.toast('FIRE WALL'); break;
      case 'pillar': this.toast('PILLAR'); break;
      case 'combo': this.toast(COMBO_NAMES[e.name], e.name === 'charged' ? 'charged' : ''); break;
      case 'hint': this.toast(e.text, 'cool'); break;
      case 'ultimate': this.banner('ULTIMATE'); break;
      case 'inferno': this.banner('BLUE INFERNO'); break;
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
