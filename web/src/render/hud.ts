import type { Game, GameEvent } from '../game/game';

const $ = (id: string) => document.getElementById(id)!;

/** DOM overlay: health, score, current move, shield energy, toasts and wave banners. */
export class Hud {
  update(g: Game): void {
    $('hpFill').style.width = `${g.hp}%`;
    $('score').textContent = String(g.score);
    $('wave').textContent = `Wave ${g.wave}`;
    const pill = $('pill');
    pill.classList.toggle('off', !g.fire.held && !g.shield.on);
    pill.classList.toggle('shield', g.shield.on);
    const [name, hint] = g.shield.on ? ['FLAME SHIELD', 'cover the red rings · drains while held']
      : g.fire.held ? ['FIREBALL', 'push toward the screen to throw']
        : ['NO FIRE', 'raise hands, palms together · or spread wide to shield'];
    $('modeName').textContent = name;
    $('modeHint').textContent = hint;
    $('shieldFill').style.width = `${Math.round(g.shield.energy * 100)}%`;
    $('shieldBar').classList.toggle('broken', g.shield.broken > 0);
    $('shieldState').textContent = g.shield.broken > 0 ? 'broken' : g.shield.on ? 'holding' : g.shield.energy < 1 ? 'recharging' : 'ready';
  }

  onEvent(e: GameEvent): void {
    switch (e.type) {
      case 'blocked': this.toast('BLOCKED', 'cool'); break;
      case 'dodged': this.toast('DODGED', 'cool'); break;
      case 'clash': this.toast('CLASH', 'cool'); break;
      case 'playerHit': this.toast('HIT', 'bad'); break;
      case 'shieldBroken': this.toast('SHIELD BROKEN', 'bad'); break;
      case 'killEnemy': this.toast('+100'); break;
      case 'wave': this.banner(`WAVE ${e.wave}`); break;
    }
  }

  private toast(text: string, cls = ''): void {
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
