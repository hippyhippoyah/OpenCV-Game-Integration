/**
 * One Euro filter (Casiez et al.): heavy smoothing when a value is still, little lag when it moves
 * fast. `minCutoff` (Hz) sets how smooth it is at rest; `beta` how quickly it loosens with speed.
 */
export class OneEuro {
  private x: number | null = null;
  private dx = 0;

  constructor(private minCutoff: number, private beta: number, private dCutoff = 1) {}

  filter(value: number, dt: number): number {
    if (this.x === null || dt <= 0) {
      this.x ??= value;
      return this.x;
    }
    const alpha = (cutoff: number) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
    const rawDx = (value - this.x) / dt;
    this.dx += (rawDx - this.dx) * alpha(this.dCutoff);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += (value - this.x) * alpha(cutoff);
    return this.x;
  }

  get value(): number | null { return this.x; }

  reset(value: number | null = null): void {
    this.x = value;
    this.dx = 0;
  }
}
