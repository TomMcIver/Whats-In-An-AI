/**
 * Seeded random number generator.
 *
 * Training runs need to be reproducible — if you change the learning rate
 * you want to know the difference came from that, not from a different
 * random initialisation.
 */

export class RNG {
  private s: number;

  constructor(seed = 42) {
    this.s = seed >>> 0 || 1;
  }

  /** uniform in [0, 1) — xorshift32 */
  next(): number {
    let x = this.s;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.s = x >>> 0;
    return this.s / 4294967296;
  }

  /** uniform in [lo, hi) */
  range(lo: number, hi: number): number {
    return lo + this.next() * (hi - lo);
  }

  /** standard normal, via Box-Muller */
  normal(mean = 0, sd = 1): number {
    const u = Math.max(this.next(), 1e-12);
    const v = this.next();
    return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** in-place Fisher-Yates */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  reseed(seed: number) {
    this.s = seed >>> 0 || 1;
  }
}
