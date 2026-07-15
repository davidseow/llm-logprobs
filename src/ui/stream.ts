import type { GenerationStep } from '../lib/engine';
import { ordinal, pct } from './format';

/**
 * The generated-token stream. Each token is washed with the series hue in
 * proportion to the probability it was chosen with (the sequential channel),
 * and tokens the sampler picked from further down the ranking get a dotted
 * underline. Text itself stays in ink.
 */
export class Stream {
  private container: HTMLElement;
  private empty: HTMLElement;
  private onSelect: (index: number) => void;
  private selected: number | null = null;

  constructor(container: HTMLElement, empty: HTMLElement, onSelect: (index: number) => void) {
    this.container = container;
    this.empty = empty;
    this.onSelect = onSelect;
  }

  reset(): void {
    this.container.textContent = '';
    this.selected = null;
    this.empty.style.display = '';
  }

  add(step: GenerationStep): void {
    this.empty.style.display = 'none';
    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'gtoken';
    el.dataset.testid = 'stream-token';
    el.dataset.index = String(step.index);
    this.applyStep(el, step);
    el.addEventListener('click', () => this.onSelect(step.index));
    this.container.appendChild(el);
  }

  /** Re-apply per-token data after a temperature re-analysis. */
  update(steps: GenerationStep[]): void {
    const els = this.container.querySelectorAll<HTMLElement>('.gtoken');
    els.forEach((el, i) => {
      const step = steps[i];
      if (step) this.applyStep(el, step);
    });
  }

  select(index: number | null): void {
    this.selected = index;
    const els = this.container.querySelectorAll<HTMLElement>('.gtoken');
    els.forEach((el, i) => el.classList.toggle('selected', i === index));
  }

  get selectedIndex(): number | null {
    return this.selected;
  }

  private applyStep(el: HTMLElement, step: GenerationStep): void {
    el.textContent = step.chosen.text;
    el.style.setProperty('--p', step.chosen.prob.toFixed(3));
    el.classList.toggle('surprise', step.chosenRank > 0);
    el.title =
      `p = ${pct(step.chosen.prob)}` +
      (step.chosenRank > 0 ? ` — sampled ${ordinal(step.chosenRank + 1)} choice` : ' — top choice') +
      ' · click to inspect';
  }
}
