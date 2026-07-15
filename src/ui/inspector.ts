import type { Candidate, GenerationStep } from '../lib/engine';
import { ordinal, pct, visible } from './format';

/**
 * The per-step distribution view: a ranked horizontal bar chart of the top
 * candidates (one series, so every bar wears the same hue; the chosen row is
 * marked by weight and a check, never by a different colour), an
 * "all other tokens" residual bar in neutral grey, and headline stats.
 */
export class Inspector {
  private head: HTMLElement;
  private bars: HTMLElement;
  private note: HTMLElement;

  constructor(root: HTMLElement) {
    this.head = root.querySelector('.inspector-head') as HTMLElement;
    this.bars = root.querySelector('.bars') as HTMLElement;
    this.note = root.querySelector('.inspector-note') as HTMLElement;
    this.reset();
  }

  reset(): void {
    this.head.textContent = '';
    this.bars.textContent = '';
    this.note.textContent =
      'Generate some text, then click any token above to see the choices the model weighed up at that moment.';
  }

  show(step: GenerationStep): void {
    this.renderHead(step);
    this.renderBars(step);
    const surprise =
      step.chosenRank > 0
        ? ` Here the sampler picked the ${ordinal(step.chosenRank + 1)}-ranked token rather than the favourite — that is temperature at work.`
        : '';
    this.note.textContent =
      `Bar length is the probability the model assigned at temperature ${step.temperature.toFixed(1)}. ` +
      `Every token in the vocabulary gets a score; only the top ${step.topK.length} are shown.` +
      surprise;
  }

  private renderHead(step: GenerationStep): void {
    this.head.textContent = '';
    const stats: Array<[string, string | HTMLElement]> = [
      ['Step', `${step.index + 1}`],
      ['Chosen token', tokenChip(step.chosen)],
      ['Probability', pct(step.chosen.prob)],
      ['Rank', step.chosenRank === 0 ? '1st (favourite)' : ordinal(step.chosenRank + 1)],
      ['Uncertainty', `${step.entropyBits.toFixed(2)} bits`],
    ];
    for (const [label, value] of stats) {
      const stat = document.createElement('div');
      stat.className = 'stat';
      const l = document.createElement('span');
      l.className = 'label';
      l.textContent = label;
      const v = document.createElement('span');
      v.className = 'value';
      if (typeof value === 'string') v.textContent = value;
      else v.appendChild(value);
      stat.append(l, v);
      this.head.appendChild(stat);
    }
  }

  private renderBars(step: GenerationStep): void {
    this.bars.textContent = '';
    const chosenInTop = step.topK.some((c) => c.tokenId === step.chosen.tokenId);

    for (const candidate of step.topK) {
      this.addRow(
        visible(candidate.text),
        candidate.prob,
        candidate.tokenId === step.chosen.tokenId,
        false,
      );
    }
    this.addRow('all other tokens', step.restMass, false, true);
    if (!chosenInTop) {
      this.addRow(
        `${visible(step.chosen.text)} (ranked ${ordinal(step.chosenRank + 1)})`,
        step.chosen.prob,
        true,
        false,
      );
    }
  }

  private addRow(label: string, prob: number, chosen: boolean, rest: boolean): void {
    const row = document.createElement('div');
    row.className = chosen ? 'bar-row chosen' : 'bar-row';
    row.dataset.testid = 'bar-row';

    const labelEl = document.createElement('span');
    labelEl.className = 'bar-label';
    labelEl.textContent = label;

    const track = document.createElement('div');
    track.className = 'bar-track';
    const fill = document.createElement('div');
    fill.className = rest ? 'bar-fill rest' : 'bar-fill';
    fill.style.width = `${Math.max(0.3, prob * 100)}%`;
    track.appendChild(fill);

    const value = document.createElement('span');
    value.className = 'bar-value';
    value.dataset.testid = 'bar-value';
    value.textContent = pct(prob);
    if (chosen) {
      const mark = document.createElement('span');
      mark.className = 'chosen-mark';
      mark.textContent = ' ✓ chosen';
      value.appendChild(mark);
    }

    row.append(labelEl, track, value);
    this.bars.appendChild(row);
  }
}

function tokenChip(candidate: Candidate): HTMLElement {
  const chip = document.createElement('span');
  chip.className = 'tok';
  chip.textContent = visible(candidate.text);
  return chip;
}
