import type { GenerationRequest } from '../lib/engine';

const PRESETS = [
  'The capital of France is',
  'Once upon a time, there was a',
  'Write one sentence about the sea.',
  'What is 2 + 2?',
];

export interface ControlsHandlers {
  onGenerate(): void;
  onTemperatureChange(t: number): void;
}

/** Prompt box, presets and generation settings. */
export class Controls {
  private promptEl: HTMLTextAreaElement;
  private tempEl: HTMLInputElement;
  private tempValueEl: HTMLElement;
  private greedyBtn: HTMLButtonElement;
  private sampleBtn: HTMLButtonElement;
  private maxTokensEl: HTMLSelectElement;
  private generateBtn: HTMLButtonElement;
  private modeValue: 'greedy' | 'sample' = 'sample';

  constructor(root: HTMLElement, handlers: ControlsHandlers) {
    this.promptEl = root.querySelector('#prompt') as HTMLTextAreaElement;
    this.tempEl = root.querySelector('#temperature') as HTMLInputElement;
    this.tempValueEl = root.querySelector('#temperature-value') as HTMLElement;
    this.greedyBtn = root.querySelector('#mode-greedy') as HTMLButtonElement;
    this.sampleBtn = root.querySelector('#mode-sample') as HTMLButtonElement;
    this.maxTokensEl = root.querySelector('#max-tokens') as HTMLSelectElement;
    this.generateBtn = root.querySelector('#generate') as HTMLButtonElement;

    const presetsEl = root.querySelector('.presets') as HTMLElement;
    for (const preset of PRESETS) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'preset-chip';
      chip.textContent = preset;
      chip.addEventListener('click', () => {
        this.promptEl.value = preset;
      });
      presetsEl.appendChild(chip);
    }

    this.tempEl.addEventListener('input', () => {
      this.tempValueEl.textContent = this.temperature.toFixed(1);
      handlers.onTemperatureChange(this.temperature);
    });
    this.greedyBtn.addEventListener('click', () => this.setMode('greedy'));
    this.sampleBtn.addEventListener('click', () => this.setMode('sample'));
    this.generateBtn.addEventListener('click', () => handlers.onGenerate());
    this.tempValueEl.textContent = this.temperature.toFixed(1);
  }

  private setMode(mode: 'greedy' | 'sample'): void {
    this.modeValue = mode;
    this.greedyBtn.setAttribute('aria-pressed', String(mode === 'greedy'));
    this.sampleBtn.setAttribute('aria-pressed', String(mode === 'sample'));
  }

  get temperature(): number {
    return Number(this.tempEl.value);
  }

  get prompt(): string {
    return this.promptEl.value.trim();
  }

  request(): GenerationRequest {
    return {
      prompt: this.prompt,
      temperature: this.temperature,
      mode: this.modeValue,
      maxNewTokens: Number(this.maxTokensEl.value),
      seed: Math.floor(Math.random() * 2 ** 31),
    };
  }

  setBusy(busy: boolean): void {
    this.generateBtn.disabled = busy;
    this.generateBtn.textContent = busy ? 'Generating…' : 'Generate';
  }

  setEnabled(enabled: boolean): void {
    this.generateBtn.disabled = !enabled;
  }
}
