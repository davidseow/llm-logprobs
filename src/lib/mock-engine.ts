import type {
  EngineInfo,
  GenerationCallbacks,
  GenerationRequest,
  GenerationStep,
  LoadProgress,
  PredictionEngine,
  PromptToken,
} from './engine';
import { argmax, mulberry32, sampleFromLogits } from './softmax';
import { buildStep } from './steps';

/**
 * A deterministic stand-in for the real model: same interface, fabricated
 * logits. It exists so the page has an instant "demo mode" (no 100 MB
 * download) and so tests don't depend on the network. The distributions are
 * invented, and the UI labels them as a replay.
 */

const VOCAB: string[] = [
  // The scripted continuation, in order.
  ' Each',
  ' word',
  ' you',
  ' see',
  ' was',
  ' picked',
  ' from',
  ' a',
  ' list',
  ' of',
  ' guesses',
  ',',
  ' ranked',
  ' by',
  ' probability',
  '.',
  // Plausible-looking alternatives.
  ' token',
  ' letter',
  ' choice',
  ' sentence',
  ' chosen',
  ' sampled',
  ' drawn',
  ' taken',
  ' likelihood',
  ' chance',
  ' confidence',
  ' scores',
  ' the',
  ' every',
  ' this',
  ' one',
  ' can',
  ' will',
  ' might',
  ' should',
  ' and',
  ' but',
  ' so',
  ';',
  ':',
  '!',
  '?',
  ' —',
  '<|demo|>',
];

const SCRIPT_END = VOCAB.indexOf('.') + 1;
const EOS_ID = VOCAB.indexOf('<|demo|>');

/** Per-step sharpness of the fake distribution: higher = more confident. */
function sharpnessFor(index: number, rand: () => number): number {
  // Vary between very confident (e.g. after a comma) and genuinely torn,
  // so the entropy readout and the colour scale have something to show.
  const wave = 2.2 + 1.8 * Math.sin(index * 1.7);
  return Math.max(0.6, wave + rand() * 0.8);
}

function fabricateLogits(index: number, seed: number): Float32Array {
  const rand = mulberry32(seed * 7919 + index * 104729 + 13);
  const logits = new Float32Array(VOCAB.length).fill(-8);
  const scripted = index < SCRIPT_END ? index : EOS_ID;
  const sharp = sharpnessFor(index, rand);

  logits[scripted] = 4 * sharp;
  // A handful of runner-up alternatives with decaying scores.
  const alternatives = 6 + Math.floor(rand() * 4);
  for (let a = 0; a < alternatives; a++) {
    const id = SCRIPT_END + Math.floor(rand() * (VOCAB.length - SCRIPT_END - 1));
    if (id === scripted) continue;
    logits[id] = Math.max(logits[id], 4 * sharp - (a + 1) * (0.9 + rand() * 1.4));
  }
  // Low-level noise across the rest of the vocabulary.
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] <= -8) logits[i] = -8 + rand() * 3;
  }
  return logits;
}

export class MockEngine implements PredictionEngine {
  private storedLogits: Float32Array[] = [];
  private storedChosen: number[] = [];
  private cancelled = false;
  private timer: ReturnType<typeof setTimeout> | undefined;

  async load(onProgress: (p: LoadProgress) => void): Promise<EngineInfo> {
    onProgress({ file: 'demo data', loaded: 1, total: 1 });
    return { device: 'mock', modelId: 'demo replay (no model)' };
  }

  generate(req: GenerationRequest, on: GenerationCallbacks): void {
    this.cancelled = false;
    this.storedLogits = [];
    this.storedChosen = [];

    on.onPromptTokens(tokenisePrompt(req.prompt));

    const rng = mulberry32(req.seed);
    let index = 0;

    const emit = () => {
      if (this.cancelled) {
        on.onDone('cancelled');
        return;
      }
      if (index >= req.maxNewTokens) {
        on.onDone('length');
        return;
      }
      const logits = fabricateLogits(index, req.seed);
      const chosen =
        req.mode === 'greedy' ? argmax(logits) : sampleFromLogits(logits, req.temperature, rng());
      this.storedLogits.push(logits);
      this.storedChosen.push(chosen);
      on.onStep(buildStep(index, logits, chosen, req.temperature, (id) => VOCAB[id] ?? '?'));
      if (chosen === EOS_ID) {
        on.onDone('eos');
        return;
      }
      index += 1;
      this.timer = setTimeout(emit, 30);
    };
    emit();
  }

  cancel(): void {
    this.cancelled = true;
    if (this.timer !== undefined) clearTimeout(this.timer);
  }

  async reanalyse(temperature: number): Promise<GenerationStep[]> {
    return this.storedLogits.map((logits, i) =>
      buildStep(i, logits, this.storedChosen[i], temperature, (id) => VOCAB[id] ?? '?'),
    );
  }
}

/** Crude whitespace tokenisation — labelled as a demo, not the real tokeniser. */
export function tokenisePrompt(prompt: string): PromptToken[] {
  const tokens: PromptToken[] = [{ id: -1, text: '<|user|>', special: true }];
  const parts = prompt.match(/\s*\S+/g) ?? [];
  parts.forEach((text, i) => tokens.push({ id: i, text, special: false }));
  tokens.push({ id: -2, text: '<|assistant|>', special: true });
  return tokens;
}
