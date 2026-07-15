import type { DoneReason, GenerationRequest, GenerationStep, LoadProgress, PromptToken } from './engine';

/** Messages from the page to the model worker. */
export type ToWorker =
  | { type: 'load' }
  | { type: 'generate'; request: GenerationRequest }
  | { type: 'cancel' }
  | { type: 'reanalyse'; temperature: number };

/** Messages from the model worker back to the page. */
export type FromWorker =
  | { type: 'progress'; progress: LoadProgress }
  | { type: 'ready'; device: string; modelId: string }
  | { type: 'prompt-tokens'; tokens: PromptToken[] }
  | { type: 'step'; step: GenerationStep }
  | { type: 'done'; reason: DoneReason }
  | { type: 'reanalysed'; steps: GenerationStep[] }
  | { type: 'error'; message: string };
