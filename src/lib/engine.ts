/** Shared types for anything that can produce token-by-token predictions. */

export interface Candidate {
  tokenId: number;
  /** Decoded text of the token, as-is (may start with a space or be a newline). */
  text: string;
  /** Raw logit at temperature 1 (pre-softmax score from the model). */
  logit: number;
  /** Probability under the currently applied temperature. */
  prob: number;
  /** Natural-log probability under the currently applied temperature. */
  logprob: number;
}

export interface GenerationStep {
  /** 0-based position within the generated continuation. */
  index: number;
  chosen: Candidate;
  /** Rank of the chosen token in the distribution (0 = the model's favourite). */
  chosenRank: number;
  /** Top candidates by probability, descending. */
  topK: Candidate[];
  /** Probability mass of every token outside topK. */
  restMass: number;
  /** Shannon entropy of the full distribution, in bits. */
  entropyBits: number;
  /** Temperature the probabilities above were computed with. */
  temperature: number;
}

export interface PromptToken {
  id: number;
  text: string;
  /** True for chat-template control tokens (rendered dimmed in the UI). */
  special: boolean;
}

export interface GenerationRequest {
  prompt: string;
  /** 0 means greedy (argmax); otherwise softmax temperature. */
  temperature: number;
  mode: 'greedy' | 'sample';
  maxNewTokens: number;
  /** Seed for the sampler so runs are reproducible. */
  seed: number;
}

export type DoneReason = 'eos' | 'length' | 'cancelled';

export interface GenerationCallbacks {
  onPromptTokens(tokens: PromptToken[]): void;
  onStep(step: GenerationStep): void;
  onDone(reason: DoneReason): void;
  onError(message: string): void;
}

export interface LoadProgress {
  file: string;
  loaded: number;
  total: number;
}

export interface EngineInfo {
  /** e.g. 'webgpu', 'wasm', or 'mock' */
  device: string;
  modelId: string;
}

export interface PredictionEngine {
  load(onProgress: (p: LoadProgress) => void): Promise<EngineInfo>;
  generate(req: GenerationRequest, on: GenerationCallbacks): void;
  cancel(): void;
  /**
   * Recompute the last generation's per-step distributions at a different
   * temperature (the chosen tokens stay fixed — this shows how temperature
   * reshapes the distribution, not what would have been sampled instead).
   */
  reanalyse(temperature: number): Promise<GenerationStep[]>;
}

/** How many candidates each step carries for the inspector. */
export const TOP_K_DISPLAY = 10;

export const MODEL_ID = 'onnx-community/SmolLM2-135M-Instruct-ONNX';
export const MODEL_DTYPE = 'q4';
export const MODEL_APPROX_SIZE = '≈ 100 MB';
