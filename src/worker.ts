/// <reference lib="webworker" />
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  Tensor,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from '@huggingface/transformers';
import { MODEL_DTYPE, MODEL_ID, type GenerationRequest, type PromptToken } from './lib/engine';
import { argmax, mulberry32, sampleFromLogits } from './lib/softmax';
import { buildStep } from './lib/steps';
import type { FromWorker, ToWorker } from './lib/worker-protocol';

let tokenizer: PreTrainedTokenizer;
let model: PreTrainedModel;
let cancelled = false;

// The last generation's raw logits and chosen ids, kept here so the page can
// ask for the distributions to be recomputed at a different temperature
// without re-running the model.
let storedLogits: Float32Array[] = [];
let storedChosen: number[] = [];

const post = (msg: FromWorker) => self.postMessage(msg);

async function load(): Promise<void> {
  const progress_callback = (p: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (p.status === 'progress' && p.file) {
      post({ type: 'progress', progress: { file: p.file, loaded: p.loaded ?? 0, total: p.total ?? 0 } });
    }
  };

  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback });

  let device = 'webgpu';
  try {
    model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      dtype: MODEL_DTYPE,
      device: 'webgpu',
      progress_callback,
    });
  } catch {
    device = 'wasm';
    model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      dtype: MODEL_DTYPE,
      device: 'wasm',
      progress_callback,
    });
  }
  post({ type: 'ready', device, modelId: MODEL_ID });
}

function decodeToken(id: number): string {
  return tokenizer.decode([id], { skip_special_tokens: false });
}

function promptTokens(ids: number[]): PromptToken[] {
  const special = new Set(tokenizer.all_special_ids ?? []);
  return ids.map((id) => ({ id, text: decodeToken(id), special: special.has(id) }));
}

async function lastPositionLogits(ids: number[]): Promise<Float32Array> {
  const n = ids.length;
  const input_ids = new Tensor('int64', BigInt64Array.from(ids, BigInt), [1, n]);
  const attention_mask = new Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]);
  const output = await model({ input_ids, attention_mask });
  const logits = output.logits; // [1, seq, vocab]
  const [, seq, vocab] = logits.dims as number[];
  const data = logits.data as Float32Array;
  return Float32Array.from(data.subarray((seq - 1) * vocab, seq * vocab));
}

async function generate(req: GenerationRequest): Promise<void> {
  cancelled = false;
  storedLogits = [];
  storedChosen = [];

  const ids = tokenizer.apply_chat_template([{ role: 'user', content: req.prompt }], {
    add_generation_prompt: true,
    tokenize: true,
    return_tensor: false,
    return_dict: false,
  }) as number[];
  post({ type: 'prompt-tokens', tokens: promptTokens(ids) });

  const eosId = tokenizer.eos_token_id;
  const rng = mulberry32(req.seed);

  for (let index = 0; index < req.maxNewTokens; index++) {
    if (cancelled) {
      post({ type: 'done', reason: 'cancelled' });
      return;
    }
    const logits = await lastPositionLogits(ids);
    const chosen =
      req.mode === 'greedy' ? argmax(logits) : sampleFromLogits(logits, req.temperature, rng());
    storedLogits.push(logits);
    storedChosen.push(chosen);
    post({ type: 'step', step: buildStep(index, logits, chosen, req.temperature, decodeToken) });
    if (chosen === eosId) {
      post({ type: 'done', reason: 'eos' });
      return;
    }
    ids.push(chosen);
  }
  post({ type: 'done', reason: 'length' });
}

function reanalyse(temperature: number): void {
  const steps = storedLogits.map((logits, i) =>
    buildStep(i, logits, storedChosen[i], temperature, decodeToken),
  );
  post({ type: 'reanalysed', steps });
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const msg = event.data;
  try {
    switch (msg.type) {
      case 'load':
        await load();
        break;
      case 'generate':
        await generate(msg.request);
        break;
      case 'cancel':
        cancelled = true;
        break;
      case 'reanalyse':
        reanalyse(msg.temperature);
        break;
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
