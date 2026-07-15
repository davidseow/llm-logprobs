import type { Candidate, GenerationStep } from './engine';
import { TOP_K_DISPLAY } from './engine';
import { analyseLogits } from './softmax';

/**
 * Turn one step's raw logits + the token that was actually chosen into the
 * GenerationStep the UI consumes. Shared by the real (worker) and mock engines,
 * and re-run with a different temperature by `reanalyse`.
 */
export function buildStep(
  index: number,
  logits: Float32Array,
  chosenId: number,
  temperature: number,
  decode: (id: number) => string,
): GenerationStep {
  const stats = analyseLogits(logits, temperature, TOP_K_DISPLAY);

  const toCandidate = (tokenId: number, logit: number, prob: number, logprob: number): Candidate => ({
    tokenId,
    text: decode(tokenId),
    logit,
    prob,
    logprob,
  });

  const topK = stats.top.map((e) => toCandidate(e.tokenId, e.logit, e.prob, e.logprob));
  const inTop = stats.top.find((e) => e.tokenId === chosenId);
  const chosenLogprob = inTop ? inTop.logprob : stats.logprobOf(chosenId);
  const chosen = toCandidate(chosenId, logits[chosenId], Math.exp(chosenLogprob), chosenLogprob);

  return {
    index,
    chosen,
    chosenRank: stats.rankOf(chosenId),
    topK,
    restMass: stats.restMass,
    entropyBits: stats.entropyBits,
    temperature,
  };
}
