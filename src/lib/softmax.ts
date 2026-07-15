/**
 * Distribution maths shared by the real and mock engines.
 * All functions take raw logits (temperature 1) and apply temperature here,
 * so a stored Float32Array per step is enough to re-analyse at any temperature.
 */

export interface TopEntry {
  tokenId: number;
  logit: number;
  prob: number;
  logprob: number;
}

export interface DistributionStats {
  top: TopEntry[];
  restMass: number;
  entropyBits: number;
  /** logprob of an arbitrary token id under this distribution. */
  logprobOf(tokenId: number): number;
  /** rank of a token id (0 = highest probability). */
  rankOf(tokenId: number): number;
}

/** Temperatures below this are treated as "effectively greedy" to avoid overflow. */
export const MIN_TEMPERATURE = 0.05;

/**
 * One pass over the logits: softmax at the given temperature, entropy,
 * and the top-N entries. O(V·N) with small N.
 */
export function analyseLogits(logits: Float32Array, temperature: number, topN: number): DistributionStats {
  const t = Math.max(temperature, MIN_TEMPERATURE);
  const n = logits.length;

  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = logits[i] / t;
    if (v > max) max = v;
  }

  let z = 0;
  for (let i = 0; i < n; i++) {
    z += Math.exp(logits[i] / t - max);
  }
  const logZ = max + Math.log(z);

  // Entropy in bits: -Σ p·log2(p) with p = exp(l/t - logZ)
  let entropyNats = 0;
  for (let i = 0; i < n; i++) {
    const lp = logits[i] / t - logZ;
    const p = Math.exp(lp);
    if (p > 0) entropyNats -= p * lp;
  }
  const entropyBits = entropyNats / Math.LN2;

  // Top-N by logit (same order as by prob at any fixed positive temperature).
  const topIds: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = logits[i];
    if (topIds.length === topN && v <= logits[topIds[topIds.length - 1]]) continue;
    let lo = 0;
    let hi = topIds.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (logits[topIds[mid]] >= v) lo = mid + 1;
      else hi = mid;
    }
    topIds.splice(lo, 0, i);
    if (topIds.length > topN) topIds.pop();
  }

  const top: TopEntry[] = topIds.map((tokenId) => {
    const logprob = logits[tokenId] / t - logZ;
    return { tokenId, logit: logits[tokenId], prob: Math.exp(logprob), logprob };
  });
  const topMass = top.reduce((acc, e) => acc + e.prob, 0);

  return {
    top,
    restMass: Math.max(0, 1 - topMass),
    entropyBits,
    logprobOf(tokenId: number): number {
      return logits[tokenId] / t - logZ;
    },
    rankOf(tokenId: number): number {
      const v = logits[tokenId];
      let rank = 0;
      for (let i = 0; i < n; i++) {
        if (logits[i] > v || (logits[i] === v && i < tokenId)) rank++;
      }
      return rank;
    },
  };
}

/** Argmax over raw logits (temperature-independent). */
export function argmax(logits: Float32Array): number {
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

/**
 * Sample a token id from softmax(logits / temperature) using the provided
 * uniform random number in [0, 1).
 */
export function sampleFromLogits(logits: Float32Array, temperature: number, u: number): number {
  const t = Math.max(temperature, MIN_TEMPERATURE);
  const n = logits.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = logits[i] / t;
    if (v > max) max = v;
  }
  let z = 0;
  for (let i = 0; i < n; i++) z += Math.exp(logits[i] / t - max);

  let acc = 0;
  const target = u * z;
  for (let i = 0; i < n; i++) {
    acc += Math.exp(logits[i] / t - max);
    if (acc >= target) return i;
  }
  return n - 1;
}

/** Deterministic PRNG (mulberry32) so sampled runs are reproducible. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
