import { describe, expect, it } from 'vitest';
import { analyseLogits, argmax, mulberry32, sampleFromLogits } from '../src/lib/softmax';
import { buildStep } from '../src/lib/steps';

const logits = Float32Array.from([2.0, 1.0, 0.5, -1.0, -3.0]);

describe('analyseLogits', () => {
  it('produces a normalised distribution', () => {
    const stats = analyseLogits(logits, 1.0, 5);
    const total = stats.top.reduce((acc, e) => acc + e.prob, 0) + stats.restMass;
    expect(total).toBeCloseTo(1.0, 6);
    expect(stats.restMass).toBeCloseTo(0, 6);
  });

  it('ranks the top entries by probability, descending', () => {
    const stats = analyseLogits(logits, 1.0, 3);
    expect(stats.top.map((e) => e.tokenId)).toEqual([0, 1, 2]);
    expect(stats.top[0].prob).toBeGreaterThan(stats.top[1].prob);
    expect(stats.restMass).toBeGreaterThan(0);
  });

  it('sharpens with low temperature and flattens with high temperature', () => {
    const cold = analyseLogits(logits, 0.2, 5);
    const warm = analyseLogits(logits, 2.0, 5);
    expect(cold.top[0].prob).toBeGreaterThan(warm.top[0].prob);
    expect(cold.entropyBits).toBeLessThan(warm.entropyBits);
  });

  it('matches a hand-computed softmax at temperature 1', () => {
    // exp(l) / Σ exp(l) for the leader
    const exps = Array.from(logits, (l) => Math.exp(l));
    const expected = exps[0] / exps.reduce((a, b) => a + b, 0);
    const stats = analyseLogits(logits, 1.0, 1);
    expect(stats.top[0].prob).toBeCloseTo(expected, 6);
  });

  it('reports uniform entropy of log2(n) bits', () => {
    const uniform = new Float32Array(8).fill(0);
    const stats = analyseLogits(uniform, 1.0, 4);
    expect(stats.entropyBits).toBeCloseTo(3, 5);
  });

  it('rankOf agrees with the top list', () => {
    const stats = analyseLogits(logits, 1.0, 5);
    stats.top.forEach((entry, i) => expect(stats.rankOf(entry.tokenId)).toBe(i));
  });
});

describe('sampling', () => {
  it('argmax finds the largest logit', () => {
    expect(argmax(logits)).toBe(0);
  });

  it('sampleFromLogits respects the cumulative distribution', () => {
    // u = 0 must give the first token scanned with non-zero mass; u→1 the last.
    expect(sampleFromLogits(logits, 1.0, 0)).toBe(0);
    expect(sampleFromLogits(logits, 1.0, 0.999999)).toBe(logits.length - 1);
  });

  it('sample frequencies roughly match probabilities', () => {
    const rng = mulberry32(1234);
    const counts = new Array(logits.length).fill(0);
    const n = 20000;
    for (let i = 0; i < n; i++) counts[sampleFromLogits(logits, 1.0, rng())]++;
    const stats = analyseLogits(logits, 1.0, logits.length);
    const p0 = stats.top.find((e) => e.tokenId === 0)!.prob;
    expect(counts[0] / n).toBeCloseTo(p0, 1);
  });

  it('mulberry32 is deterministic', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });
});

describe('buildStep', () => {
  it('describes a chosen token inside the top-k', () => {
    const step = buildStep(3, logits, 1, 1.0, (id) => `t${id}`);
    expect(step.index).toBe(3);
    expect(step.chosen.text).toBe('t1');
    expect(step.chosenRank).toBe(1);
    expect(step.topK.length).toBeLessThanOrEqual(10);
    expect(step.chosen.prob).toBeCloseTo(step.topK[1].prob, 10);
  });

  it('handles a chosen token outside the top-k', () => {
    // 12 logits descending by index, so token 11 is ranked last — outside a top-10.
    const wide = Float32Array.from({ length: 12 }, (_, i) => -i);
    const step = buildStep(0, wide, 11, 1.0, (id) => `t${id}`);
    expect(step.chosenRank).toBe(11);
    expect(step.topK.some((c) => c.tokenId === 11)).toBe(false);
    expect(step.chosen.prob).toBeGreaterThan(0);
    expect(step.chosen.prob).toBeLessThan(step.topK[step.topK.length - 1].prob);
  });

  it('recomputing at a new temperature keeps the chosen token but reshapes probs', () => {
    const t1 = buildStep(0, logits, 0, 1.0, (id) => `t${id}`);
    const t2 = buildStep(0, logits, 0, 2.0, (id) => `t${id}`);
    expect(t2.chosen.tokenId).toBe(t1.chosen.tokenId);
    expect(t2.chosen.prob).toBeLessThan(t1.chosen.prob);
    expect(t2.entropyBits).toBeGreaterThan(t1.entropyBits);
  });
});
