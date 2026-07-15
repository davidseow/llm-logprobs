import { describe, expect, it } from 'vitest';
import type { DoneReason, GenerationStep, PromptToken } from '../src/lib/engine';
import { MockEngine, tokenisePrompt } from '../src/lib/mock-engine';

function run(engine: MockEngine, mode: 'greedy' | 'sample', temperature = 0.8) {
  return new Promise<{ steps: GenerationStep[]; prompt: PromptToken[]; reason: DoneReason }>(
    (resolve, reject) => {
      const steps: GenerationStep[] = [];
      let prompt: PromptToken[] = [];
      engine.generate(
        { prompt: 'Hello there', temperature, mode, maxNewTokens: 40, seed: 7 },
        {
          onPromptTokens: (t) => (prompt = t),
          onStep: (s) => steps.push(s),
          onDone: (reason) => resolve({ steps, prompt, reason }),
          onError: (m) => reject(new Error(m)),
        },
      );
    },
  );
}

describe('MockEngine', () => {
  it('loads instantly and identifies itself as a mock', async () => {
    const engine = new MockEngine();
    const info = await engine.load(() => {});
    expect(info.device).toBe('mock');
  });

  it('emits prompt tokens with chat-template specials', async () => {
    const engine = new MockEngine();
    const { prompt } = await run(engine, 'greedy');
    expect(prompt[0].special).toBe(true);
    expect(prompt.some((t) => !t.special)).toBe(true);
  });

  it('greedy mode replays the scripted continuation and ends at eos', async () => {
    const engine = new MockEngine();
    const { steps, reason } = await run(engine, 'greedy');
    expect(reason).toBe('eos');
    expect(steps.length).toBeGreaterThan(10);
    for (const step of steps) {
      expect(step.chosenRank).toBe(0);
      const mass = step.topK.reduce((a, c) => a + c.prob, 0) + step.restMass;
      expect(mass).toBeCloseTo(1, 5);
      expect(step.entropyBits).toBeGreaterThan(0);
    }
    const text = steps.map((s) => s.chosen.text).join('');
    expect(text).toContain('probability');
  });

  it('reanalyse keeps chosen tokens and reshapes the distribution', async () => {
    const engine = new MockEngine();
    const { steps } = await run(engine, 'greedy');
    const hot = await engine.reanalyse(2.0);
    expect(hot.length).toBe(steps.length);
    hot.forEach((step, i) => {
      expect(step.chosen.tokenId).toBe(steps[i].chosen.tokenId);
      expect(step.entropyBits).toBeGreaterThan(steps[i].entropyBits);
    });
  });

  it('is deterministic for a fixed seed when sampling', async () => {
    const a = await run(new MockEngine(), 'sample');
    const b = await run(new MockEngine(), 'sample');
    expect(a.steps.map((s) => s.chosen.tokenId)).toEqual(b.steps.map((s) => s.chosen.tokenId));
  });
});

describe('tokenisePrompt', () => {
  it('keeps leading whitespace attached to tokens', () => {
    const tokens = tokenisePrompt('Hello brave world');
    const texts = tokens.filter((t) => !t.special).map((t) => t.text);
    expect(texts).toEqual(['Hello', ' brave', ' world']);
  });
});
