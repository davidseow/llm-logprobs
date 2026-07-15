# How an LLM picks its next token

An interactive, single-page explainer: type a prompt, watch a real language model generate
text **token by token**, and inspect the probability distribution behind every choice.
Everything runs in the browser — there is no server and nothing you type leaves the page.

## What it teaches

1. **Tokenisation** — the prompt is shown as the token pieces the model actually receives,
   including the chat-template control tokens that are normally hidden.
2. **Scoring** — at every step the model assigns a raw score (logit) to *every* token in its
   vocabulary; the top ten are drawn as a ranked bar chart, with the rest folded into an
   "all other tokens" bar and an entropy readout.
3. **Softmax and temperature** — after a run, moving the temperature slider recomputes every
   step's distribution from the stored logits, live, without re-running the model.
4. **Choice** — greedy ("always top") versus sampling; tokens the sampler picked from further
   down the ranking get a dotted underline, and each token's background is washed in
   proportion to the probability it was chosen with.

## The two engines

- **Real model** — [SmolLM2-135M-Instruct](https://huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX)
  (Apache-2.0, by Hugging Face TB), run with
  [🤗 Transformers.js](https://huggingface.co/docs/transformers.js) at 4-bit quantisation on
  WebGPU, falling back to WASM. It is a ~100 MB one-off download, cached by the browser.
  Because the forward pass runs locally, the page computes the **full** next-token
  distribution — hosted APIs typically expose at most the top 20 alternatives
  (OpenAI `top_logprobs`), or none at all (Anthropic).
- **Instant demo** — a deterministic mock with fabricated distributions, clearly labelled,
  so the page works with zero download (and so tests don't need the network).

A 135M-parameter model writes clumsy prose; the point is that the *mechanism* — score,
softmax, sample, repeat — is identical in frontier models.

## Development

```bash
npm install
npm run dev        # local dev server
npm test           # unit tests (distribution maths, mock engine)
npm run e2e        # Playwright end-to-end tests against the demo engine
npm run build      # typecheck + production build
```

The e2e suite runs against demo mode (`/?demo=1`); testing the real model needs a browser
with network access to huggingface.co. Deployment to GitHub Pages is automated in
`.github/workflows/deploy.yml` (pushes to `main`).

## How the real engine works

`src/worker.ts` runs in a Web Worker: it tokenises with the model's chat template, then loops —
one `forward()` call per step, softmax (with temperature) over the last position's logits,
top-k + entropy extraction, then greedy or seeded sampling. Raw logits for each step are kept
in the worker so the temperature slider can re-analyse a finished run exactly, at any
temperature, without regenerating. The maths lives in `src/lib/softmax.ts` and is shared with
the mock engine.
