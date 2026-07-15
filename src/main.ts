import type { GenerationStep, LoadProgress, PredictionEngine, PromptToken } from './lib/engine';
import { MODEL_APPROX_SIZE, MODEL_ID } from './lib/engine';
import { MockEngine } from './lib/mock-engine';
import { WorkerEngine } from './lib/worker-engine';
import { Controls } from './ui/controls';
import { Inspector } from './ui/inspector';
import { Stream } from './ui/stream';
import { renderPromptTokens } from './ui/tokeniser-strip';

const app = document.querySelector('#app') as HTMLElement;

app.innerHTML = `
  <div class="wrap">
    <header class="site">
      <h1>How an LLM picks its next token</h1>
      <button class="theme-toggle" id="theme-toggle" type="button">Theme</button>
    </header>
    <p class="lede">
      A language model does not write sentences — it repeatedly answers one question:
      <em>given everything so far, how likely is each token in my vocabulary to come next?</em>
      Type a prompt, watch the answer unfold one token at a time, and click any token to see
      the choices behind it. Everything runs in your browser; nothing you type leaves this page.
    </p>

    <div class="error-banner" id="error-banner" role="alert"></div>

    <section class="card" id="loader-card">
      <h2>1 · Choose an engine</h2>
      <div class="loader-buttons">
        <button class="primary" id="load-model" type="button" data-testid="load-model">
          Load the real model (${MODEL_APPROX_SIZE})
        </button>
        <button class="secondary" id="load-demo" type="button" data-testid="load-demo">
          Instant demo (replayed data)
        </button>
      </div>
      <p class="hint" id="loader-hint">
        The real model is SmolLM2-135M — small enough to run in a browser tab. It downloads once and is
        cached. The demo mode needs no download but its probabilities are fabricated for illustration.
      </p>
      <div class="progress-track" id="progress-track" style="display:none">
        <div class="progress-fill" id="progress-fill"></div>
      </div>
      <p class="engine-badge" id="engine-badge" style="display:none" data-testid="engine-badge">
        <span class="dot"></span><span id="engine-badge-text"></span>
      </p>
    </section>

    <section class="card" id="prompt-card">
      <h2>2 · Write a prompt</h2>
      <div class="presets"></div>
      <textarea id="prompt" data-testid="prompt" placeholder="Type a short prompt…">The capital of France is</textarea>
      <div class="controls-row">
        <div class="control">
          <label for="temperature">Temperature <span class="value" id="temperature-value">0.8</span></label>
          <input type="range" id="temperature" data-testid="temperature" min="0.1" max="2" step="0.1" value="0.8" />
        </div>
        <div class="control">
          <label id="mode-label">Choice rule</label>
          <div class="seg" role="group" aria-labelledby="mode-label">
            <button id="mode-greedy" type="button" aria-pressed="false">Always top</button>
            <button id="mode-sample" type="button" aria-pressed="true">Sample</button>
          </div>
        </div>
        <div class="control">
          <label for="max-tokens">Max tokens</label>
          <select id="max-tokens">
            <option>16</option>
            <option>32</option>
            <option selected>48</option>
            <option>64</option>
          </select>
        </div>
        <div class="generate-spacer"></div>
        <button class="primary" id="generate" type="button" data-testid="generate" disabled>Generate</button>
      </div>
      <p class="hint">
        Temperature reshapes the probabilities before a token is picked: low values sharpen them
        (safe, repetitive), high values flatten them (varied, riskier). After a run, move the slider
        to see every distribution reshape — no re-generation needed.
      </p>
    </section>

    <section class="card">
      <h2>3 · What the model actually sees</h2>
      <div class="token-row" id="prompt-tokens" data-testid="prompt-tokens"></div>
      <p class="hint">
        Your prompt, split into tokens — the only units the model knows. ␣ marks a space:
        it belongs to the token. Dashed chips are chat-template control tokens that are
        normally hidden.
      </p>
    </section>

    <section class="card">
      <h2>4 · The generated tokens</h2>
      <p class="stream-empty" id="stream-empty">Nothing yet — press Generate.</p>
      <div class="token-row" id="stream" data-testid="stream"></div>
      <div class="stream-key">
        <span>less confident</span>
        <span class="ramp"></span>
        <span>more confident</span>
        <span style="margin-left:12px">dotted underline = not the top choice</span>
      </div>
    </section>

    <section class="card" id="inspector" data-testid="inspector">
      <h2>5 · Inside one decision</h2>
      <div class="inspector-head"></div>
      <div class="bars"></div>
      <p class="inspector-note"></p>
    </section>

    <section class="card explainer">
      <h2>What you are looking at</h2>
      <ol>
        <li><strong>Tokenisation.</strong> Text is split into tokens — whole words, word pieces,
          punctuation. The model never sees letters or words, only token ids.</li>
        <li><strong>Scoring.</strong> For the sequence so far, the network produces one raw score
          (a <em>logit</em>) for every token in its vocabulary — all of them, every step.</li>
        <li><strong>Softmax.</strong> The scores are turned into probabilities that sum to 1.
          Temperature divides the scores first: below 1 exaggerates the leader, above 1 levels the field.</li>
        <li><strong>Choice.</strong> One token is picked — the favourite ("always top"), or drawn at
          random in proportion to its probability ("sample"). It is appended and the loop repeats.
          That is the whole trick, repeated until an end-of-text token wins.</li>
      </ol>
      <p class="hint">
        The model here (SmolLM2-135M, Apache-2.0, by Hugging Face TB) is roughly ten-thousand times
        smaller than a frontier model, so expect clumsy prose — but the mechanism you are watching
        is exactly the same one.
      </p>
    </section>

    <footer class="site">
      Built with <a href="https://huggingface.co/docs/transformers.js">🤗 Transformers.js</a>, which runs the
      model with WebGPU (or WASM as a fallback) entirely in your browser.
      The full probability distribution is computed locally — hosted APIs typically expose at most the
      top 20 alternatives, or none at all.
      Source on <a href="https://github.com/davidseow/llm-logprobs">GitHub</a>.
    </footer>
  </div>
`;

// ---- state ----
let engine: PredictionEngine | null = null;
let steps: GenerationStep[] = [];
let generating = false;
let reanalyseTimer: ReturnType<typeof setTimeout> | undefined;

// ---- elements ----
const errorBanner = document.querySelector('#error-banner') as HTMLElement;
const loadModelBtn = document.querySelector('#load-model') as HTMLButtonElement;
const loadDemoBtn = document.querySelector('#load-demo') as HTMLButtonElement;
const progressTrack = document.querySelector('#progress-track') as HTMLElement;
const progressFill = document.querySelector('#progress-fill') as HTMLElement;
const engineBadge = document.querySelector('#engine-badge') as HTMLElement;
const engineBadgeText = document.querySelector('#engine-badge-text') as HTMLElement;
const loaderHint = document.querySelector('#loader-hint') as HTMLElement;
const promptTokensEl = document.querySelector('#prompt-tokens') as HTMLElement;

const inspector = new Inspector(document.querySelector('#inspector') as HTMLElement);
const stream = new Stream(
  document.querySelector('#stream') as HTMLElement,
  document.querySelector('#stream-empty') as HTMLElement,
  (index) => {
    stream.select(index);
    const step = steps[index];
    if (step) inspector.show(step);
  },
);
const controls = new Controls(app, {
  onGenerate: generate,
  onTemperatureChange: scheduleReanalyse,
});

// ---- theme toggle ----
const themeToggle = document.querySelector('#theme-toggle') as HTMLButtonElement;
themeToggle.addEventListener('click', () => {
  const root = document.documentElement;
  const dark =
    root.dataset.theme === 'dark' ||
    (!root.dataset.theme && window.matchMedia('(prefers-color-scheme: dark)').matches);
  root.dataset.theme = dark ? 'light' : 'dark';
});

// ---- engine loading ----
function showError(message: string): void {
  errorBanner.textContent = message;
  errorBanner.classList.add('visible');
}

async function chooseEngine(kind: 'model' | 'demo'): Promise<void> {
  errorBanner.classList.remove('visible');
  loadModelBtn.disabled = true;
  loadDemoBtn.disabled = true;

  const chosen: PredictionEngine = kind === 'model' ? new WorkerEngine() : new MockEngine();
  if (kind === 'model') {
    progressTrack.style.display = '';
    loaderHint.textContent = `Downloading ${MODEL_ID} — this happens once, then it is cached by your browser.`;
  }

  const perFile = new Map<string, LoadProgress>();
  try {
    const info = await chosen.load((p) => {
      perFile.set(p.file, p);
      let loaded = 0;
      let total = 0;
      for (const f of perFile.values()) {
        loaded += f.loaded;
        total += f.total;
      }
      if (total > 0) progressFill.style.width = `${Math.min(100, (loaded / total) * 100)}%`;
    });
    engine = chosen;
    progressTrack.style.display = 'none';
    engineBadge.style.display = '';
    engineBadgeText.textContent =
      info.device === 'mock'
        ? 'Demo mode — fabricated probabilities, no model loaded'
        : `${info.modelId} · running on ${info.device}`;
    loaderHint.textContent =
      info.device === 'mock'
        ? 'Numbers below are invented for illustration. Load the real model to see genuine ones.'
        : 'Model ready. The full next-token distribution is computed on your device at every step.';
    controls.setEnabled(true);
  } catch (err) {
    showError(
      `Could not load the model (${err instanceof Error ? err.message : String(err)}). ` +
        'Check your connection, or try the instant demo instead.',
    );
    loadModelBtn.disabled = false;
    loadDemoBtn.disabled = false;
    progressTrack.style.display = 'none';
  }
}

loadModelBtn.addEventListener('click', () => void chooseEngine('model'));
loadDemoBtn.addEventListener('click', () => void chooseEngine('demo'));

// ---- generation ----
function generate(): void {
  if (!engine || generating) return;
  if (!controls.prompt) {
    showError('Type a prompt first.');
    return;
  }
  errorBanner.classList.remove('visible');
  generating = true;
  steps = [];
  stream.reset();
  inspector.reset();
  controls.setBusy(true);

  engine.generate(controls.request(), {
    onPromptTokens(tokens: PromptToken[]) {
      renderPromptTokens(promptTokensEl, tokens);
    },
    onStep(step: GenerationStep) {
      steps.push(step);
      stream.add(step);
      // Follow the newest decision until the reader picks one to pin.
      if (stream.selectedIndex === null) inspector.show(step);
    },
    onDone() {
      generating = false;
      controls.setBusy(false);
    },
    onError(message: string) {
      generating = false;
      controls.setBusy(false);
      showError(message);
    },
  });
}

// ---- temperature re-analysis (after a run has finished) ----
function scheduleReanalyse(temperature: number): void {
  if (!engine || generating || steps.length === 0) return;
  clearTimeout(reanalyseTimer);
  reanalyseTimer = setTimeout(async () => {
    if (!engine) return;
    const updated = await engine.reanalyse(temperature);
    if (updated.length === 0) return;
    steps = updated;
    stream.update(steps);
    const pinned = stream.selectedIndex;
    const show = pinned !== null ? steps[pinned] : steps[steps.length - 1];
    if (show) inspector.show(show);
  }, 120);
}

// Test/deep-link hook: ?demo=1 boots straight into demo mode.
if (new URLSearchParams(location.search).get('demo') === '1') {
  void chooseEngine('demo');
}
