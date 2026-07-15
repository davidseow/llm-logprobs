import type {
  EngineInfo,
  GenerationCallbacks,
  GenerationRequest,
  GenerationStep,
  LoadProgress,
  PredictionEngine,
} from './engine';
import type { FromWorker, ToWorker } from './worker-protocol';

/** Page-side handle for the real model running in a Web Worker. */
export class WorkerEngine implements PredictionEngine {
  private worker: Worker;
  private callbacks: GenerationCallbacks | undefined;
  private onProgress: ((p: LoadProgress) => void) | undefined;
  private loadResolve: ((info: EngineInfo) => void) | undefined;
  private loadReject: ((err: Error) => void) | undefined;
  private reanalyseResolve: ((steps: GenerationStep[]) => void) | undefined;

  constructor() {
    this.worker = new Worker(new URL('../worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<FromWorker>) => this.handle(event.data);
    this.worker.onerror = (event) => {
      const err = new Error(event.message || 'worker crashed');
      this.loadReject?.(err);
      this.callbacks?.onError(err.message);
    };
  }

  private send(msg: ToWorker): void {
    this.worker.postMessage(msg);
  }

  private handle(msg: FromWorker): void {
    switch (msg.type) {
      case 'progress':
        this.onProgress?.(msg.progress);
        break;
      case 'ready':
        this.loadResolve?.({ device: msg.device, modelId: msg.modelId });
        break;
      case 'prompt-tokens':
        this.callbacks?.onPromptTokens(msg.tokens);
        break;
      case 'step':
        this.callbacks?.onStep(msg.step);
        break;
      case 'done':
        this.callbacks?.onDone(msg.reason);
        break;
      case 'reanalysed':
        this.reanalyseResolve?.(msg.steps);
        break;
      case 'error':
        this.loadReject?.(new Error(msg.message));
        this.callbacks?.onError(msg.message);
        break;
    }
  }

  load(onProgress: (p: LoadProgress) => void): Promise<EngineInfo> {
    this.onProgress = onProgress;
    return new Promise((resolve, reject) => {
      this.loadResolve = resolve;
      this.loadReject = reject;
      this.send({ type: 'load' });
    });
  }

  generate(req: GenerationRequest, on: GenerationCallbacks): void {
    this.callbacks = on;
    this.send({ type: 'generate', request: req });
  }

  cancel(): void {
    this.send({ type: 'cancel' });
  }

  reanalyse(temperature: number): Promise<GenerationStep[]> {
    return new Promise((resolve) => {
      this.reanalyseResolve = resolve;
      this.send({ type: 'reanalyse', temperature });
    });
  }
}
