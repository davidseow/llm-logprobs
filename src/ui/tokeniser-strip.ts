import type { PromptToken } from '../lib/engine';
import { visible } from './format';

/** Renders the prompt as the token pieces the model actually receives. */
export function renderPromptTokens(container: HTMLElement, tokens: PromptToken[]): void {
  container.textContent = '';
  for (const token of tokens) {
    const el = document.createElement('span');
    el.className = token.special ? 'ptoken special' : 'ptoken';
    el.textContent = visible(token.text);
    if (token.special) el.title = 'Chat-template control token — usually hidden from users';
    container.appendChild(el);
  }
}
