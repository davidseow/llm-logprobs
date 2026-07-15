import { defineConfig } from 'vite';
import { configDefaults } from 'vitest/config';

export default defineConfig({
  // GitHub Pages serves project sites from a sub-path; CI sets BASE_PATH=/llm-logprobs/
  base: process.env.BASE_PATH ?? '/',
  build: {
    target: 'es2022',
  },
  test: {
    // Keep vitest to the unit tests in tests/**; the e2e/*.spec.ts files are Playwright.
    exclude: [...configDefaults.exclude, 'e2e/**'],
  },
});
