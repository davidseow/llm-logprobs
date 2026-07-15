import { defineConfig } from 'vite';

export default defineConfig({
  // GitHub Pages serves project sites from a sub-path; CI sets BASE_PATH=/llm-logprobs/
  base: process.env.BASE_PATH ?? '/',
  build: {
    target: 'es2022',
  },
});
