/** Make whitespace visible in token labels (a space is a real part of a token). */
export function visible(text: string): string {
  return text.replace(/ /g, '␣').replace(/\n/g, '⏎').replace(/\t/g, '⇥');
}

/** Human-friendly percentage with sensible precision at both ends. */
export function pct(p: number): string {
  const v = p * 100;
  if (v >= 99.95) return '100%';
  if (v >= 10) return `${v.toFixed(0)}%`;
  if (v >= 1) return `${v.toFixed(1)}%`;
  if (v >= 0.01) return `${v.toFixed(2)}%`;
  return '<0.01%';
}

export function ordinal(n: number): string {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] ?? s[v] ?? s[0]}`;
}
