import { expect, test } from '@playwright/test';

// All e2e runs use demo mode (?demo=1): the real model needs a ~100 MB
// download, which CI and sandboxes cannot rely on. Demo mode exercises the
// identical UI pipeline with fabricated, deterministic distributions.

test('generates tokens step by step and inspects a decision', async ({ page }) => {
  await page.goto('/?demo=1');

  await expect(page.getByTestId('engine-badge')).toContainText('Demo mode');

  const generate = page.getByTestId('generate');
  await expect(generate).toBeEnabled();
  await generate.click();

  // Prompt is tokenised, including dimmed chat-template specials.
  await expect(page.getByTestId('prompt-tokens').locator('.ptoken.special').first()).toBeVisible();

  // Tokens appear one by one; wait for the scripted run to finish.
  const tokens = page.getByTestId('stream-token');
  await expect(tokens.nth(5)).toBeVisible();
  await expect(generate).toBeEnabled({ timeout: 15_000 });
  const count = await tokens.count();
  expect(count).toBeGreaterThan(10);

  // The inspector follows generation, then pins the clicked token.
  await tokens.nth(2).click();
  await expect(page.getByTestId('inspector')).toContainText('Step');
  const rows = page.getByTestId('bar-row');
  expect(await rows.count()).toBeGreaterThanOrEqual(5);
  await expect(page.getByTestId('inspector')).toContainText('✓ chosen');
});

test('temperature slider reshapes the inspected distribution without regenerating', async ({
  page,
}) => {
  await page.goto('/?demo=1');
  await page.getByTestId('generate').click();
  await expect(page.getByTestId('generate')).toBeEnabled({ timeout: 15_000 });

  const tokens = page.getByTestId('stream-token');
  await tokens.nth(1).click();
  const firstValue = page.getByTestId('bar-value').first();
  const before = await firstValue.textContent();
  const countBefore = await tokens.count();

  await page.getByTestId('temperature').fill('2');
  await expect(firstValue).not.toHaveText(before ?? '', { timeout: 5_000 });
  // Same tokens, new numbers: reanalysis must not regenerate.
  expect(await tokens.count()).toBe(countBefore);
});
