import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test('A11y: Landing page', async ({ page }) => {
  await page.goto('https://calyphi.com');
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa'])
    .analyze();
  console.log('Landing violations:', results.violations.length);
  for (const v of results.violations) {
    console.log(`[${v.impact}] ${v.id}: ${v.description}`);
    console.log(`  Affected: ${v.nodes.length} elements`);
    for (const n of v.nodes.slice(0, 3)) {
      console.log(`    ${n.target[0]}`);
    }
  }
});

test('A11y: App with demo fit', async ({ page }) => {
  await page.goto('https://calyphi.com/app');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa'])
    .analyze();
  console.log('App violations:', results.violations.length);
  for (const v of results.violations) {
    console.log(`[${v.impact}] ${v.id}: ${v.description}`);
    console.log(`  Affected: ${v.nodes.length} elements`);
    for (const n of v.nodes.slice(0, 3)) {
      console.log(`    ${n.target[0]}`);
    }
  }
});

test('A11y: Pro modal', async ({ page }) => {
  await page.goto('https://calyphi.com/app');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });
  await page.click('text=Export PDF');
  await page.waitForSelector('text=Coming soon', { timeout: 5000 });
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa'])
    .analyze();
  console.log('Modal violations:', results.violations.length);
  for (const v of results.violations) {
    console.log(`[${v.impact}] ${v.id}: ${v.description}`);
    for (const n of v.nodes.slice(0, 3)) {
      console.log(`    ${n.target[0]}`);
    }
  }
});

test('Keyboard: Tab through app controls', async ({ page }) => {
  await page.goto('https://calyphi.com/app');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });

  const focusable: string[] = [];
  let lastTag = '';
  for (let i = 0; i < 50; i++) {
    await page.keyboard.press('Tab');
    const tag = await page.evaluate(() => {
      const el = document.activeElement;
      return el ? `${el.tagName}:${el.textContent?.slice(0,30) || el.getAttribute('aria-label') || ''}` : 'NONE';
    });
    if (tag === lastTag) break;
    lastTag = tag;
    focusable.push(tag);
  }
  console.log('Tab order:', focusable.join(' → '));

  const hasFocusStyle = await page.evaluate(() => {
    const el = document.activeElement;
    if (!el) return false;
    const style = window.getComputedStyle(el);
    return style.outline !== 'none' || style.boxShadow !== 'none';
  });
  console.log('Focus visible on last element:', hasFocusStyle);
});

test('Color contrast: Critical text elements', async ({ page }) => {
  await page.goto('https://calyphi.com/app');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });
  const results = await new AxeBuilder({ page })
    .withRules(['color-contrast'])
    .analyze();
  console.log('Contrast violations:', results.violations.length);
  for (const v of results.violations) {
    for (const n of v.nodes) {
      console.log(`  ${n.target[0]}: ${n.message}`);
    }
  }
});
