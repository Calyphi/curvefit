import { test, expect } from '@playwright/test';

test('Copy: No "v4" anywhere', async ({ page }) => {
  await page.goto('https://calyphi.com');
  const html = await page.content();
  const hasV4 = html.includes('v4') && !html.includes('ipv4');
  console.log(`Landing has "v4": ${hasV4}`);

  await page.goto('https://calyphi.com/app');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });
  const appHtml = await page.content();
  const appHasV4 = appHtml.includes('CurveFit v4');
  console.log(`App has "CurveFit v4": ${appHasV4}`);
});

test('Copy: No "Physics-driven"', async ({ page }) => {
  await page.goto('https://calyphi.com');
  const html = await page.content();
  const found = html.toLowerCase().includes('physics-driven');
  console.log(`"Physics-driven" found: ${found}`);
  expect(found).toBe(false);
});

test('Copy: No "correct model"', async ({ page }) => {
  await page.goto('https://calyphi.com/app');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });
  const text = await page.locator('body').innerText();
  const found = text.toLowerCase().includes('correct model');
  console.log(`"correct model" found: ${found}`);
  expect(found).toBe(false);

  const hasBest = text.includes('best among tested models');
  console.log(`"best among tested models" found: ${hasBest}`);
});

test('Copy: CTA consistency', async ({ page }) => {
  await page.goto('https://calyphi.com');
  const links = await page.locator('a[href="/app"]').allInnerTexts();
  console.log('CTA texts:', links);
  for (const text of links) {
    const isCorrect = text.includes('Open CurveFit');
    console.log(`  "${text}" — ${isCorrect ? 'OK' : 'INCONSISTENT'}`);
  }
});

test('Copy: Title and meta tags', async ({ page }) => {
  await page.goto('https://calyphi.com');
  const title = await page.title();
  console.log(`Landing title: "${title}"`);

  const ogTitle = await page.getAttribute('meta[property="og:title"]', 'content');
  const ogDesc = await page.getAttribute('meta[property="og:description"]', 'content');
  console.log(`OG title: "${ogTitle}"`);
  console.log(`OG desc: "${ogDesc}"`);

  const hasPhysics = [title, ogTitle, ogDesc]
    .filter(Boolean)
    .some(t => t!.toLowerCase().includes('physics-driven'));
  console.log(`Any meta has "physics-driven": ${hasPhysics}`);
});

test('Copy: Brand name consistency', async ({ page }) => {
  await page.goto('https://calyphi.com');
  const html = await page.content();

  const badForms = ['Curvefit', 'curveFit', 'curve fit', 'CURVEFIT',
                     'calyphi' ].filter(form => {
    if (form === 'calyphi') {
      return html.replace(/href="[^"]*"/g, '')
                 .replace(/https?:\/\/[^\s"<]*/g, '')
                 .includes('calyphi') &&
             !html.includes('Calyphi');
    }
    return html.includes(form);
  });

  console.log('Bad brand forms found:', badForms.length > 0 ? badForms : 'None');
});

test('Copy: Typo scan', async ({ page }) => {
  await page.goto('https://calyphi.com');
  const landingText = await page.locator('body').innerText();

  await page.goto('https://calyphi.com/app');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });
  const appText = await page.locator('body').innerText();

  const allText = landingText + ' ' + appText;

  const typos: Record<string, string> = {
    'teh ': 'the', 'recieve': 'receive', 'seperate': 'separate',
    'occured': 'occurred', 'paramters': 'parameters',
    'Levenberg Marquardt': 'Levenberg-Marquardt (needs hyphen)',
    'AICC': 'AICc', 'Aicc': 'AICc', 'aicc': 'AICc',
    'Akaiki': 'Akaike', 'Michaelis Menten': 'Michaelis-Menten',
    'nonlinear': 'check: nonlinear vs non-linear consistency',
    'optimisation': 'check: optimisation vs optimization consistency',
  };

  for (const [typo, fix] of Object.entries(typos)) {
    if (allText.includes(typo)) {
      console.log(`FOUND: "${typo}" → should be "${fix}"`);
    }
  }
});

test('Copy: Extract all visible text (landing)', async ({ page }) => {
  await page.goto('https://calyphi.com');
  const text = await page.locator('body').innerText();
  console.log('=== LANDING PAGE TEXT ===');
  console.log(text);
});

test('Copy: Extract all visible text (app)', async ({ page }) => {
  await page.goto('https://calyphi.com/app');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });
  const text = await page.locator('body').innerText();
  console.log('=== APP PAGE TEXT ===');
  console.log(text);
});

test('Copy: CI description updated', async ({ page }) => {
  await page.goto('https://calyphi.com/app');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });
  const text = await page.locator('body').innerText();

  const hasOld = text.includes('δ-method') || text.includes('(JᵀJ)');
  const hasNew = text.includes('analytical delta-method approximation');
  console.log(`Old CI text present: ${hasOld}`);
  console.log(`New CI text present: ${hasNew}`);
});

test('Copy: GitHub links present', async ({ page }) => {
  await page.goto('https://calyphi.com');
  const landingGH = await page.locator('a[href*="github.com/calyphi"]').count();
  console.log(`Landing GitHub links: ${landingGH}`);

  await page.goto('https://calyphi.com/app');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });
  const appGH = await page.locator('a[href*="github.com/calyphi"]').count();
  console.log(`App GitHub links: ${appGH}`);
});
