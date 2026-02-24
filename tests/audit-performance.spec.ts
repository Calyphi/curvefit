import { test } from '@playwright/test';

test('Perf: Landing page load', async ({ page }) => {
  const client = await page.context().newCDPSession(page);
  await client.send('Performance.enable');

  const start = Date.now();
  const response = await page.goto('https://calyphi.com');
  await page.waitForLoadState('networkidle');
  const loadTime = Date.now() - start;

  console.log(`Status: ${response?.status()}`);
  console.log(`Load time: ${loadTime}ms`);

  const metrics = await client.send('Performance.getMetrics');
  for (const m of metrics.metrics) {
    if (['JSHeapUsedSize', 'JSHeapTotalSize', 'LayoutCount',
         'RecalcStyleCount', 'DomContentLoaded'].includes(m.name)) {
      console.log(`${m.name}: ${m.value}`);
    }
  }
});

test('Perf: App page load + demo auto-fit', async ({ page }) => {
  const start = Date.now();
  await page.goto('https://calyphi.com/app');
  const loadTime = Date.now() - start;
  console.log(`App initial load: ${loadTime}ms`);

  const fitStart = Date.now();
  await page.waitForSelector('text=RANKING', { timeout: 30000 });
  const fitTime = Date.now() - fitStart;
  console.log(`Demo auto-fit time: ${fitTime}ms`);
  console.log(`Total time to interactive: ${Date.now() - start}ms`);
});

test('Perf: Manual fit timing (Gaussian Peak)', async ({ page }) => {
  await page.goto('https://calyphi.com/app');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });

  await page.click('text=Gaussian Peak');
  const start = Date.now();
  await page.click('text=Auto-Fit All Models');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });
  const fitTime = Date.now() - start;
  console.log(`Gaussian Peak fit: ${fitTime}ms`);
});

test('Perf: Bundle size analysis', async ({ page }) => {
  const resources: { url: string; size: number; type: string }[] = [];
  page.on('response', async (response) => {
    const url = response.url();
    const headers = response.headers();
    const size = parseInt(headers['content-length'] || '0');
    const type = headers['content-type'] || '';
    if (url.includes('calyphi.com')) {
      resources.push({
        url: url.split('/').pop()?.slice(0, 60) || '',
        size,
        type: type.split(';')[0]
      });
    }
  });

  await page.goto('https://calyphi.com/app');
  await page.waitForLoadState('networkidle');

  resources.sort((a, b) => b.size - a.size);
  let totalJS = 0, totalCSS = 0, totalFont = 0, totalImg = 0, total = 0;
  for (const r of resources) {
    total += r.size;
    if (r.type.includes('javascript')) totalJS += r.size;
    if (r.type.includes('css')) totalCSS += r.size;
    if (r.type.includes('font')) totalFont += r.size;
    if (r.type.includes('image')) totalImg += r.size;
  }

  console.log(`Total: ${(total/1024).toFixed(0)}KB`);
  console.log(`JS: ${(totalJS/1024).toFixed(0)}KB`);
  console.log(`CSS: ${(totalCSS/1024).toFixed(0)}KB`);
  console.log(`Fonts: ${(totalFont/1024).toFixed(0)}KB`);
  console.log(`Images: ${(totalImg/1024).toFixed(0)}KB`);
  console.log('\nTop 10 largest:');
  for (const r of resources.slice(0, 10)) {
    console.log(`  ${(r.size/1024).toFixed(0)}KB  ${r.type}  ${r.url}`);
  }
});

test('Perf: Memory after 5 consecutive fits', async ({ page }) => {
  await page.goto('https://calyphi.com/app');
  await page.waitForSelector('text=RANKING', { timeout: 30000 });

  const samples = ['Enzyme Kinetics', 'Dose-Response', 'Bacterial Growth',
                    'Gaussian Peak', 'Radioactive Decay'];

  const heapBefore = await page.evaluate(() =>
    (performance as any).memory?.usedJSHeapSize || 0
  );

  for (const sample of samples) {
    await page.click(`text=${sample}`);
    await page.click('text=Auto-Fit All Models');
    await page.waitForSelector('text=RANKING', { timeout: 30000 });
  }

  const heapAfter = await page.evaluate(() =>
    (performance as any).memory?.usedJSHeapSize || 0
  );

  const growth = heapBefore > 0 ?
    ((heapAfter - heapBefore) / heapBefore * 100).toFixed(1) : 'N/A';
  console.log(`Heap before: ${(heapBefore/1024/1024).toFixed(1)}MB`);
  console.log(`Heap after: ${(heapAfter/1024/1024).toFixed(1)}MB`);
  console.log(`Growth: ${growth}%`);
});
