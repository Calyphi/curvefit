import { chromium } from "@playwright/test";

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto("https://calyphi.com/app", { waitUntil: "networkidle" });
  
  // Click Enzyme Kinetics sample
  await page.locator("button", { hasText: "Enzyme Kinetics" }).click();
  await page.waitForTimeout(400);
  
  // Click Auto-Fit
  await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
  
  // Wait for ranking to appear
  await page.waitForSelector('[role="listbox"]', { timeout: 20000 });
  await page.waitForTimeout(500);
  
  // Take screenshot of the full page
  await page.screenshot({ 
    path: "public/hero-screenshot.png",
    fullPage: false,  // viewport only — the app fills the viewport
  });
  
  console.log("Screenshot saved to public/hero-screenshot.png");
  await browser.close();
})();
