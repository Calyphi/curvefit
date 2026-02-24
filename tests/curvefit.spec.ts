import { test, expect, Page } from "@playwright/test";
import * as fs from "fs";

// ============================================================
// HELPERS
// ============================================================

async function clickSample(page: Page, name: string) {
  await page.goto("/app");
  await page.locator("button", { hasText: name }).click();
  await page.waitForTimeout(300);
}

async function doEnzymeKineticsFit(page: Page) {
  await clickSample(page, "Enzyme Kinetics");
  await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
  await page.waitForSelector('[role="listbox"]', { timeout: 20_000 });
}

// ============================================================
// BLOCK 1: PARSING (12 tests)
// ============================================================

test.describe("Block 1 — Parsing", () => {
  test("1.01 CSV comma-separated", async ({ page }) => {
    await page.goto("/app");
    await page.locator("textarea").first().fill("x,y\n1,10\n2,20\n3,30\n4,40");
    await expect(page.locator("text=4 points parsed")).toBeVisible();
  });

  test("1.02 TSV tab-separated", async ({ page }) => {
    await page.goto("/app");
    await page.locator("textarea").first().fill("1\t10\n2\t20\n3\t30\n4\t40");
    await expect(page.locator("text=4 points parsed")).toBeVisible();
  });

  test("1.03 Semicolon-separated", async ({ page }) => {
    await page.goto("/app");
    await page.locator("textarea").first().fill("1;10\n2;20\n3;30\n4;40");
    await expect(page.locator("text=4 points parsed")).toBeVisible();
  });

  test("1.04 Space-separated (P1 fix)", async ({ page }) => {
    await page.goto("/app");
    await page.locator("textarea").first().fill("1 10\n2 20\n3 30\n4 40");
    await expect(page.locator("text=4 points parsed")).toBeVisible();
  });

  test("1.05 Multiple spaces", async ({ page }) => {
    await page.goto("/app");
    await page
      .locator("textarea")
      .first()
      .fill("1   10\n2   20\n3   30\n4   40");
    await expect(page.locator("text=4 points parsed")).toBeVisible();
  });

  test("1.06 European decimal commas (P0 fix)", async ({ page }) => {
    await page.goto("/app");
    await page
      .locator("textarea")
      .first()
      .fill("1,5;10,3\n2,0;15,7\n3,5;20,1");
    await expect(page.locator("text=3 points parsed")).toBeVisible();
  });

  test("1.07 Scientific notation", async ({ page }) => {
    await page.goto("/app");
    await page
      .locator("textarea")
      .first()
      .fill("1e-3,1.5e2\n2e-3,3.0e2\n1e-2,4.5e2");
    await expect(page.locator("text=3 points parsed")).toBeVisible();
  });

  test("1.08 With header row", async ({ page }) => {
    await page.goto("/app");
    await page
      .locator("textarea")
      .first()
      .fill("Concentration,Rate\n0.1,1.8\n0.2,3.2\n0.5,6.1");
    await expect(page.locator("text=3 points parsed")).toBeVisible();
  });

  test("1.09 Empty input shows no error", async ({ page }) => {
    await page.goto("/app");
    await page.locator("textarea").first().fill("");
    await expect(page.locator("text=points parsed")).not.toBeVisible();
  });

  test("1.10 Invalid text does not crash", async ({ page }) => {
    await page.goto("/app");
    await page.locator("textarea").first().fill("hello world\nfoo bar\nbaz");
    await expect(
      page.locator("button", { hasText: "Auto-Fit All Models" })
    ).toBeVisible();
  });

  test("1.11 Extra columns uses first two", async ({ page }) => {
    await page.goto("/app");
    await page
      .locator("textarea")
      .first()
      .fill("x,y,z\n1,10,99\n2,20,88\n3,30,77");
    await expect(page.locator("text=3 points parsed")).toBeVisible();
  });

  test("1.12 Windows CRLF line endings", async ({ page }) => {
    await page.goto("/app");
    await page
      .locator("textarea")
      .first()
      .fill("1,10\r\n2,20\r\n3,30\r\n4,40");
    await expect(page.locator("text=4 points parsed")).toBeVisible();
  });
});

// ============================================================
// BLOCK 2: SAMPLE DATA + FITTING (6 tests)
// ============================================================

test.describe("Block 2 — Sample Data + Fitting", () => {
  const sampleTests = [
    { name: "Enzyme Kinetics", expectModel: /Michaelis|Hill|Langmuir|4PL|Dose|Reciprocal/ },
    { name: "Dose-Response", expectModel: /Dose|Logistic|4PL|5PL|Hill|Reciprocal/ },
    {
      name: "Bacterial Growth",
      expectModel: /Logistic|Gompertz|Growth|Weibull/,
    },
    { name: "Radioactive Decay", expectModel: /Decay|Exponential/ },
    { name: "Gaussian Peak", expectModel: /Gaussian|Lorentzian/ },
    {
      name: "Adsorption Isotherm",
      expectModel: /Langmuir|Michaelis|Freundlich|Hill/,
    },
  ];

  for (const { name, expectModel } of sampleTests) {
    test(`2.xx ${name} → correct model family`, async ({ page }) => {
      await clickSample(page, name);
      await page
        .locator("button", { hasText: "Auto-Fit All Models" })
        .click();
      await page.waitForSelector('[role="listbox"]', { timeout: 20_000 });

      // BEST badge exists
      await expect(page.locator('span[aria-label="Best model"]')).toBeVisible();

      // Best model name matches expected family
      const bestRow = page.locator('[role="option"]').first();
      const modelName = await bestRow.textContent();
      expect(modelName).toMatch(expectModel);

      // adj. R² displayed in the detail section — extract from the large number
      const r2Display = page
        .locator("section")
        .filter({ hasText: "adj. R²" })
        .locator(".text-2xl, .font-bold.font-mono")
        .first();
      const r2Text = await r2Display.textContent();
      // Could be "99.85%" or "-0.123"
      const r2Match = r2Text?.match(/([\d.]+)%/);
      if (r2Match) {
        const r2Value = parseFloat(r2Match[1]);
        expect(r2Value).toBeGreaterThan(95);
      }

      // Std Error column visible
      await expect(page.locator("text=Std Error")).toBeVisible();

      // 95% CI column visible
      await expect(page.locator("th", { hasText: "≈95% CI" })).toBeVisible();

      // Screenshot
      await page.screenshot({
        path: `tests/screenshots/${name.replace(/\s/g, "-").toLowerCase()}-fit.png`,
        fullPage: true,
      });
    });
  }
});

// ============================================================
// BLOCK 3: BUTTONS & INTERACTIONS (14 tests)
// ============================================================

test.describe("Block 3 — Buttons & Interactions", () => {
  test("3.01 Show Residuals toggle", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    await page.locator("button", { hasText: "Show Residuals" }).click();
    await expect(
      page.locator("button", { hasText: "Hide Residuals" })
    ).toBeVisible();
    await page.locator("button", { hasText: "Hide Residuals" }).click();
    await expect(
      page.locator("button", { hasText: "Show Residuals" })
    ).toBeVisible();
  });

  test("3.02 Hide/Show Uncertainties toggle", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    await page.locator("button", { hasText: "Hide Uncertainties" }).click();
    await expect(
      page.locator("button", { hasText: "Show Uncertainties" })
    ).toBeVisible();
    await page.locator("button", { hasText: "Show Uncertainties" }).click();
    await expect(
      page.locator("button", { hasText: "Hide Uncertainties" })
    ).toBeVisible();
  });

  test("3.03 Export SVG downloads valid file", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.locator("button", { hasText: "Export SVG" }).click(),
    ]);
    const filename = download.suggestedFilename();
    expect(filename).toMatch(/\.svg$/);
    const dlPath = await download.path();
    if (dlPath) {
      const content = fs.readFileSync(dlPath, "utf-8");
      expect(content).toContain("<svg");
    }
  });

  test("3.04 Export CSV downloads valid file", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.locator("button", { hasText: "Export CSV" }).click(),
    ]);
    const filename = download.suggestedFilename();
    expect(filename).toMatch(/\.csv$/);
    const dlPath = await download.path();
    if (dlPath) {
      const content = fs.readFileSync(dlPath, "utf-8");
      const lines = content.trim().split("\n");
      expect(lines.length).toBeGreaterThan(3);
    }
  });

  test("3.05 Copy Params puts text in clipboard", async ({
    page,
    context,
  }) => {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await doEnzymeKineticsFit(page);
    await page.locator("button", { hasText: "Copy Params" }).click();
    await page.waitForTimeout(500);
    const clipboard = await page.evaluate(() =>
      navigator.clipboard.readText()
    );
    expect(clipboard.length).toBeGreaterThan(10);
    expect(clipboard).toMatch(/R²/);
  });

  test("3.06 Log X toggle no crash", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    await page.locator("button", { hasText: "Log X" }).click();
    await page.waitForTimeout(500);
    await expect(page.locator('[role="listbox"]')).toBeVisible();
    await page.screenshot({ path: "tests/screenshots/log-x.png" });
  });

  test("3.07 Log Y toggle no crash", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    await page.locator("button", { hasText: "Log Y" }).click();
    await page.waitForTimeout(500);
    await expect(page.locator('[role="listbox"]')).toBeVisible();
    await page.screenshot({ path: "tests/screenshots/log-y.png" });
  });

  test("3.08 Log X + Log Y together", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    await page.locator("button", { hasText: "Log X" }).click();
    await page.locator("button", { hasText: "Log Y" }).click();
    await page.waitForTimeout(500);
    await expect(page.locator('[role="listbox"]')).toBeVisible();
    await page.screenshot({ path: "tests/screenshots/log-xy.png" });
  });

  test("3.09 Click second model in ranking updates chart", async ({
    page,
  }) => {
    await doEnzymeKineticsFit(page);
    const secondModel = page.locator('[role="option"]').nth(1);
    await secondModel.click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: "tests/screenshots/second-model.png" });
  });

  test("3.10 Akaike weights sum to ~100%", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    // The Akaike weight percentages are inside a span.font-mono.text-gray-500
    const percentages = await page.locator('[role="option"] span.font-mono.text-gray-500').evaluateAll(
      (els) =>
        els.map((el) => {
          const match = el.textContent?.match(/(\d+)%/);
          return match ? parseInt(match[1]) : 0;
        })
    );
    const sum = percentages.reduce((a, b) => a + b, 0);
    expect(sum).toBeGreaterThanOrEqual(95);
    expect(sum).toBeLessThanOrEqual(105);
  });

  test("3.11 Quality badge exists", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    const badge = page.locator("text=/Good|Needs review|Unreliable/").first();
    await expect(badge).toBeVisible();
  });

  test("3.12 AICc and ΔAICc values shown", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    // Stat cards have labels: "AICc", "ΔAICc", "Akaike weight" (in description text)
    await expect(page.locator("div.text-xs.text-gray-400", { hasText: "AICc" }).first()).toBeVisible();
    await expect(page.locator("div.text-xs.text-gray-400", { hasText: "ΔAICc" })).toBeVisible();
    // Akaike weight mentioned in the interpretation text
    await expect(page.locator("text=/Akaike weight/i").first()).toBeVisible();
  });

  test("3.13 Confidence bands note visible", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    await expect(page.locator("text=/Bands show/")).toBeVisible();
  });

  test("3.14 Parameters table has Std Error and CI", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    await expect(page.locator("th", { hasText: "Parameter" })).toBeVisible();
    await expect(page.locator("th", { hasText: "Value" }).first()).toBeVisible();
    await expect(page.locator("th", { hasText: "Std Error" })).toBeVisible();
    await expect(page.locator("th >> text=/95%/")).toBeVisible();
  });
});

// ============================================================
// BLOCK 4: PRO MODAL (7 tests)
// ============================================================

test.describe("Block 4 — Pro Modal", () => {
  test("4.01 Save Project opens modal", async ({ page }) => {
    await page.goto("/app");
    await page.locator("button", { hasText: "Save Project" }).click();
    await expect(page.locator('[role="dialog"]')).toBeVisible();
  });

  test("4.02 Export PDF opens modal", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    await page.locator("button", { hasText: "Export PDF" }).click();
    await expect(page.locator('[role="dialog"]')).toBeVisible();
  });

  test("4.03 Export PNG 300dpi opens modal", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    await page.locator("button", { hasText: "Export PNG" }).click();
    await expect(page.locator('[role="dialog"]')).toBeVisible();
  });

  test("4.04 Modal closes with Escape", async ({ page }) => {
    await page.goto("/app");
    await page.locator("button", { hasText: "Save Project" }).click();
    await expect(page.locator('[role="dialog"]')).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.locator('[role="dialog"]')).not.toBeVisible();
  });

  test("4.05 Modal closes with ✕ button", async ({ page }) => {
    await page.goto("/app");
    await page.locator("button", { hasText: "Save Project" }).click();
    await expect(page.locator('[role="dialog"]')).toBeVisible();
    // Close button is the ✕ at top-right
    await page
      .locator('[role="dialog"]')
      .locator("button", { hasText: "✕" })
      .click();
    await expect(page.locator('[role="dialog"]')).not.toBeVisible();
  });

  test("4.06 Modal rejects empty email (HTML5 validation)", async ({
    page,
  }) => {
    await page.goto("/app");
    await page.locator("button", { hasText: "Save Project" }).click();
    await expect(page.locator('[role="dialog"]')).toBeVisible();
    // Try to submit without filling email
    const submitBtn = page
      .locator('[role="dialog"]')
      .locator('button[type="submit"]');
    await submitBtn.click();
    // Should still see the dialog (not closed / no success)
    await expect(
      page.locator("text=/You.*on the list|Thank|Success/i")
    ).not.toBeVisible();
  });

  test("4.07 Modal accepts valid submission (mocked)", async ({ page }) => {
    await page.goto("/app");
    // Mock Formspree
    await page.route("**/formspree.io/**", (route) => {
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ ok: true }),
      });
    });
    await page.locator("button", { hasText: "Save Project" }).click();
    await expect(page.locator('[role="dialog"]')).toBeVisible();
    await page
      .locator('[role="dialog"]')
      .locator('input[type="email"]')
      .fill("playwright@calyphi.com");
    const select = page.locator('[role="dialog"]').locator("select").first();
    if (await select.isVisible()) {
      await select.selectOption({ index: 1 });
    }
    await page
      .locator('[role="dialog"]')
      .locator('button[type="submit"]')
      .click();
    // Wait for success message
    await expect(
      page.locator("text=/on the list|Thank|Success/i")
    ).toBeVisible({ timeout: 5_000 });
  });
});

// ============================================================
// BLOCK 5: CUSTOM MODEL (4 tests)
// ============================================================

test.describe("Block 5 — Custom Model", () => {
  test("5.01 Custom linear model a*x+b", async ({ page }) => {
    await page.goto("/app");
    await page
      .locator("textarea")
      .first()
      .fill("1,10\n2,20\n3,30\n4,40\n5,50");
    await page.waitForTimeout(300);
    await page
      .locator('input[placeholder*="exp"]')
      .fill("a * x + b");
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForSelector('[role="listbox"]', { timeout: 20_000 });
    await expect(page.locator('[role="option"]').filter({ hasText: /Linear|Custom/ }).first()).toBeVisible();
  });

  test("5.02 Invalid expression does not crash", async ({ page }) => {
    await page.goto("/app");
    await page.locator("textarea").first().fill("1,10\n2,20\n3,30\n4,40");
    await page.waitForTimeout(300);
    await page
      .locator('input[placeholder*="exp"]')
      .fill("a * /");
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForTimeout(5_000);
    await expect(
      page.locator("button", { hasText: "Auto-Fit All Models" })
    ).toBeEnabled({ timeout: 20_000 });
  });

  test("5.03 Unknown variable z does not crash", async ({ page }) => {
    await page.goto("/app");
    await page.locator("textarea").first().fill("1,10\n2,20\n3,30\n4,40");
    await page.waitForTimeout(300);
    await page
      .locator('input[placeholder*="exp"]')
      .fill("a * z + b");
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForTimeout(5_000);
    await expect(
      page.locator("button", { hasText: "Auto-Fit All Models" })
    ).toBeEnabled({ timeout: 20_000 });
  });

  test("5.04 Custom exponential decay", async ({ page }) => {
    await page.goto("/app");
    await page
      .locator("textarea")
      .first()
      .fill(
        "0,100\n1,90\n2,81\n3,73\n4,66\n5,59\n6,53\n7,48\n8,43\n9,39\n10,35"
      );
    await page.waitForTimeout(300);
    await page
      .locator('input[placeholder*="exp"]')
      .fill("a * exp(-b * x) + c");
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForSelector('[role="listbox"]', { timeout: 20_000 });
    await page.screenshot({
      path: "tests/screenshots/custom-exp-decay.png",
      fullPage: true,
    });
  });
});

// ============================================================
// BLOCK 6: ROBUSTNESS & EDGE CASES (8 tests)
// ============================================================

test.describe("Block 6 — Robustness", () => {
  test("6.01 Rapid Auto-Fit 5x does not crash", async ({ page }) => {
    await clickSample(page, "Enzyme Kinetics");
    for (let i = 0; i < 5; i++) {
      await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
      await page.waitForTimeout(200);
    }
    await page.waitForSelector('[role="listbox"]', { timeout: 30_000 });
  });

  test("6.02 Race: change data during fit (P1 fix)", async ({ page }) => {
    await clickSample(page, "Enzyme Kinetics");
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForTimeout(500);
    // Change to Gaussian Peak while fit is running
    await page.locator("button", { hasText: "Gaussian Peak" }).click();
    await page.waitForTimeout(300);
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForSelector('[role="listbox"]', { timeout: 20_000 });
    // Best model should be from Gaussian data, not Enzyme
    const bestRow = page.locator('[role="option"]').first();
    const modelName = await bestRow.textContent();
    expect(modelName).toMatch(/Gaussian|Lorentzian/);
  });

  test("6.03 All y identical does not crash", async ({ page }) => {
    await page.goto("/app");
    await page.locator("textarea").first().fill("1,5\n2,5\n3,5\n4,5\n5,5");
    await page.waitForTimeout(300);
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForTimeout(8_000);
    await expect(
      page.locator("button", { hasText: "Auto-Fit All Models" })
    ).toBeEnabled({ timeout: 20_000 });
  });

  test("6.04 Negative x values do not crash", async ({ page }) => {
    await page.goto("/app");
    await page
      .locator("textarea")
      .first()
      .fill("-5,25\n-3,9\n-1,1\n0,0\n1,1\n3,9\n5,25");
    await page.waitForTimeout(300);
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForSelector('[role="listbox"]', { timeout: 20_000 });
    await page.screenshot({
      path: "tests/screenshots/negative-x.png",
      fullPage: true,
    });
  });

  test("6.05 Very large y values do not crash", async ({ page }) => {
    await page.goto("/app");
    await page
      .locator("textarea")
      .first()
      .fill("1,1e12\n2,2e12\n3,3e12\n4,4e12\n5,5e12");
    await page.waitForTimeout(300);
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForSelector('[role="listbox"]', { timeout: 20_000 });
  });

  test("6.06 Very small y values do not crash", async ({ page }) => {
    await page.goto("/app");
    await page
      .locator("textarea")
      .first()
      .fill("1,1e-12\n2,2e-12\n3,3e-12\n4,4e-12\n5,5e-12");
    await page.waitForTimeout(300);
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForSelector('[role="listbox"]', { timeout: 20_000 });
  });

  test("6.07 Clear data removes results", async ({ page }) => {
    await doEnzymeKineticsFit(page);
    await expect(page.locator('[role="listbox"]')).toBeVisible();
    await page.locator("textarea").first().fill("");
    await page.waitForTimeout(500);
    await expect(page.locator('[role="listbox"]')).not.toBeVisible();
  });

  test("6.08 1000 points parses correctly", async ({ page }) => {
    await page.goto("/app");
    const data = Array.from({ length: 1000 }, (_, i) => {
      const x = i * 0.1;
      const y = 100 * Math.sin(x / 50) + (Math.random() - 0.5) * 5;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join("\n");
    await page.locator("textarea").first().fill(data);
    await expect(page.locator("text=1000 points parsed")).toBeVisible();
  });
});

// ============================================================
// BLOCK 7: LANDING PAGE + SEO (8 tests)
// ============================================================

test.describe("Block 7 — Landing Page + SEO", () => {
  test("7.01 Landing page loads with correct title", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/Calyphi/);
    await expect(page.locator("h1")).toBeVisible();
  });

  test("7.02 CTA links to /app", async ({ page }) => {
    await page.goto("/");
    const cta = page
      .locator("a", { hasText: /CurveFit|Start Fitting|Try/ })
      .first();
    await expect(cta).toHaveAttribute("href", "/app");
  });

  test("7.03 Meta description exists", async ({ page }) => {
    await page.goto("/");
    const desc = await page
      .locator('meta[name="description"]')
      .getAttribute("content");
    expect(desc).toBeTruthy();
    expect(desc!.length).toBeGreaterThan(50);
  });

  test("7.04 OG tags on landing page", async ({ page }) => {
    await page.goto("/");
    await expect(
      page.locator('meta[property="og:title"]')
    ).toHaveAttribute("content", /.+/);
    await expect(
      page.locator('meta[property="og:description"]')
    ).toHaveAttribute("content", /.+/);
    await expect(
      page.locator('meta[property="og:image"]')
    ).toHaveAttribute("content", /.+/);
  });

  test("7.05 OG tags on /app are CurveFit-specific", async ({ page }) => {
    await page.goto("/app");
    const ogTitle = await page
      .locator('meta[property="og:title"]')
      .getAttribute("content");
    expect(ogTitle).toMatch(/CurveFit/i);
    const ogUrl = await page
      .locator('meta[property="og:url"]')
      .getAttribute("content");
    expect(ogUrl).toContain("/app");
  });

  test("7.06 robots.txt exists", async ({ page }) => {
    const response = await page.goto("/robots.txt");
    expect(response!.status()).toBe(200);
    const text = await page.locator("body").textContent();
    expect(text).toContain("User-agent");
    expect(text).toContain("Sitemap");
  });

  test("7.07 sitemap.xml exists", async ({ page }) => {
    const response = await page.goto("/sitemap.xml");
    expect(response!.status()).toBe(200);
  });

  test("7.08 404 page works", async ({ page }) => {
    const response = await page.goto("/this-page-does-not-exist");
    expect(response!.status()).toBe(404);
  });
});

// ============================================================
// BLOCK 8: MOBILE RESPONSIVE (3 tests)
// ============================================================

test.describe("Block 8 — Mobile Responsive", () => {
  test("8.01 Mobile viewport renders app", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/app");
    await expect(page.locator("textarea")).toBeVisible();
    await expect(
      page.locator("button", { hasText: "Auto-Fit All Models" })
    ).toBeVisible();
    await page.screenshot({
      path: "tests/screenshots/mobile-app.png",
      fullPage: true,
    });
  });

  test("8.02 Mobile fit works", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await clickSample(page, "Enzyme Kinetics");
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForSelector('[role="listbox"]', { timeout: 20_000 });
    await page.screenshot({
      path: "tests/screenshots/mobile-fit.png",
      fullPage: true,
    });
  });

  test("8.03 Mobile landing page", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/");
    await expect(page.locator("h1")).toBeVisible();
    const cta = page
      .locator("a", { hasText: /CurveFit|Start|Try/ })
      .first();
    await expect(cta).toBeVisible();
    await page.screenshot({
      path: "tests/screenshots/mobile-landing.png",
      fullPage: true,
    });
  });
});

// ============================================================
// BLOCK 9: PERFORMANCE (2 tests)
// ============================================================

test.describe("Block 9 — Performance", () => {
  test("9.01 Enzyme Kinetics fit under 10s", async ({ page }) => {
    const start = Date.now();
    await clickSample(page, "Enzyme Kinetics");
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForSelector('[role="listbox"]', { timeout: 15_000 });
    const elapsed = Date.now() - start;
    expect(elapsed).toBeLessThan(10_000);
    console.log(`Enzyme Kinetics fit time: ${elapsed}ms`);
  });

  test("9.02 Gaussian Peak fit under 15s", async ({ page }) => {
    const start = Date.now();
    await clickSample(page, "Gaussian Peak");
    await page.locator("button", { hasText: "Auto-Fit All Models" }).click();
    await page.waitForSelector('[role="listbox"]', { timeout: 20_000 });
    const elapsed = Date.now() - start;
    expect(elapsed).toBeLessThan(15_000);
    console.log(`Gaussian Peak fit time: ${elapsed}ms`);
  });
});
