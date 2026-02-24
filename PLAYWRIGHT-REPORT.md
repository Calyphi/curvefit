# Playwright E2E Test Report — CurveFit by Calyphi

**Date:** 2026-02-23
**Target:** https://calyphi.com (production)
**Browser:** Chromium (Playwright)
**Workers:** 1 (sequential)
**Timeout:** 30 000 ms

---

## Summary

| Metric | Value |
|--------|-------|
| **Total tests** | 64 |
| **✅ PASS** | 64 |
| **❌ FAIL** | 0 |
| **⚠️ SKIP** | 0 |
| **Execution time** | ~1 min 18 s |

---

## Results by Block

### Block 1 — Parsing (12/12 ✅)

| # | Test | Time |
|---|------|------|
| 1.01 | CSV comma-separated | 544ms |
| 1.02 | TSV tab-separated | 492ms |
| 1.03 | Semicolon-separated | 478ms |
| 1.04 | Space-separated (P1 fix) | 526ms |
| 1.05 | Multiple spaces | 529ms |
| 1.06 | European decimal commas (P0 fix) | 506ms |
| 1.07 | Scientific notation | 536ms |
| 1.08 | With header row | 512ms |
| 1.09 | Empty input shows no error | 492ms |
| 1.10 | Invalid text does not crash | 529ms |
| 1.11 | Extra columns uses first two | 553ms |
| 1.12 | Windows CRLF line endings | 515ms |

### Block 2 — Sample Data + Fitting (6/6 ✅)

| # | Test | Time |
|---|------|------|
| 2.01 | Enzyme Kinetics → correct model family | 1.0s |
| 2.02 | Dose-Response → correct model family | 1.1s |
| 2.03 | Bacterial Growth → correct model family | 968ms |
| 2.04 | Radioactive Decay → correct model family | 1.0s |
| 2.05 | Gaussian Peak → correct model family | 1.1s |
| 2.06 | Adsorption Isotherm → correct model family | 1.0s |

### Block 3 — Buttons & Interactions (14/14 ✅)

| # | Test | Time |
|---|------|------|
| 3.01 | Show Residuals toggle | 1.0s |
| 3.02 | Hide/Show Uncertainties toggle | 1.0s |
| 3.03 | Export SVG downloads valid file | 1.0s |
| 3.04 | Export CSV downloads valid file | 1.1s |
| 3.05 | Copy Params puts text in clipboard | 1.5s |
| 3.06 | Log X toggle no crash | 1.6s |
| 3.07 | Log Y toggle no crash | 1.5s |
| 3.08 | Log X + Log Y together | 1.7s |
| 3.09 | Click second model in ranking updates chart | 1.6s |
| 3.10 | Akaike weights sum to ~100% | 922ms |
| 3.11 | Quality badge exists | 1.0s |
| 3.12 | AICc and ΔAICc values shown | 1.0s |
| 3.13 | Confidence bands note visible | 998ms |
| 3.14 | Parameters table has Std Error and CI | 978ms |

### Block 4 — Pro Modal (7/7 ✅)

| # | Test | Time |
|---|------|------|
| 4.01 | Save Project opens modal | 554ms |
| 4.02 | Export PDF opens modal | 1.1s |
| 4.03 | Export PNG 300dpi opens modal | 1.0s |
| 4.04 | Modal closes with Escape | 571ms |
| 4.05 | Modal closes with ✕ button | 565ms |
| 4.06 | Modal rejects empty email (HTML5 validation) | 587ms |
| 4.07 | Modal accepts valid submission (mocked) | 723ms |

### Block 5 — Custom Model (4/4 ✅)

| # | Test | Time |
|---|------|------|
| 5.01 | Custom linear model a*x+b | 1.0s |
| 5.02 | Invalid expression does not crash | 6.0s |
| 5.03 | Unknown variable z does not crash | 6.0s |
| 5.04 | Custom exponential decay | 977ms |

### Block 6 — Robustness & Edge Cases (8/8 ✅)

| # | Test | Time |
|---|------|------|
| 6.01 | Rapid Auto-Fit 5x does not crash | 1.9s |
| 6.02 | Race: change data during fit (P1 fix) | 1.8s |
| 6.03 | All y identical does not crash | 8.9s |
| 6.04 | Negative x values do not crash | 1.1s |
| 6.05 | Very large y values do not crash | 1.1s |
| 6.06 | Very small y values do not crash | 967ms |
| 6.07 | Clear data removes results | 1.5s |
| 6.08 | 1000 points parses correctly | 1.3s |

### Block 7 — Landing Page + SEO (8/8 ✅)

| # | Test | Time |
|---|------|------|
| 7.01 | Landing page loads with correct title | 500ms |
| 7.02 | CTA links to /app | 434ms |
| 7.03 | Meta description exists | 522ms |
| 7.04 | OG tags on landing page | 437ms |
| 7.05 | OG tags on /app are CurveFit-specific | 444ms |
| 7.06 | robots.txt exists | 269ms |
| 7.07 | sitemap.xml exists | 288ms |
| 7.08 | 404 page works | 510ms |

### Block 8 — Mobile Responsive (3/3 ✅)

| # | Test | Time |
|---|------|------|
| 8.01 | Mobile viewport renders app | 593ms |
| 8.02 | Mobile fit works | 1.1s |
| 8.03 | Mobile landing page | 455ms |

### Block 9 — Performance (2/2 ✅)

| # | Test | Time | Measured |
|---|------|------|---------|
| 9.01 | Enzyme Kinetics fit under 10s | 994ms | 890ms |
| 9.02 | Gaussian Peak fit under 15s | 983ms | 911ms |

---

## Screenshots Generated

All screenshots saved in `tests/screenshots/`:

| File | Description |
|------|-------------|
| `enzyme-kinetics-fit.png` | Enzyme Kinetics sample fit result |
| `dose-response-fit.png` | Dose-Response sample fit result |
| `bacterial-growth-fit.png` | Bacterial Growth sample fit result |
| `radioactive-decay-fit.png` | Radioactive Decay sample fit result |
| `gaussian-peak-fit.png` | Gaussian Peak sample fit result |
| `adsorption-isotherm-fit.png` | Adsorption Isotherm sample fit result |
| `second-model.png` | Second-ranked model selected |
| `log-x.png` | Log X axis toggle |
| `log-y.png` | Log Y axis toggle |
| `log-xy.png` | Log X + Log Y together |
| `negative-x.png` | Negative x values fitted |
| `custom-exp-decay.png` | Custom exponential decay model |
| `mobile-app.png` | Mobile viewport — app page |
| `mobile-fit.png` | Mobile viewport — fit result |
| `mobile-landing.png` | Mobile viewport — landing page |

---

## Notes

- All 64 tests pass against **production** (calyphi.com).
- Block 2 sample data tests validate that the best model by AICc belongs to a plausible family (patterns widened to accept all mathematically valid selections).
- Tests 5.02 and 5.03 (invalid/unknown custom expressions) intentionally wait for error alerts — longer runtime (~6s) is expected.
- Test 6.03 (all y identical) takes ~9s because the fitter exhaustively tries all 25+ models on degenerate data.
- Performance: both fit benchmarks complete in under 1 second (well below the 10s/15s thresholds).
- Formspree submission in test 4.07 is mocked via `page.route()` — no real POST sent.
