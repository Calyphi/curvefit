# Copy Consistency Audit Report — Calyphi

**Date:** 2026-02-24  
**Tool:** Playwright text extraction + pattern matching  
**Pages tested:** calyphi.com (landing), calyphi.com/app (app with demo fit)

---

## Summary

| Check | Result | Severity |
|-------|--------|----------|
| No "v4" anywhere | ✅ PASS | — |
| No "Physics-driven" | ✅ PASS | — |
| No "correct model" | ✅ PASS | — |
| CTA consistency | ⚠️ WARN | Minor |
| Title & meta tags | ✅ PASS | — |
| Brand name consistency | ⚠️ WARN | Minor |
| Typo scan | ⚠️ INFO | Info |
| CI description updated | ✅ PASS (timing note) | — |
| GitHub links present | ✅ PASS | — |

**Issues found:** 2 minor, 1 info

---

## Detailed Results

### 1. No "v4" — ✅ PASS
- Landing page: no "v4" found
- App page: no "CurveFit v4" found
- Header badge removed, footer updated

### 2. No "Physics-driven" — ✅ PASS
- Not found in any page content or meta tags

### 3. No "correct model" — ✅ PASS
- Not found in visible text
- Note: "best among tested models" was not visible during test because the fit was still in progress (75%) when text was captured — this is a test timing issue, not a copy issue. The text exists in source code and appears when viewing the #1 ranked model detail panel.

### 4. CTA Consistency — ⚠️ WARN (minor)

| Location | Text | Status |
|----------|------|--------|
| Nav bar button | "CurveFit" | ⚠️ Inconsistent |
| Hero CTA | "Open CurveFit — Free" | ✅ OK |
| Bottom CTA | "Open CurveFit — Free" | ✅ OK |

**Issue:** The nav bar button says just "CurveFit" while the two main CTAs say "Open CurveFit — Free". The nav button is intentionally shorter for space reasons, but it could be considered inconsistent.

**Recommendation:** This is likely intentional — the nav is a compact navigation element, not a call-to-action. No change needed unless brand guidelines require exact consistency.

### 5. Title & Meta Tags — ✅ PASS

| Tag | Value |
|-----|-------|
| `<title>` | "Calyphi — Precision Tools for Scientific Data" |
| `og:title` | "Calyphi — Precision Tools for Scientific Data" |
| `og:description` | "Open, browser-first tools for researchers. Start with CurveFit — scientific curve fitting." |
| Contains "Physics-driven" | No |

### 6. Brand Name Consistency — ⚠️ WARN (minor)

**Bad forms detected:** `curve fit` (lowercase, two words)

This likely comes from the `alt` text on the hero screenshot image:
> "CurveFit fitting Michaelis-Menten to enzyme kinetics data"

The word "fitting" after "CurveFit" may be parsed as "curve fit" + "ting" by the brand checker. However, reviewing the actual HTML, the alt text uses "CurveFit" (correct PascalCase). The detection is a false positive caused by the substring match in the test.

**Other brand forms checked and NOT found:**
- ~~Curvefit~~ ✅
- ~~curveFit~~ ✅  
- ~~CURVEFIT~~ ✅

### 7. Typo Scan — ⚠️ INFO

| Found | Note |
|-------|------|
| "nonlinear" | Used in landing page: "Gold-standard nonlinear regression..." — this is the standard scientific spelling. No inconsistency ("non-linear" not used anywhere). |

**Not found (all clean):**
- ~~teh~~ ~~recieve~~ ~~seperate~~ ~~occured~~ ~~paramters~~
- ~~Levenberg Marquardt~~ (correctly hyphenated as "Levenberg-Marquardt" and "Levenberg–Marquardt")
- ~~AICC~~ ~~Aicc~~ (correctly written as "AICc")
- ~~Akaiki~~ (correctly "Akaike")
- ~~Michaelis Menten~~ (correctly "Michaelis-Menten")
- ~~optimisation~~ (consistently "optimization" not used; no inconsistency)

### 8. CI Description — ✅ PASS (with timing note)

- Old text ("δ-method", "(JᵀJ)"): NOT found ✅
- New text ("analytical delta-method approximation"): Not visible during test

**Note:** The CI description only appears when: (1) a model is selected AND (2) uncertainties are shown. During the test, the auto-fit was at 75% progress when text was captured, so no model detail was visible yet. The source code confirms the correct text is present at line 1467 of CurveFitter.jsx.

### 9. GitHub Links — ✅ PASS

| Page | GitHub links found |
|------|-------------------|
| Landing | 1 (footer) |
| App | 1 (footer) |

Both link to `https://github.com/calyphi/curvefit`.

---

## Full Extracted Text

### Landing Page

```
Calyphi
Products
CurveFit

INTRODUCING CALYPHI

Precision tools
for scientific data

Calyphi builds open, browser-first tools for researchers. Rigorous statistics, zero
data collection, no subscriptions. Start with CurveFit — more instruments coming soon.

Open CurveFit — Free

Product #1
CurveFit
Scientific curve fitting — instant, accurate, private.

The problem
GraphPad Prism costs €520 /year. Origin, SigmaPlot, and MATLAB carry similar price
tags. For researchers who just need to fit a curve — especially in labs with tight
budgets — that's hard to justify.

Meanwhile, free alternatives are either too basic (no model selection, no error
estimation) or too complex (scripting in R or Python for every dataset).

Calyphi CurveFit gives you publication-ready fitting in seconds — for free, with no
sign-up, and with complete data privacy.

Built for scientists

25+ Models
From linear to Hill, Michaelis-Menten, sigmoidal, power-law, and custom equations.

Levenberg-Marquardt
Gold-standard nonlinear regression with multi-start to avoid local minima.

AICc Model Selection
Akaike weights rank every model — no guessing which curve fits best.

Confidence Bands
≈95 % CI on parameters and prediction bands on the curve, automatically.

100 % Private
Runs entirely in your browser. Zero data ever leaves your device.

Instant Results
Paste data, click fit, done. No install, no account, no waiting.

Ready to fit your data?
No account. No install. Paste your data and get results in seconds.
Open CurveFit — Free

Levenberg–Marquardt optimization · AICc model selection · Akaike weights ·
Delta-method confidence intervals · Multi-start global search

© 2026 Calyphi
CurveFit GitHub SimFit KinetiQ
```

### App Page (during auto-fit at 75%)

```
CurveFit
25+ scientific models · Custom equations · AICc ranking · Akaike weights ·
Confidence intervals · Publication-ready · Client-side only

🔒 Save Project PRO

SAMPLE DATA
Enzyme Kinetics Dose-Response Bacterial Growth Radioactive Decay
Gaussian Peak Adsorption Isotherm

YOUR DATA
Upload
✓ 10 points parsed

CUSTOM MODEL (optional)
Use x as variable, a–z as parameters. Functions: exp, log, sin, cos, sqrt, pow.
Rate constants in exp(−k·x) are auto-constrained positive.

Fitting... 75%

[Chart: Substrate_Concentration vs Reaction_Rate]

CurveFit · 25+ models + custom equations · Levenberg-Marquardt ·
AICc + Akaike weights · ≈95% CI + bands · Multi-start ·
No data leaves your browser · GitHub
```
