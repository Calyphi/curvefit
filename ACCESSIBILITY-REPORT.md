# Accessibility Audit Report — Calyphi

**Date:** 2026-02-24  
**Tool:** axe-core via @axe-core/playwright  
**Standards:** WCAG 2.0 Level A + AA  
**Pages tested:** calyphi.com (landing), calyphi.com/app (app with demo fit), Pro modal

---

## Summary

| Page | Violations | Critical | Serious | Moderate | Minor |
|------|-----------|----------|---------|----------|-------|
| Landing | 1 | 0 | 1 | 0 | 0 |
| App (demo fit) | 1 | 0 | 1 | 0 | 0 |
| Pro Modal | 1 | 0 | 1 | 0 | 0 |

**Overall score: WARN** — 0 critical, 3 serious (all color-contrast)

---

## Violations

### 1. `color-contrast` (serious) — All pages

**Rule:** Ensure the contrast between foreground and background colors meets WCAG 2 AA minimum contrast ratio thresholds (4.5:1 for normal text, 3:1 for large text).

#### Landing page — 9 affected elements

| Element | Issue |
|---------|-------|
| `.rounded-lg` (nav CTA button) | Cyan text on cyan-600 bg — contrast too low |
| `.mt-10 > .inline-flex` (hero CTA) | White text on cyan-600 bg — borderline |
| `.mt-8` (bottom CTA) | White text on cyan-600 bg — borderline |
| + 6 more elements | Various gray-on-dark combinations |

**Fix suggestion:** Increase contrast on cyan-600 buttons (try bg-cyan-700 or brighter text). Review text-gray-500 on bg-gray-950 (ratio ~3.5:1, needs 4.5:1).

#### App page — 22 affected elements

| Element | Issue |
|---------|-------|
| `.normal-case` (custom model input label area) | Gray text on dark bg |
| `.text-gray-600.mt-1` | Very low contrast gray-600 on gray-900 |
| `.py-3` (textarea area) | Gray placeholder on dark bg |
| `.bg-blue-500` (BEST badge text) | White on blue-500 may be borderline |
| `.text-gray-500.font-mono` × 9 | Akaike weight percentages in ranking list |
| `.text-gray-500` (parameter table CI column) × 4 | CI range values |
| `.mt-1.5` (CI footnote) | Italic gray-500 text |
| `footer` | Footer text gray-500 on gray-950 |
| `a` (GitHub link) | gray-500 link on gray-950 |

**Fix suggestion:** Replace `text-gray-500` with `text-gray-400` for body text on gray-950 backgrounds. The Akaike weight percentages and CI values are the most affected — they carry scientific meaning and should be readable.

#### Pro Modal — same base issues as app page, plus:
- Modal backdrop doesn't fully isolate contrast context

---

## Keyboard Navigation

**Tab order (50 elements tested):**

1. Save Project (PRO) button
2. Sample data buttons: Enzyme Kinetics → Dose-Response → Bacterial Growth → Radioactive Decay → Gaussian Peak → Adsorption Isotherm
3. Textarea (data input)
4. Custom model input
5. Auto-Fit All Models button
6. Model ranking listbox (all 25 models individually focusable, supports ArrowUp/ArrowDown)
7. Chart controls: Log X → Log Y
8. Action toolbar: Show Residuals → Hide Uncertainties → Export SVG → Export CSV → Export PDF (PRO) → Export PNG 300dpi (PRO) → Copy Params
9. GitHub footer link
10. Wraps back to top

**Focus visibility:** ✅ PASS — focus ring visible on last focused element (all buttons have `focus:ring-2 focus:ring-blue-500`).

**Keyboard issues:** None found. The ranking listbox supports both Tab (move between elements) and ArrowUp/ArrowDown (move within list). All interactive elements are reachable.

---

## Color Contrast Details

22 elements fail WCAG AA contrast requirements, all in the app page:

- **Most common pattern:** `text-gray-500` (#6B7280) on `bg-gray-950` (#030712) — contrast ratio ≈ 3.8:1 (needs 4.5:1 for normal text)
- **Second pattern:** `text-gray-600` (#4B5563) on `bg-gray-900` (#111827) — contrast ratio ≈ 2.8:1 (critically low)
- **Cyan buttons:** `bg-cyan-600` (#0891B2) with white text — ratio ≈ 3.2:1 (needs 4.5:1 for normal text, passes 3:1 for large text)

**Recommended fixes (by priority):**

1. `text-gray-600` → `text-gray-400` (highest impact — currently ≈2.8:1)
2. `text-gray-500` for data values → `text-gray-400` (22 elements)
3. Cyan CTA buttons: either darken to `bg-cyan-700` or increase font-size/weight to qualify as "large text" (already font-semibold, may pass at 18px+)

---

## What's good

- All interactive elements have ARIA labels or roles
- Model ranking uses `role="listbox"` with `role="option"` and `aria-selected`
- Chart has `role="img"` with descriptive `aria-label`
- Toolbars use `role="toolbar"` with `aria-label`
- Toggle buttons use `aria-pressed`
- Warnings use `role="alert"`
- Quality badges use `role="status"`
- Complete keyboard navigation with visible focus indicators
- No missing alt text, no missing form labels
- Semantic HTML structure (nav, main, header, footer, section)
