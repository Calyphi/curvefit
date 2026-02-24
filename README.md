# Calyphi CurveFit

**Scientific curve fitting — instant, accurate, private.**

CurveFit is a free, open-source, browser-based tool for nonlinear regression. It fits 25+ scientific models to your data, ranks them with AICc / Akaike weights, and reports confidence intervals — all without your data ever leaving your device.

**Live app:** [calyphi.com/app](https://calyphi.com/app)

---

## Features

- **25+ built-in models** — linear, polynomial, exponential, Hill, Michaelis-Menten, sigmoidal, Gaussian, power-law, logistic, and more
- **Custom equations** — enter any formula with automatic parameter detection
- **Levenberg-Marquardt** optimisation with multi-start global search
- **AICc model selection** with Akaike weights for evidence-based ranking
- **≈95% confidence intervals** on parameters (delta-method) and prediction bands on the curve
- **Quality badges** — Good / Needs Review / Unreliable flags per model
- **Export** — SVG, CSV, clipboard. PDF and PNG (Pro, coming soon)
- **100% client-side** — no server, no account, no data upload

## Tech stack

- [Next.js 15](https://nextjs.org/) (App Router, TypeScript)
- [Recharts](https://recharts.org/) for interactive charts
- [Tailwind CSS 4](https://tailwindcss.com/) for styling
- [Vercel](https://vercel.com/) for hosting
- [Playwright](https://playwright.dev/) for E2E testing (64 tests)

## Getting started

```bash
npm install
npm run dev
```

Open [localhost:3000](http://localhost:3000).

## Running tests

```bash
npx playwright install
npx playwright test
```

## License

MIT — see [LICENSE](./LICENSE).

---

**Calyphi** builds open, browser-first tools for researchers.
