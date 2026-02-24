import Link from "next/link";

const features = [
  {
    title: "25+ Models",
    desc: "From linear to Hill, Michaelis-Menten, sigmoidal, power-law, and custom equations.",
  },
  {
    title: "Levenberg-Marquardt",
    desc: "Gold-standard nonlinear regression with multi-start to avoid local minima.",
  },
  {
    title: "AICc Model Selection",
    desc: "Akaike weights rank every model — no guessing which curve fits best.",
  },
  {
    title: "Confidence Bands",
    desc: "≈95 % CI on parameters and prediction bands on the curve, automatically.",
  },
  {
    title: "100 % Private",
    desc: "Runs entirely in your browser. Zero data ever leaves your device.",
  },
  {
    title: "Instant Results",
    desc: "Paste data, click fit, done. No install, no account, no waiting.",
  },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Nav */}
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
        <Link href="/" className="text-xl font-bold tracking-tight">
          <span className="text-cyan-400">Calyphi</span>
        </Link>
        <div className="flex items-center gap-6">
          <a href="#products" className="text-sm text-gray-400 transition hover:text-white">
            Products
          </a>
          <Link
            href="/app"
            className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-cyan-500"
          >
            CurveFit
          </Link>
        </div>
      </nav>

      {/* Hero — side-by-side layout */}
      <header className="mx-auto max-w-6xl px-6 pb-16 pt-24">
        <div className="flex flex-col items-center gap-12 lg:flex-row lg:gap-16">
          {/* Left — text */}
          <div className="flex-1 text-center lg:text-left">
            <p className="mb-4 text-sm font-medium uppercase tracking-widest text-cyan-400">
              Introducing Calyphi
            </p>
            <h1 className="text-4xl font-extrabold leading-tight tracking-tight sm:text-5xl md:text-6xl">
              Precision tools
              <br />
              <span className="text-cyan-400">for scientific data</span>
            </h1>
            <p className="mx-auto mt-6 max-w-xl text-lg text-gray-400 lg:mx-0">
              Calyphi builds open, browser-first tools for researchers. Rigorous statistics, zero
              data collection, no subscriptions. Start with CurveFit — more instruments coming soon.
            </p>
            <div className="mt-10 flex flex-col items-center gap-4 sm:flex-row lg:justify-start">
              <Link
                href="/app"
                className="inline-flex items-center gap-2 rounded-xl bg-cyan-600 px-8 py-3.5 text-base font-semibold text-white shadow-lg shadow-cyan-600/25 transition hover:bg-cyan-500"
              >
                Open CurveFit — Free
              </Link>
            </div>
          </div>

          {/* Right — screenshot */}
          <div className="flex-1">
            <img
              src="/hero-screenshot.png"
              alt="CurveFit fitting Michaelis-Menten to enzyme kinetics data"
              className="w-full rounded-xl shadow-2xl shadow-black/50 border border-gray-800"
              width={1280}
              height={800}
            />
          </div>
        </div>
      </header>

      {/* CurveFit — first product */}
      <section id="products" className="mx-auto max-w-4xl px-6 pb-8 pt-16 scroll-mt-20">
        <div className="flex items-center gap-3">
          <span className="rounded-full bg-cyan-600/20 px-3 py-1 text-xs font-semibold text-cyan-400">
            Product #1
          </span>
        </div>
        <h2 className="mt-4 text-3xl font-bold sm:text-4xl">CurveFit</h2>
        <p className="mt-2 text-lg text-gray-400">
          Scientific curve fitting — instant, accurate, private.
        </p>
      </section>

      {/* Problem */}
      <section className="mx-auto max-w-4xl px-6 pt-8 pb-20">
        <div className="rounded-2xl border border-gray-800 bg-gray-900/60 p-8 md:p-12">
          <h3 className="text-2xl font-bold">The problem</h3>
          <p className="mt-4 text-gray-400 leading-relaxed">
            GraphPad Prism costs <strong className="text-white">&euro;520 /year</strong>. Origin,
            SigmaPlot, and MATLAB carry similar price tags. For researchers who just need
            to fit a curve — especially in labs with tight budgets — that&apos;s hard to justify.
          </p>
          <p className="mt-4 text-gray-400 leading-relaxed">
            Meanwhile, free alternatives are either too basic (no model selection, no error
            estimation) or too complex (scripting in R or Python for every dataset).
          </p>
          <p className="mt-4 text-gray-300 font-medium">
            Calyphi CurveFit gives you publication-ready fitting in seconds — for free, with no
            sign-up, and with complete data privacy.
          </p>
        </div>
      </section>

      {/* Features Grid */}
      <section className="mx-auto max-w-6xl px-6 pt-4 pb-24">
        <h3 className="mb-12 text-center text-3xl font-bold">Built for scientists</h3>
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f) => (
            <div
              key={f.title}
              className="rounded-xl border border-gray-800 bg-gray-900/40 p-6 pl-5 border-l-2 border-l-blue-500 transition hover:border-blue-700"
            >
              <h4 className="text-lg font-semibold text-white">{f.title}</h4>
              <p className="mt-1 text-sm text-gray-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-gray-800 py-20 text-center">
        <h2 className="text-3xl font-bold">Ready to fit your data?</h2>
        <p className="mx-auto mt-4 max-w-lg text-gray-400">
          No account. No install. Paste your data and get results in seconds.
        </p>
        <Link
          href="/app"
          className="mt-8 inline-flex items-center gap-2 rounded-xl bg-cyan-600 px-8 py-3.5 text-base font-semibold text-white shadow-lg shadow-cyan-600/25 transition hover:bg-cyan-500"
        >
          Open CurveFit — Free
        </Link>
      </section>

      {/* Methodology */}
      <div className="text-center px-6 pt-4 pb-8">
        <p className="text-xs text-gray-500">
          Levenberg–Marquardt optimization · AICc model selection · Akaike weights · Delta-method confidence intervals · Multi-start global search
        </p>
      </div>

      {/* Footer */}
      <footer className="border-t border-gray-800 px-6 py-8">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 sm:flex-row">
          <span className="text-sm text-gray-500">&copy; 2026 Calyphi</span>
          <div className="flex gap-6 text-sm text-gray-500">
            <a href="#products" className="transition hover:text-gray-300">CurveFit</a>
            <a href="https://github.com/calyphi/curvefit" target="_blank" rel="noopener noreferrer" className="transition hover:text-gray-300">GitHub</a>
            <span className="cursor-default text-gray-700" title="Coming soon">SimFit</span>
            <span className="cursor-default text-gray-700" title="Coming soon">KinetiQ</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
