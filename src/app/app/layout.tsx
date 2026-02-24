import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "CurveFit by Calyphi — Scientific Curve Fitting",
  description:
    "Fit 25+ models to your data with Levenberg-Marquardt, AICc model selection, and confidence bands. 100% in-browser — your data never leaves your device.",
  openGraph: {
    title: "CurveFit by Calyphi — Scientific Curve Fitting",
    description:
      "Free, browser-based curve fitting with 25+ models, AICc ranking, and confidence intervals. No install, no account.",
    url: "https://calyphi.com/app",
    siteName: "Calyphi",
    type: "website",
    images: [
      {
        url: "https://calyphi.com/og-curvefit.png",
        width: 1200,
        height: 630,
        alt: "CurveFit by Calyphi — Scientific Curve Fitting",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "CurveFit by Calyphi — Scientific Curve Fitting",
    description:
      "Free, browser-based curve fitting with 25+ models and AICc ranking.",
    images: ["https://calyphi.com/og-curvefit.png"],
  },
};

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
