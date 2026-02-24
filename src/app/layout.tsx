import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import SentryProvider from "@/components/SentryProvider";
import ErrorBoundary from "@/components/ErrorBoundary";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: {
    default: "Calyphi — Precision Tools for Scientific Data",
    template: "%s",
  },
  description:
    "Calyphi builds open, browser-first tools for researchers. Rigorous statistics, zero data collection, no subscriptions. Start with CurveFit.",
  metadataBase: new URL("https://calyphi.com"),
  openGraph: {
    title: "Calyphi — Precision Tools for Scientific Data",
    description: "Open, browser-first tools for researchers. Start with CurveFit — scientific curve fitting.",
    url: "https://calyphi.com",
    siteName: "Calyphi",
    type: "website",
    images: [
      {
        url: "https://calyphi.com/og-calyphi.png",
        width: 1200,
        height: 630,
        alt: "Calyphi — Precision Tools for Scientific Data",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Calyphi — Precision Tools for Scientific Data",
    description: "Open, browser-first tools for researchers. Start with CurveFit.",
    images: ["https://calyphi.com/og-calyphi.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <SentryProvider>
          <ErrorBoundary>{children}</ErrorBoundary>
        </SentryProvider>
        <Analytics />
      </body>
    </html>
  );
}
