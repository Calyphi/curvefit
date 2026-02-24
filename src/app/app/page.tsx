"use client";

import dynamic from "next/dynamic";

const CurveFitter = dynamic(() => import("@/components/CurveFitter"), {
  ssr: false,
  loading: () => (
    <div className="flex h-screen items-center justify-center bg-gray-950">
      <div className="text-center">
        <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-4 border-cyan-400 border-t-transparent" />
        <p className="text-gray-400">Loading CurveFit…</p>
      </div>
    </div>
  ),
});

export default function AppPage() {
  return <CurveFitter />;
}
