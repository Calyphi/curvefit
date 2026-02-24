"use client";

import { useEffect } from "react";
import { initSentry } from "@/lib/sentry";

export default function SentryProvider({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    initSentry();
  }, []);

  return <>{children}</>;
}
