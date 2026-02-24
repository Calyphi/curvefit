"use client";

import React from "react";
import * as Sentry from "@sentry/nextjs";

interface Props {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    Sentry.captureException(error, { extra: { componentStack: errorInfo.componentStack } });
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div className="flex min-h-[50vh] flex-col items-center justify-center gap-4 p-8 text-center">
            <div className="text-4xl">⚠️</div>
            <h2 className="text-xl font-semibold text-white">Something went wrong</h2>
            <p className="max-w-md text-sm text-gray-400">
              An unexpected error occurred. The error has been reported automatically.
            </p>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-cyan-500"
            >
              Try again
            </button>
          </div>
        )
      );
    }

    return this.props.children;
  }
}
