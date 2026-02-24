"use client";

import { useState, useEffect, useRef } from "react";

const TOOLS = [
  "GraphPad Prism",
  "MATLAB",
  "Python / R",
  "Excel",
  "Origin",
  "Other",
  "Nothing",
];

export default function ProModal({ open, onClose }) {
  const [email, setEmail] = useState("");
  const [tool, setTool] = useState("");
  const [status, setStatus] = useState("idle"); // idle | sending | sent | error
  const dialogRef = useRef(null);

  // Close on Escape
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    if (open) {
      window.addEventListener("keydown", onKey);
      return () => window.removeEventListener("keydown", onKey);
    }
  }, [open, onClose]);

  // Trap focus when open
  useEffect(() => {
    if (open && dialogRef.current) {
      dialogRef.current.focus();
    }
  }, [open]);

  if (!open) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus("sending");

    try {
      const res = await fetch("https://formspree.io/f/mreaorqe", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({
          email,
          tool,
          timestamp: new Date().toISOString(),
        }),
      });
      if (!res.ok) throw new Error("Form submission failed");
      setStatus("sent");
    } catch {
      setStatus("error");
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="pro-modal-title"
        tabIndex={-1}
        className="relative mx-4 w-full max-w-md rounded-2xl border border-gray-700 bg-gray-900 p-6 shadow-2xl focus:outline-none"
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute right-4 top-4 text-gray-500 hover:text-gray-300 transition"
          aria-label="Close"
        >
          ✕
        </button>

        {status === "sent" ? (
          <div className="py-8 text-center">
            <div className="text-4xl mb-3">🎉</div>
            <h2 className="text-xl font-bold text-white">You&apos;re on the list!</h2>
            <p className="mt-2 text-sm text-gray-400">
              We&apos;ll notify you as soon as CurveFit Pro launches.
            </p>
            <button
              onClick={onClose}
              className="mt-6 rounded-lg bg-cyan-600 px-6 py-2 text-sm font-medium text-white transition hover:bg-cyan-500"
            >
              Close
            </button>
          </div>
        ) : (
          <>
            {/* Badge */}
            <div className="mb-4 flex items-center gap-2">
              <span className="rounded-full bg-amber-500/20 px-2.5 py-0.5 text-xs font-semibold text-amber-400">
                PRO
              </span>
            </div>

            <h2 id="pro-modal-title" className="text-xl font-bold text-white">
              Coming soon to CurveFit Pro
            </h2>
            <p className="mt-2 text-sm text-gray-400">
              PDF reports, high-res exports, saved projects, and more.
            </p>

            <form onSubmit={handleSubmit} className="mt-6 space-y-4">
              {/* Email */}
              <div>
                <label htmlFor="pro-email" className="block text-xs font-medium text-gray-400 mb-1">
                  Email
                </label>
                <input
                  id="pro-email"
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@university.edu"
                  className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                />
              </div>

              {/* Current tool */}
              <div>
                <label htmlFor="pro-tool" className="block text-xs font-medium text-gray-400 mb-1">
                  What do you use for curve fitting today?
                </label>
                <select
                  id="pro-tool"
                  required
                  value={tool}
                  onChange={(e) => setTool(e.target.value)}
                  className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                >
                  <option value="" disabled>Select…</option>
                  {TOOLS.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              {status === "error" && (
                <p className="text-xs text-red-400">
                  Something went wrong. Please try again.
                </p>
              )}

              <button
                type="submit"
                disabled={status === "sending"}
                className="w-full rounded-lg bg-cyan-600 py-2.5 text-sm font-semibold text-white transition hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {status === "sending" ? "Sending…" : "Notify me when Pro launches"}
              </button>
            </form>
          </>
        )}
      </div>
    </div>
  );
}
