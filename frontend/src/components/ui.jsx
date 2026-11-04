import { useEffect, useState } from "react";

export function PageHeader({ title, children }) {
  return (
    <header className="mb-8 max-w-2xl">
      <h1 className="t-h1">{title}</h1>
      {children && <p className="mt-2 text-graphite">{children}</p>}
    </header>
  );
}

// Shown while a request runs: a ghost document with a scan band passing over it.
// Counts seconds so a long first run (model download) doesn't look frozen.
export function Working({ label }) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="sheet examine p-6" role="status" aria-live="polite">
      <div className="flex items-baseline justify-between gap-4">
        <p className="t-h2">{label}</p>
        <p className="t-small text-graphite">{seconds}s</p>
      </div>
      <div className="mt-5 space-y-3" aria-hidden="true">
        <div className="ghost-line w-11/12" />
        <div className="flex gap-2">
          <div className="ghost-line w-1/4" />
          <div className="ghost-line ghost-mark w-1/5" />
          <div className="ghost-line w-2/5" />
        </div>
        <div className="ghost-line w-10/12" />
        <div className="flex gap-2">
          <div className="ghost-line w-1/3" />
          <div className="ghost-line ghost-mark real w-1/6" />
          <div className="ghost-line w-1/4" />
        </div>
        <div className="ghost-line w-7/12" />
      </div>
      {seconds > 12 && (
        <p className="t-small mt-5 max-w-[60ch] text-graphite">
          The first check after starting the app loads the AI models, which can take a few minutes. Later checks take
          seconds.
        </p>
      )}
    </div>
  );
}

export function ErrorNote({ message, title = "That didn't work" }) {
  if (!message) return null;
  return (
    <div className="rounded-[10px] border border-fake/25 bg-fake-tint px-4 py-3" role="alert">
      <p className="font-semibold text-fake">{title}</p>
      <p className="t-small mt-0.5 text-ink">{message}</p>
    </div>
  );
}

export function LanguagePicker({ value, onChange }) {
  const options = [
    ["auto", "Detect"],
    ["en", "English"],
    ["te", "తెలుగు"],
  ];
  return (
    <div className="segmented" role="group" aria-label="Language">
      {options.map(([v, label]) => (
        <button key={v} type="button" aria-pressed={value === v} onClick={() => onChange(v)}>
          {label}
        </button>
      ))}
    </div>
  );
}

export function Switch({ checked, onChange, label, hint }) {
  return (
    <label className="flex cursor-pointer items-start gap-3">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        className="switch mt-0.5"
        onClick={() => onChange(!checked)}
      />
      <span>
        <span className="block text-sm font-medium">{label}</span>
        {hint && <span className="t-small block text-graphite">{hint}</span>}
      </span>
    </label>
  );
}

export function Chip({ tone = "neutral", children }) {
  const tones = {
    fake: "bg-fake-tint text-fake",
    real: "bg-real-tint text-real",
    mixed: "bg-mixed-tint text-mixed",
    neutral: "bg-rule-soft text-graphite",
  };
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full py-0.5 pr-2.5 pl-2 text-[0.8125rem] font-semibold ${tones[tone]}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" />
      {children}
    </span>
  );
}

// Right-hand column before any result exists: a faint outline of the report, with what it will contain.
export function EmptyReport({ title, points }) {
  return (
    <div className="relative overflow-hidden rounded-[14px] border border-dashed border-rule">
      <div className="pointer-events-none space-y-6 p-6 opacity-70" aria-hidden="true">
        <div className="verdict-strip" style={{ "--strip": "var(--color-rule)" }}>
          <div className="ghost-line h-7 w-2/5" />
          <div className="mt-5 h-2.5 rounded-full bg-rule-soft" />
        </div>
        <div className="space-y-3">
          <div className="ghost-line w-full" />
          <div className="flex gap-2">
            <div className="ghost-line w-1/3" />
            <div className="ghost-line ghost-mark w-1/5" />
            <div className="ghost-line w-1/3" />
          </div>
          <div className="ghost-line w-11/12" />
          <div className="flex gap-2">
            <div className="ghost-line ghost-mark real w-1/6" />
            <div className="ghost-line w-2/3" />
          </div>
        </div>
        <div className="space-y-3">
          {["bg-real", "bg-fake", "bg-mist"].map((c) => (
            <div key={c} className="flex items-center gap-3">
              <span className={`h-8 w-[3px] rounded-full ${c} opacity-60`} />
              <div className="ghost-line flex-1" />
              <div className="ghost-line w-16" />
            </div>
          ))}
        </div>
      </div>
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-paper via-paper/95 to-transparent px-6 pt-16 pb-6">
        <p className="t-h3">{title}</p>
        <ul className="mt-2 space-y-1.5 text-graphite">
          {points.map((p) => (
            <li key={p} className="flex gap-2">
              <span className="mt-2 h-1.5 w-1.5 flex-none rounded-full bg-mist" aria-hidden="true" />
              {p}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// Actions under a finished report. Printing uses the print stylesheet (forms and navigation hidden).
export function ReportActions({ onReset }) {
  return (
    <div className="no-print flex flex-wrap items-center gap-2">
      <button type="button" className="btn-quiet" onClick={() => window.print()}>
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" aria-hidden="true">
          <path d="M7 9V3h10v6M7 17H5a2 2 0 0 1-2-2v-4a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2h-2M7 14h10v7H7z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
        </svg>
        Save as PDF
      </button>
      {onReset && (
        <button type="button" className="btn-quiet" onClick={onReset}>
          Start a new check
        </button>
      )}
    </div>
  );
}
