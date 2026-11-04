import { useLayoutEffect, useRef, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import useHealth from "../hooks/useHealth.js";
import { useTheme } from "../lib/theme.js";

const ICONS = {
  home: "M3 10.5 12 4l9 6.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z",
  text: "M5 5h14M5 9.5h14M5 14h9M5 18.5h6",
  image: "M4 5h16v14H4zM4 15l4.5-4.5 4 4L15 12l5 5M15.5 8.5h.01",
  claims: "M9 11l2.2 2.2L15.5 9M12 3l7 3v5c0 4.4-3 8.3-7 10-4-1.7-7-5.6-7-10V6z",
  batch: "M4 7h16M4 12h16M4 17h16M8 4v16",
  history: "M12 7v5l3 2M3.5 12a8.5 8.5 0 1 0 2.5-6M3 4v4h4",
};

const LINKS = [
  { to: "/", label: "Start", icon: "home" },
  { to: "/text", label: "Check news text", icon: "text" },
  { to: "/image", label: "Check an image", icon: "image" },
  { to: "/claims", label: "Verify claims", icon: "claims" },
  { to: "/batch", label: "Check many at once", icon: "batch" },
  { to: "/history", label: "History", icon: "history" },
];

function Icon({ name }) {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4 flex-none" fill="none" aria-hidden="true">
      <path d={ICONS[name]} stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Status({ name, state }) {
  const map = {
    loaded: ["bg-real", "Ready"],
    ready: ["bg-real", "Ready"],
    not_trained: ["bg-mixed", "Not trained"],
    pretrained: ["bg-real", "Ready"],
    offline: ["bg-fake", "Offline"],
  };
  const [dot, text] = map[state] || ["bg-mist", "Checking"];
  const motion = text === "Ready" ? "dot-live" : text === "Checking" ? "dot-wait" : "";
  return (
    <li className="flex items-center justify-between gap-2 py-[3px] text-[0.75rem]">
      <span className="truncate text-[var(--rail-ink)]">{name}</span>
      <span className="flex flex-none items-center gap-1.5 text-[var(--rail-muted)]">
        <span className={`h-1.5 w-1.5 rounded-full ${dot} ${motion}`} />
        {text}
      </span>
    </li>
  );
}

export function Logo() {
  return (
    <NavLink to="/" className="flex items-center gap-2 rounded-md">
      <svg viewBox="0 0 32 32" className="h-7 w-7" aria-hidden="true">
        <rect x=".5" y=".5" width="31" height="31" rx="8" fill="#0A0A0A" stroke="#2a2a2a" />
        <g className="logo-lens">
          <circle cx="14" cy="14" r="6.5" fill="none" stroke="#fff" strokeWidth="2.4" />
          <path d="M19 19l5.5 5.5" stroke="#fff" strokeWidth="2.6" strokeLinecap="round" />
        </g>
        <path className="logo-scan" d="M10.8 14.2h6.4" stroke="#3B82F6" strokeWidth="3.2" strokeLinecap="round" />
      </svg>
      <span className="text-[0.98rem] font-bold tracking-[-0.025em]">TruthLens AI</span>
    </NavLink>
  );
}

function ThemeSwitch() {
  const [theme, setTheme] = useTheme();
  return (
    <div className="segmented w-full" role="group" aria-label="Colour theme">
      {[
        ["system", "Auto"],
        ["light", "Light"],
        ["dark", "Dark"],
      ].map(([v, label]) => (
        <button key={v} type="button" className="flex-1" aria-pressed={theme === v} onClick={() => setTheme(v)}>
          {label}
        </button>
      ))}
    </div>
  );
}

// Desktop nav: one marker glides to the active tool, a softer one follows the pointer.
function RailNav() {
  const nav = useRef(null);
  const { pathname } = useLocation();
  const [active, setActive] = useState(null);
  const [hover, setHover] = useState(null);

  useLayoutEffect(() => {
    const el = nav.current?.querySelector('[aria-current="page"]');
    setActive(el ? { y: el.offsetTop, h: el.offsetHeight } : null);
  }, [pathname]);

  return (
    <nav ref={nav} className="relative mt-7 space-y-px" aria-label="Tools" onMouseLeave={() => setHover(null)}>
      <span
        className="rail-hover"
        style={hover ? { transform: `translateY(${hover.y}px)`, height: hover.h, opacity: 1 } : { opacity: 0, height: hover?.h }}
        aria-hidden="true"
      />
      {active && (
        <span className="rail-marker" style={{ transform: `translateY(${active.y}px)`, height: active.h }} aria-hidden="true" />
      )}
      {LINKS.map((l, i) => (
        <NavLink
          key={l.to}
          to={l.to}
          end
          className="rail-link rail-in"
          style={{ "--i": i }}
          onMouseEnter={(e) => setHover({ y: e.currentTarget.offsetTop, h: e.currentTarget.offsetHeight })}
        >
          <Icon name={l.icon} />
          <span>{l.label}</span>
        </NavLink>
      ))}
    </nav>
  );
}

export default function Sidebar() {
  const health = useHealth();
  const models = health?.data?.models;
  const offline = health && !health.ok;


  return (
    <>
      {/* Desktop rail */}
      <aside className="rail sticky top-0 hidden h-screen w-56 flex-none flex-col border-r border-[var(--rail-line)] px-3 py-5 lg:flex">
        <div className="px-2">
          <Logo />
        </div>
        <RailNav />
        <div className="rail-in mt-auto border-t border-[var(--rail-line)] px-2 pt-4" style={{ "--i": 8 }}>
          <p className="mb-1 text-[0.6875rem] font-semibold text-[var(--rail-muted)]">Models</p>
          <ul>
            <Status name="English text" state={offline ? "offline" : models?.text_en} />
            <Status name="Telugu text" state={offline ? "offline" : models?.text_te} />
            <Status name="Images" state={offline ? "offline" : models?.image} />
            <Status name="Claim checking" state={offline ? "offline" : models ? "pretrained" : undefined} />
          </ul>
          {offline && <p className="mt-2 text-[0.75rem] text-fake">The backend isn't running. Start the app with start.bat.</p>}
          <div className="mt-4">
            <ThemeSwitch />
          </div>
        </div>
      </aside>

      {/* Mobile top bar */}
      <header className="rail sticky top-0 z-20 border-b border-[var(--rail-line)] lg:hidden">
        <div className="flex items-center justify-between px-4 pt-3">
          <Logo />
          {offline && <span className="t-small text-fake">Backend offline</span>}
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 py-2" aria-label="Tools">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} end className="rail-link flex-none whitespace-nowrap py-1.5">
              <span>{l.label}</span>
            </NavLink>
          ))}
        </nav>
      </header>
    </>
  );
}
