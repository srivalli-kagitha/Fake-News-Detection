import { useEffect, useState } from "react";

// State that survives switching between pages (cleared on a refresh, which starts a clean page).
// Values live in this module, not in the page, so a check that is still running when you open
// another page still lands here, and the result is waiting when you come back.
const memory = new Map();
const listeners = new Map();

// Older versions kept results in sessionStorage across refreshes: remove those leftovers.
try {
  Object.keys(sessionStorage)
    .filter((k) => k.startsWith("truthlens:"))
    .forEach((k) => sessionStorage.removeItem(k));
} catch {
  // storage blocked: nothing to clean
}

export function setKept(key, value) {
  const next = typeof value === "function" ? value(memory.get(key)) : value;
  memory.set(key, next);
  listeners.get(key)?.forEach((fn) => fn(next));
}

// Like useState, but kept across page switches as described above.
export function useKept(key, initial) {
  const [value, setValue] = useState(() => (memory.has(key) ? memory.get(key) : initial));
  useEffect(() => {
    if (!listeners.has(key)) listeners.set(key, new Set());
    const set = listeners.get(key);
    set.add(setValue);
    if (memory.has(key)) setValue(memory.get(key)); // changed while this page was closed
    return () => set.delete(setValue);
  }, [key]);
  return [value, (v) => setKept(key, v)];
}
