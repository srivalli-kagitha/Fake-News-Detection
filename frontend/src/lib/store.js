import { useEffect, useState } from "react";

// State that survives switching pages AND a page refresh.
// - Values live in this module, not in the page, so a check that is still running when you open
//   another page still lands here, and the result is waiting when you come back.
// - Values are also copied to sessionStorage (this browser tab only), so a refresh keeps them.
//   Files (an uploaded image) cannot be stored there; the report keeps a thumbnail instead.
const memory = new Map();
const listeners = new Map();
const PREFIX = "truthlens:";

function readStored(key) {
  try {
    const raw = sessionStorage.getItem(PREFIX + key);
    return raw === null ? undefined : JSON.parse(raw);
  } catch {
    return undefined;
  }
}

function writeStored(key, value) {
  try {
    if (value === null || value === undefined) sessionStorage.removeItem(PREFIX + key);
    else if (!(value instanceof Blob)) sessionStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // storage full or blocked: the value still lives in memory for this session
  }
}

export function getKept(key, initial) {
  if (!memory.has(key)) {
    const stored = readStored(key);
    memory.set(key, stored === undefined ? initial : stored);
  }
  return memory.get(key);
}

export function setKept(key, value, persist = true) {
  const next = typeof value === "function" ? value(memory.get(key)) : value;
  memory.set(key, next);
  if (persist) writeStored(key, next);
  listeners.get(key)?.forEach((fn) => fn(next));
}

// Like useState, but kept as described above. persist=false for things like "loading" that must not
// come back after a refresh (the request itself does not survive one).
export function useKept(key, initial, persist = true) {
  const [value, setValue] = useState(() => (persist ? getKept(key, initial) : memory.has(key) ? memory.get(key) : initial));
  useEffect(() => {
    if (!listeners.has(key)) listeners.set(key, new Set());
    const set = listeners.get(key);
    set.add(setValue);
    if (memory.has(key)) setValue(memory.get(key)); // changed while this page was closed
    return () => set.delete(setValue);
  }, [key]);
  return [value, (v) => setKept(key, v, persist)];
}
