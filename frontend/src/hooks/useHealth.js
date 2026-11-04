import { useEffect, useState } from "react";
import { getHealth } from "../api.js";

// One shared, polled copy of the backend status, so new models show up without a reload.
let cache = null;
let timer = null;
const listeners = new Set();

async function refresh() {
  try {
    cache = { ok: true, data: await getHealth() };
  } catch {
    cache = { ok: false, data: null };
  }
  listeners.forEach((fn) => fn(cache));
}

export default function useHealth() {
  const [state, setState] = useState(cache);
  useEffect(() => {
    listeners.add(setState);
    if (!cache) refresh();
    if (!timer) timer = setInterval(refresh, 15000);
    return () => {
      listeners.delete(setState);
      if (!listeners.size) {
        clearInterval(timer);
        timer = null;
      }
    };
  }, []);
  return state;
}
