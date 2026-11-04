import { useEffect, useState } from "react";

// In-memory state that survives switching between pages (cleared on a full reload).
const memory = new Map();

export function useKept(key, initial) {
  const [value, setValue] = useState(() => (memory.has(key) ? memory.get(key) : initial));
  useEffect(() => {
    memory.set(key, value);
  }, [key, value]);
  return [value, setValue];
}
