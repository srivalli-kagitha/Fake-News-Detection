import { useEffect, useState } from "react";

// "system" follows the OS setting; "light"/"dark" force it. Saved per browser.
const KEY = "truthlens-theme";

function read() {
  try {
    return localStorage.getItem(KEY) || "system";
  } catch {
    return "system";
  }
}

function apply(theme) {
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
}

export function useTheme() {
  const [theme, setTheme] = useState(read);
  useEffect(() => {
    apply(theme);
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      /* private mode: theme still applies for this visit */
    }
  }, [theme]);
  return [theme, setTheme];
}
