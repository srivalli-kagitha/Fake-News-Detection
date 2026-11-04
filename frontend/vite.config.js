import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backend = env.BACKEND_URL || "http://127.0.0.1:8000";
  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      strictPort: true,
      // All /api calls go to the FastAPI backend - no CORS issues.
      proxy: { "/api": { target: backend, changeOrigin: true, timeout: 0, proxyTimeout: 0 } },
    },
  };
});
