import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// CI sets VITE_BASE to "/<repo-name>/" for GitHub Pages project sites.
export default defineConfig({
  base: process.env.VITE_BASE ?? "/",
  plugins: [react()],
});
