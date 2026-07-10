import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,          // exposes on 0.0.0.0 → shows Network URL
    port: 5173,
    proxy: { "/api": "http://localhost:5000" },
  },
});
