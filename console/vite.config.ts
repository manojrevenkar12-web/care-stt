import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({ plugins: [react()], server: { proxy: { "/v2": process.env.CARE_STT_API ?? "http://localhost:8000" } } });
