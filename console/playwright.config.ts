import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  use: { baseURL: process.env.CARE_STT_CONSOLE ?? "http://localhost:8000", viewport: { width: 1280, height: 900 } },
  reporter: [["list"]],
});
