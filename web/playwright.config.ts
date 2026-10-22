import { defineConfig, devices } from "@playwright/test";

// Runs against the static export built with NEXT_PUBLIC_MOCK=1 (no API needed):
//   NEXT_PUBLIC_MOCK=1 npm run build && npm run e2e
export default defineConfig({
  testDir: "tests/e2e",
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: { baseURL: "http://localhost:4174", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
  webServer: { command: "node scripts/serve.mjs 4174", url: "http://localhost:4174", reuseExistingServer: !process.env.CI },
});
