import { defineConfig, devices } from "@playwright/test";

// The console served by Caddy in the compose stack (deploy/ci-stack.sh), real API behind.
export default defineConfig({
  testDir: "tests/stack",
  timeout: 180_000,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: { baseURL: process.env.STACK_URL ?? "http://localhost:8080", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
});
