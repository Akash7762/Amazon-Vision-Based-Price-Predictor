import { defineConfig, devices } from "@playwright/test";
import base from "./playwright.config";

// Not tests: makes the README's screenshots and the frames of its demo GIF,
// using the same running app as the end-to-end tests.
//
//   npx playwright test -c media.config.ts
//   python scripts/make_demo_gif.py
export default defineConfig({
  ...base,
  testDir: "media",
  reporter: "list",
  retries: 0,
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], channel: "chrome", viewport: { width: 1100, height: 720 } } },
    { name: "phone", use: { ...devices["Pixel 7"], channel: "chrome" } },
  ],
});
