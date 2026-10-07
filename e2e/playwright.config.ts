import { defineConfig, devices } from "@playwright/test";
import path from "node:path";

// The tests run the whole app: the FastAPI backend with the real model, and a
// production build of the web app that forwards /api to it. They use their own
// ports and their own build folder (frontend/.next-e2e), so an app you already
// have running on 8000 and 3000 is left alone.
export const API_PORT = 8100;
export const WEB_PORT = 3100;

const repo = path.resolve(__dirname, "..");
const python =
  process.env.PYTHON ?? path.join(repo, "venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");

export default defineConfig({
  testDir: "tests",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"], ["html", { open: "never" }], ["json", { outputFile: "test-results/results.json" }]],
  use: {
    baseURL: `http://localhost:${WEB_PORT}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  // The browsers already on the PC, so nothing extra to download. The phone is
  // Chrome emulating a Pixel 7: its screen size, touch and user agent.
  projects: [
    { name: "desktop-chrome", use: { ...devices["Desktop Chrome"], channel: "chrome" } },
    { name: "desktop-edge", use: { ...devices["Desktop Edge"], channel: "msedge" } },
    { name: "android-phone", use: { ...devices["Pixel 7"], channel: "chrome" } },
  ],
  webServer: [
    {
      command: `"${python}" -m uvicorn backend.app.main:app --port ${API_PORT} --timeout-keep-alive 75`,
      cwd: repo,
      url: `http://127.0.0.1:${API_PORT}/health`,
      timeout: 120_000,
      reuseExistingServer: !process.env.CI,
    },
    {
      command: `npm run build && npm run start -- --port ${WEB_PORT}`,
      cwd: path.join(repo, "frontend"),
      env: { API_URL: `http://127.0.0.1:${API_PORT}`, NEXT_DIST_DIR: ".next-e2e" },
      url: `http://127.0.0.1:${WEB_PORT}/manifest.webmanifest`,
      timeout: 300_000,
      reuseExistingServer: !process.env.CI,
    },
  ],
});
