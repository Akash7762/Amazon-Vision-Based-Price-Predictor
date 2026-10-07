// The installable app: manifest, icons, what Chrome thinks of it, and opening
// with no server at all once it has been visited.
import { expect, test } from "@playwright/test";
import { execSync, spawn, type ChildProcess } from "node:child_process";
import path from "node:path";

test("manifest has what an install needs, and every icon is the size it claims", async ({ request }) => {
  const manifest = await (await request.get("/manifest.webmanifest")).json();
  expect(manifest).toMatchObject({
    name: "Vision Price Predictor",
    short_name: "Price Predictor",
    start_url: "/",
    display: "standalone",
  });
  expect(manifest.icons.map((i: { purpose?: string }) => i.purpose ?? "any")).toContain("maskable");
  for (const icon of manifest.icons) {
    const res = await request.get(icon.src);
    expect(res.headers()["content-type"]).toBe("image/png");
    const png = await res.body();
    expect(`${png.readUInt32BE(16)}x${png.readUInt32BE(20)}`, icon.src).toBe(icon.sizes); // PNG header
  }
});

test("Chrome finds nothing stopping an install", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Choose a photo")).toBeVisible();
  const cdp = await page.context().newCDPSession(page);
  const { installabilityErrors } = await cdp.send("Page.getInstallabilityErrors");
  // Test browsers run like incognito windows, where Chrome never offers to
  // install anything; any other reason would show up here.
  expect(installabilityErrors.map((e) => e.errorId).filter((id) => id !== "in-incognito")).toEqual([]);
});

// Its own server, on its own port, so the test can switch it off.
const OFFLINE_PORT = 3101;
const frontend = path.resolve(__dirname, "..", "..", "frontend");

function startServer() {
  return spawn(process.execPath, ["node_modules/next/dist/bin/next", "start", "--port", String(OFFLINE_PORT)], {
    cwd: frontend,
    env: { ...process.env, NEXT_DIST_DIR: ".next-e2e" },
    stdio: "ignore",
  });
}

function stopServer(server: ChildProcess) {
  if (server.exitCode !== null || !server.pid) return;
  if (process.platform === "win32") execSync(`taskkill /pid ${server.pid} /T /F`, { stdio: "ignore" });
  else server.kill();
}

async function serverUp() {
  try {
    return (await fetch(`http://127.0.0.1:${OFFLINE_PORT}/manifest.webmanifest`)).ok;
  } catch {
    return false;
  }
}

test("after one visit it opens with the server switched off", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome", "starts its own server, once is enough");
  const server = startServer();
  try {
    await expect.poll(serverUp, { timeout: 30_000 }).toBe(true);
    await page.goto(`http://localhost:${OFFLINE_PORT}/`);
    const button = page.getByText("Choose a photo");
    const styled = await button.evaluate((el) => getComputedStyle(el).backgroundColor);

    // Wait until the service worker holds the page and everything it loaded.
    await expect
      .poll(
        () =>
          page.evaluate(async () => {
            const loaded = performance
              .getEntriesByType("resource")
              .map((e) => e.name)
              .filter((url) => url.startsWith(`${location.origin}/_next/static/`));
            const missing = [];
            for (const url of [`${location.origin}/`, ...loaded]) if (!(await caches.match(url))) missing.push(url);
            return missing;
          }),
        { timeout: 15_000 },
      )
      .toEqual([]);

    stopServer(server);
    await expect.poll(serverUp).toBe(false);

    await page.reload();
    await expect(page.getByRole("heading", { name: "Vision Price Predictor" })).toBeVisible();
    expect(await button.evaluate((el) => getComputedStyle(el).backgroundColor)).toBe(styled);
    await expect(page.getByRole("status")).toContainText("isn't reachable");
  } finally {
    stopServer(server);
  }
});
