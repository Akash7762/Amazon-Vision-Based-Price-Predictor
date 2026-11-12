// The README's screenshots (docs/images/) and the frames of its demo GIF
// (test-results/demo-frames/, made into docs/images/demo.gif by
// scripts/make_demo_gif.py). Run with: npx playwright test -c media.config.ts
import { expect, test, type Page } from "@playwright/test";
import path from "node:path";
import { choose, expected, photoPath, shownEstimate } from "../tests/helpers";

const IMAGES = path.resolve(__dirname, "..", "..", "docs", "images");
const FRAMES = path.resolve(__dirname, "..", "test-results", "demo-frames");
const photo = expected.photos.find((p) => p.file === "olive-oil.jpg")!;

// CSS pixels, so a phone screenshot isn't 2.6 times bigger than the screen.
const shoot = (page: Page, file: string) => page.screenshot({ path: file, scale: "css", animations: "disabled" });

test.use({ serviceWorkers: "block", colorScheme: "light" });

test("start and result screens", async ({ page }, testInfo) => {
  const device = testInfo.project.name;
  await page.goto("/");
  await expect(page.getByText("Choose a photo")).toBeVisible();
  await shoot(page, path.join(IMAGES, `${device}-start.png`));

  await choose(page, photoPath(photo.file));
  await shownEstimate(page);
  await shoot(page, path.join(IMAGES, `${device}-result.png`));
});

test("demo frames: choose a photo, wait, see the price", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "one GIF is enough");
  let frame = 0;
  const grab = () => shoot(page, path.join(FRAMES, `${String(++frame).padStart(2, "0")}.png`));

  await page.goto("/");
  await expect(page.getByText("Choose a photo")).toBeVisible();
  await grab();

  // Hold the answer back a moment so the "working" state can be photographed.
  let release = () => {};
  const held = new Promise<void>((resolve) => (release = resolve));
  await page.route("**/api/predict", async (route) => (await held, route.continue()));
  await choose(page, photoPath(photo.file));
  await expect(page.getByText("Estimating the price…")).toBeVisible();
  await grab();

  release();
  await shownEstimate(page);
  await grab();
});
