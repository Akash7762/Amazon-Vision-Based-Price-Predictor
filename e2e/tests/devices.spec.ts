// Phone and desktop parity: each runs these at its own screen size and input
// type (touch or mouse), and the layout is checked from 320 px up to 1920 px.
import { expect, test, type Page } from "@playwright/test";
import { choose, expectCents, expected, isPhone, photoPath, shownEstimate } from "./helpers";

const photo = expected.photos[1];

/** How far the page is wider than the screen: anything above 0 means sideways scrolling. */
const overflow = (page: Page) =>
  page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);

test("phones get a camera button, desktops a drop-or-paste hint", async ({ page }, testInfo) => {
  await page.goto("/");
  const camera = page.getByText("Take a photo");
  const hint = page.getByText("or drop a photo here, or paste one");
  if (isPhone(testInfo)) {
    await expect(camera).toBeVisible();
    await expect(page.getByLabel("Take a photo")).toHaveAttribute("capture", "environment");
    await expect(page.getByLabel("Take a photo")).toHaveAttribute("accept", "image/*");
    await expect(hint).toBeHidden();
  } else {
    await expect(camera).toBeHidden();
    await expect(hint).toBeVisible();
  }
  await expect(page.getByText("Choose a photo")).toBeVisible();
  await expect(page.getByLabel("Choose a photo")).toHaveAttribute("accept", "image/*");
});

test("the same photo gets the same price on every device", async ({ page }) => {
  await page.goto("/");
  await choose(page, photoPath(photo.file));
  expectCents((await shownEstimate(page)).price, photo.price);
});

test("nothing scrolls sideways on this screen", async ({ page }) => {
  await page.goto("/");
  expect(await overflow(page)).toBe(0);
  await choose(page, photoPath(photo.file));
  await shownEstimate(page);
  expect(await overflow(page)).toBe(0);
});

test("buttons are big enough to tap", async ({ page }, testInfo) => {
  test.skip(!isPhone(testInfo), "touch screens only");
  await page.goto("/");
  for (const name of ["Take a photo", "Choose a photo"]) {
    const box = await page.getByText(name).boundingBox();
    expect(box!.height, name).toBeGreaterThanOrEqual(48);
  }
  await choose(page, photoPath(photo.file));
  await shownEstimate(page);
  const box = await page.getByRole("button", { name: "Price another photo" }).boundingBox();
  expect(box!.height).toBeGreaterThanOrEqual(48);
});

test("layout holds from a small phone to a big monitor", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome", "resizes the window itself, once is enough");
  await page.goto("/");
  await choose(page, photoPath(photo.file));
  await shownEstimate(page);
  const preview = page.getByAltText("The photo being priced");
  const price = page.locator('p:has-text("Estimated price") + p');

  for (const width of [320, 360, 390, 412, 768, 1024, 1440, 1920]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await overflow(page), `${width} px`).toBe(0);
    const photoBox = (await preview.boundingBox())!;
    const priceBox = (await price.boundingBox())!;
    if (width >= 720) {
      expect(priceBox.x, `${width} px: result beside the photo`).toBeGreaterThanOrEqual(photoBox.x + photoBox.width);
    } else {
      expect(priceBox.y, `${width} px: result under the photo`).toBeGreaterThanOrEqual(photoBox.y + photoBox.height);
    }
  }
});

test("works with the keyboard alone", async ({ page }, testInfo) => {
  test.skip(isPhone(testInfo), "keyboard use is a desktop thing");
  await page.goto("/");
  await page.keyboard.press("Tab");
  await expect(page.getByLabel("Choose a photo")).toBeFocused();
  // The focus shows on the button-like label around the hidden file input.
  const ring = await page.getByText("Choose a photo").evaluate((el) => getComputedStyle(el).outlineStyle);
  expect(ring).toBe("solid");

  const chooser = page.waitForEvent("filechooser");
  await page.keyboard.press("Enter");
  await (await chooser).setFiles(photoPath(photo.file));
  expectCents((await shownEstimate(page)).price, photo.price);

  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "Price another photo" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByText("Add a product photo")).toBeVisible();
});
