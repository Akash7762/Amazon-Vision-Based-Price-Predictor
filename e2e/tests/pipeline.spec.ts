// The full pipeline: a real product photo goes from the browser through
// Next.js and the FastAPI backend to the model, and the price comes back to
// the screen. Every expected price comes from the Phase 3 reference code, so
// these tests also prove the app doesn't change the photo on the way.
import { expect, test, type Page } from "@playwright/test";
import {
  bigPhoto,
  choose,
  errorMessage,
  expectCents,
  expected,
  isPhone,
  photoBytes,
  photoPath,
  recordUploads,
  shownEstimate,
  transferJs,
  uploads,
} from "./helpers";

test("each test photo gets the reference price and range on screen", async ({ page }, testInfo) => {
  await recordUploads(page);
  await page.goto("/");

  for (const [i, photo] of expected.photos.entries()) {
    if (i > 0) await page.getByRole("button", { name: "Price another photo" }).click();
    const started = Date.now();
    await choose(page, photoPath(photo.file));
    const shown = await shownEstimate(page);
    testInfo.annotations.push({ type: "timing", description: `${photo.file}: ${Date.now() - started} ms to price on screen` });

    expectCents(shown.price, photo.price);
    expectCents(shown.low, photo.low);
    expectCents(shown.high, photo.high);
    await expect(page.getByText(`Model ${expected.model_version}`)).toBeVisible();
    await expect(page.getByAltText("The photo being priced")).toBeVisible();
  }

  // Small JPEGs leave the browser exactly as they are.
  expect(await uploads(page)).toEqual(
    expected.photos.map((p) => ({ name: p.file, type: "image/jpeg", size: photoBytes(p.file).length, width: 224, height: 224 })),
  );
});

test("a big photo is shrunk to a 1600 px JPEG before upload, and the price barely moves", async ({ page, request }, testInfo) => {
  await recordUploads(page);
  await page.goto("/");
  const png = await bigPhoto(page, expected.photos[1].file, 3000);
  expect(png.length).toBeLessThan(10 * 1024 * 1024);

  // The same photo sent whole, so the API does all the shrinking itself.
  const whole = await request.post("/api/predict", {
    multipart: { file: { name: "IMG_2041.png", mimeType: "image/png", buffer: png } },
  });
  const direct: number = (await whole.json()).price;

  const started = Date.now();
  await choose(page, { name: "IMG_2041.png", mimeType: "image/png", buffer: png });
  const shown = await shownEstimate(page);
  testInfo.annotations.push({
    type: "timing",
    description: `3000x3000 PNG (${(png.length / 1024 / 1024).toFixed(1)} MB): ${Date.now() - started} ms to price on screen`,
  });

  const [sent] = await uploads(page);
  expect(sent).toMatchObject({ name: "IMG_2041.jpg", type: "image/jpeg", width: 1600, height: 1600 });
  expect(sent.size).toBeLessThan(png.length / 4);
  const change = Math.abs(shown.price - direct) / direct;
  testInfo.annotations.push({
    type: "price change",
    description: `sent whole $${direct}, shrunk by the browser $${shown.price} (${(100 * change).toFixed(1)}%)`,
  });
  // 2% on the development PC. Blowing the photo up and back down again moves
  // it 13%, so 5% still catches a shrink that blurs or shifts colours.
  expect(change).toBeLessThan(0.05);
});

// Pasting and dropping only work once the page's script is running. A person
// can't be that quick, but a test can, so these try again until it is.
const startScreen = (page: Page) => page.getByText("Add a product photo");

test("a pasted photo is priced", async ({ page }) => {
  const photo = expected.photos[0];
  await page.goto("/");
  await expect(async () => {
    await page.evaluate(({ b64, name }) => {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const clipboard = new DataTransfer();
      clipboard.items.add(new File([bytes], name, { type: "image/jpeg" }));
      document.body.dispatchEvent(new ClipboardEvent("paste", { clipboardData: clipboard, bubbles: true }));
    }, transferJs(photo.file));
    await expect(startScreen(page)).toBeHidden({ timeout: 1000 });
  }).toPass();
  expectCents((await shownEstimate(page)).price, photo.price);
});

test("a photo dropped on the box is priced", async ({ page }, testInfo) => {
  test.skip(isPhone(testInfo), "phones have no drag and drop");
  const photo = expected.photos[2];
  await page.goto("/");
  const dropped = await page.evaluateHandle(({ b64, name }) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const data = new DataTransfer();
    data.items.add(new File([bytes], name, { type: "image/jpeg" }));
    return data;
  }, transferJs(photo.file));
  const box = startScreen(page).locator("..");
  await expect(async () => {
    await box.dispatchEvent("dragover", { dataTransfer: dropped });
    await box.dispatchEvent("drop", { dataTransfer: dropped });
    await expect(startScreen(page)).toBeHidden({ timeout: 1000 });
  }).toPass();
  expectCents((await shownEstimate(page)).price, photo.price);
});

test.describe("files the app can't price", () => {
  test("not an image: the API's own message, and no pointless Try again", async ({ page }) => {
    await page.goto("/");
    await choose(page, { name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("milk, eggs, coffee") });
    await expect(errorMessage(page)).toHaveText(
      "Unsupported file type 'text/plain'. Send a JPEG, PNG or WebP photo.",
    );
    await expect(page.getByRole("button", { name: "Try again" })).toHaveCount(0);
    await page.getByRole("button", { name: "Choose another photo" }).click();
    await expect(page.getByText("Add a product photo")).toBeVisible();
  });

  test("a broken JPEG: says it can't be read", async ({ page }) => {
    await page.goto("/");
    const junk = Buffer.alloc(50_000, 0x5a);
    await choose(page, { name: "photo.jpg", mimeType: "image/jpeg", buffer: junk });
    await expect(errorMessage(page)).toHaveText(
      "The file is not a readable image. Send a JPEG, PNG or WebP photo.",
    );
    await expect(page.getByRole("button", { name: "Try again" })).toHaveCount(0);
  });

  test("over 10 MB and unreadable by the browser: refused before uploading", async ({ page }) => {
    await recordUploads(page);
    await page.goto("/");
    await choose(page, { name: "IMG_0007.heic", mimeType: "image/heic", buffer: Buffer.alloc(11 * 1024 * 1024, 1) });
    await expect(errorMessage(page)).toHaveText("That file is larger than 10 MB. Try a smaller photo.");
    await expect(page.getByRole("button", { name: "Try again" })).toHaveCount(0);
    expect(await uploads(page)).toEqual([]);
  });
});

// The rest talk to the server directly; the browser doesn't matter, so once is enough.
test.describe("the server side, through the web app's /api", () => {
  test.beforeEach(({}, testInfo) => {
    test.skip(testInfo.project.name !== "desktop-chrome", "server behaviour, checked once");
  });

  const form = (name: string, buffer: Buffer) => ({ multipart: { file: { name, mimeType: "image/jpeg", buffer } } });

  test("an upload over the limit is refused without breaking the next one", async ({ request }) => {
    // Next.js used to pass on only the first 10 MB of such an upload, which left
    // the API's connection out of step: the next photo failed with a 500.
    const photo = expected.photos[0];
    for (const mb of [11, 30]) {
      const big = await request.post("/api/predict", form("huge.jpg", Buffer.alloc(mb * 1024 * 1024, 7)));
      expect(big.status()).toBe(413);
      expect(await big.json()).toEqual({ detail: "File is larger than 10 MB." });

      const next = await request.post("/api/predict", form(photo.file, photoBytes(photo.file)));
      expect(next.status()).toBe(200);
      expectCents((await next.json()).price, photo.price);
    }
  });

  test("a file just under 10 MB is passed on whole", async ({ request }) => {
    // With the form around it the request is just over 10 MB. Cut short there,
    // the API would wait for the rest and the request would hang.
    const almost = await request.post("/api/predict", form("photo.jpg", Buffer.alloc(10 * 1024 * 1024 - 50, 7)));
    expect(almost.status()).toBe(400);
    expect((await almost.json()).detail).toBe("The file is not a readable image. Send a JPEG, PNG or WebP photo.");
  });

  test("parallel uploads each get their own price", async ({ request }) => {
    const jobs = Array.from({ length: 12 }, (_, i) => expected.photos[i % expected.photos.length]);
    const answers = await Promise.all(jobs.map((p) => request.post("/api/predict", form(p.file, photoBytes(p.file)))));
    for (const [i, answer] of answers.entries()) {
      expect(answer.status()).toBe(200);
      expectCents((await answer.json()).price, jobs[i].price);
    }
  });

  test("timing: one photo at a time through the web app", async ({ request }, testInfo) => {
    const photo = expected.photos[1];
    const times: number[] = [];
    for (let i = 0; i < 20; i++) {
      const started = performance.now();
      const answer = await request.post("/api/predict", form(photo.file, photoBytes(photo.file)));
      expect(answer.status()).toBe(200);
      times.push(performance.now() - started);
    }
    times.sort((a, b) => a - b);
    const median = times[10];
    const p95 = times[18];
    testInfo.annotations.push({ type: "timing", description: `20 requests: median ${median.toFixed(0)} ms, 95th percentile ${p95.toFixed(0)} ms` });
    expect(median).toBeLessThan(2000);
  });
});
