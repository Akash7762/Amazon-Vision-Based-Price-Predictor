import { expect, type Page, type TestInfo } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const FIXTURES = path.join(__dirname, "..", "fixtures");

export type Photo = {
  file: string;
  product: string;
  listed_price: number;
  price: number;
  low: number;
  high: number;
};

// What the API should answer for each test photo, from the Phase 3 reference
// code (fixtures/make_expected.py).
export const expected: {
  model_version: string;
  model_sha256: string;
  range_coverage: number;
  photos: Photo[];
} = JSON.parse(fs.readFileSync(path.join(FIXTURES, "expected.json"), "utf8"));

export const photoPath = (file: string) => path.join(FIXTURES, "photos", file);
export const photoBytes = (file: string) => fs.readFileSync(photoPath(file));

export const isPhone = (testInfo: TestInfo) => !!testInfo.project.use.isMobile;

/** "$10.70" -> 10.7 */
export const dollars = (text: string) => Number(text.replace(/[$,]/g, ""));

/** Equal to the cent, give or take one: the model's float output can round differently on another CPU. */
export function expectCents(actual: number, wanted: number) {
  expect(Math.abs(actual - wanted), `$${actual} vs $${wanted}`).toBeLessThanOrEqual(0.015);
}

/** The app's error message. (Next.js also keeps an empty role="alert" route announcer, outside <main>.) */
export const errorMessage = (page: Page) => page.getByRole("main").getByRole("alert");

/** Picks a file with the "Choose a photo" button. */
export async function choose(page: Page, file: string | { name: string; mimeType: string; buffer: Buffer }) {
  await page.getByLabel("Choose a photo").setInputFiles(file);
}

/** The estimate on the result screen, as numbers. */
export async function shownEstimate(page: Page) {
  const price = page.locator('p:has-text("Estimated price") + p');
  await expect(price).toBeVisible();
  const explain = (await page.getByText(/^Likely between/).textContent()) ?? "";
  const range = explain.match(/between \$([\d.,]+) and \$([\d.,]+)/);
  expect(range, explain).not.toBeNull();
  return { price: dollars((await price.textContent()) ?? ""), low: dollars(range![1]), high: dollars(range![2]) };
}

export type Upload = { name: string; type: string; size: number; width: number; height: number };

/**
 * Keeps a note of every photo the page uploads (name, type, size, pixel size),
 * so a test can check what actually left the browser. Call before page.goto.
 */
export async function recordUploads(page: Page) {
  await page.addInitScript(() => {
    const uploads: object[] = [];
    Object.assign(window, { uploads });
    const realFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      const file = init?.body instanceof FormData ? init.body.get("file") : null;
      if (file instanceof File) {
        let width = 0;
        let height = 0;
        try {
          const bitmap = await createImageBitmap(file);
          ({ width, height } = bitmap);
          bitmap.close();
        } catch {
          // not an image the browser can read
        }
        uploads.push({ name: file.name, type: file.type, size: file.size, width, height });
      }
      return realFetch(input, init);
    };
  });
}

export const uploads = (page: Page) =>
  page.evaluate(() => (window as unknown as { uploads: Upload[] }).uploads);

/** A product photo blown up to `size` x `size` pixels, as a PNG: like a big photo straight off a phone. */
export async function bigPhoto(page: Page, file: string, size: number): Promise<Buffer> {
  const png = await page.evaluate(
    async ({ b64, size }) => {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const bitmap = await createImageBitmap(new Blob([bytes]));
      const canvas = new OffscreenCanvas(size, size);
      const ctx = canvas.getContext("2d")!;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(bitmap, 0, 0, size, size);
      const blob = await canvas.convertToBlob({ type: "image/png" });
      const out = new Uint8Array(await blob.arrayBuffer());
      let s = "";
      for (let i = 0; i < out.length; i += 0x8000) s += String.fromCharCode(...out.subarray(i, i + 0x8000));
      return btoa(s);
    },
    { b64: photoBytes(file).toString("base64"), size },
  );
  return Buffer.from(png, "base64");
}

/** A DataTransfer holding one photo, as dragging or pasting would carry it. */
export function transferJs(file: string) {
  return { b64: photoBytes(file).toString("base64"), name: file };
}
