// Accessibility: axe-core checks every screen against WCAG 2.2 A and AA
// (contrast, labels, roles, ...), in light and dark mode.
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { choose, errorMessage, expected, photoPath, shownEstimate } from "./helpers";

const WCAG = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

async function expectNoViolations(page: Page) {
  const { violations, passes } = await new AxeBuilder({ page }).withTags(WCAG).analyze();
  const found = violations.map((v) => `${v.id} (${v.impact}): ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`);
  expect(found).toEqual([]);
  expect(passes.length).toBeGreaterThan(10); // it really checked the page
}

for (const colorScheme of ["light", "dark"] as const) {
  test.describe(`${colorScheme} mode`, () => {
    test.use({ colorScheme, serviceWorkers: "block" });

    test("start screen", async ({ page }) => {
      await page.goto("/");
      await expect(page.getByText("Choose a photo")).toBeVisible();
      await expectNoViolations(page);
    });

    test("result screen", async ({ page }) => {
      await page.goto("/");
      await choose(page, photoPath(expected.photos[0].file));
      await shownEstimate(page);
      await expectNoViolations(page);
    });

    test("error screen", async ({ page }) => {
      await page.goto("/");
      await choose(page, { name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("not a photo") });
      await expect(errorMessage(page)).toBeVisible();
      await expectNoViolations(page);
    });

    test("service-down banner", async ({ page }) => {
      await page.route("**/api/health", (route) => route.abort("connectionrefused"));
      await page.goto("/");
      await expect(page.getByRole("status")).toBeVisible();
      await expectNoViolations(page);
    });
  });
}
