// What the user sees when the network or the price service lets them down.
// Failures are simulated by intercepting the app's /api requests.
import { expect, test } from "@playwright/test";
import { choose, errorMessage, expectCents, expected, photoPath, shownEstimate } from "./helpers";

const photo = expected.photos[0];
const serviceDown = "The price service isn't responding right now. Try again in a moment.";

// A service worker could answer requests before the interception sees them.
test.use({ serviceWorkers: "block" });

test.afterEach(async ({ page }) => {
  await page.unrouteAll({ behavior: "ignoreErrors" });
});

test("service unreachable when the page opens: a banner, gone after Check again", async ({ page }) => {
  await page.route("**/api/health", (route) => route.abort("connectionrefused"));
  await page.goto("/");
  const banner = page.getByRole("status").filter({ hasText: "isn't reachable" });
  await expect(banner).toBeVisible();

  await page.unroute("**/api/health");
  await banner.getByRole("button", { name: "Check again" }).click();
  await expect(banner).toBeHidden();
});

test("service stops mid-way: a clear message, and Try again works once it's back", async ({ page }) => {
  await page.goto("/");
  await page.route("**/api/predict", (route) => route.abort("connectionrefused"));
  await choose(page, photoPath(photo.file));
  await expect(errorMessage(page)).toHaveText(serviceDown);
  await expect(page.getByRole("status")).toContainText("isn't reachable");

  await page.unroute("**/api/predict");
  await page.getByRole("button", { name: "Try again" }).click();
  expectCents((await shownEstimate(page)).price, photo.price);
  await expect(page.getByRole("status")).toBeHidden();
});

test("the web app can't reach the API (a 502 page): same message, and a retry", async ({ page }) => {
  await page.goto("/");
  await page.route("**/api/predict", (route) =>
    route.fulfill({ status: 502, contentType: "text/html", body: "<h1>502 Bad Gateway</h1>" }),
  );
  await choose(page, photoPath(photo.file));
  await expect(errorMessage(page)).toHaveText(serviceDown);
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
});

test("a server error: same message, not the server's text", async ({ page }) => {
  await page.goto("/");
  await page.route("**/api/predict", (route) =>
    route.fulfill({ status: 500, contentType: "application/json", body: '{"detail":"Traceback (most recent call last)"}' }),
  );
  await choose(page, photoPath(photo.file));
  await expect(errorMessage(page)).toHaveText(serviceDown);
});

test("a service that never answers: gives up after 30 seconds, with Try again", async ({ page }) => {
  await page.clock.install();
  await page.goto("/");
  await page.route("**/api/predict", () => {
    // never answers
  });
  await choose(page, photoPath(photo.file));
  const working = page.getByText("Estimating the price…");
  await expect(working).toBeVisible();

  await page.clock.fastForward(29_000);
  await expect(working).toBeVisible();
  await page.clock.fastForward(2_000);
  await expect(errorMessage(page)).toHaveText("That took too long. Check your connection and try again.");
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
});

test("offline: says so, and works again once back online", async ({ page, context }) => {
  await page.goto("/");
  await context.setOffline(true);
  await choose(page, photoPath(photo.file));
  await expect(errorMessage(page)).toHaveText("You're offline. Connect to the internet and try again.");

  await context.setOffline(false);
  await page.getByRole("button", { name: "Try again" }).click();
  expectCents((await shownEstimate(page)).price, photo.price);
});
