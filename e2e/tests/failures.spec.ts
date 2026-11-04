// What the user sees when the network or the price service lets them down.
// Failures are simulated by intercepting the app's /api requests.
import { expect, test, type Route } from "@playwright/test";
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

test("a service that never answers: gives up after 90 seconds, with Try again", async ({ page }) => {
  await page.clock.install();
  await page.goto("/");
  await page.route("**/api/predict", () => {
    // never answers
  });
  await choose(page, photoPath(photo.file));
  const working = page.getByText("Estimating the price…");
  await expect(working).toBeVisible();

  await page.clock.fastForward(89_000);
  await expect(working).toBeVisible();
  await page.clock.fastForward(2_000);
  await expect(errorMessage(page)).toHaveText("That took too long. Check your connection and try again.");
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
});

/** A route handler that holds requests until `release()` is called: a server that's asleep. */
function sleepingServer() {
  let release = () => {};
  const awake = new Promise<void>((resolve) => (release = resolve));
  return { release, handler: async (route: Route) => (await awake, route.continue()) };
}

test("a sleeping service: says it's waking up, and the note goes once it answers", async ({ page }) => {
  const server = sleepingServer();
  await page.route("**/api/health", server.handler);
  await page.clock.install();
  await page.goto("/");
  const waking = page.getByRole("status").filter({ hasText: "Waking up the price service" });

  await page.clock.fastForward(3_000);
  await expect(waking).toBeHidden(); // a quick answer never shows it
  await page.clock.fastForward(2_000);
  await expect(waking).toBeVisible();

  server.release();
  await expect(waking).toBeHidden();
  await expect(page.getByRole("status")).toBeHidden();
});

test("a slow photo: a note that the service may have been asleep, then the price", async ({ page }) => {
  const server = sleepingServer();
  await page.clock.install();
  await page.goto("/");
  await page.route("**/api/predict", server.handler);
  await choose(page, photoPath(photo.file));
  const note = page.getByText("Still working. If the service was asleep");

  await page.clock.fastForward(7_000);
  await expect(note).toBeHidden();
  await page.clock.fastForward(2_000);
  await expect(note).toBeVisible();

  server.release();
  expectCents((await shownEstimate(page)).price, photo.price);
  await expect(note).toBeHidden();
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
