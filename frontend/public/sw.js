// Service worker: lets the installed app open offline. It never touches /api,
// so a price always comes from the model, never from a cache.
//
// Change CACHE to drop old caches when these rules change.
const CACHE = "price-predictor-v1";
const SHELL = ["/", "/manifest.webmanifest", "/icons/icon-192.png", "/icons/icon-512.png"];

self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

// The page sends the build files it loaded before this worker existed (see
// components/ServiceWorkerRegistration.tsx). Only this site's own /_next/static
// files are accepted.
self.addEventListener("message", (event) => {
  if (event.data?.type !== "cache-urls" || !Array.isArray(event.data.urls)) return;
  const urls = event.data.urls.filter((u) => {
    try {
      const url = new URL(u);
      return url.origin === self.location.origin && url.pathname.startsWith("/_next/static/");
    } catch {
      return false;
    }
  });
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(urls)));
});

function store(request, response) {
  if (response.ok) {
    const copy = response.clone();
    caches.open(CACHE).then((cache) => cache.put(request, copy));
  }
  return response;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/")) return; // prices: always the network

  // Build files have content hashes in their names, so a cached copy is never stale.
  if (url.pathname.startsWith("/_next/static/") || url.pathname.startsWith("/icons/")) {
    event.respondWith(
      caches.match(request).then((hit) => hit || fetch(request).then((res) => store(request, res))),
    );
    return;
  }

  // Pages: network first so updates show up, the saved copy when offline.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((res) => store(request, res))
        .catch(() => caches.match(request).then((hit) => hit || caches.match("/"))),
    );
  }
});
