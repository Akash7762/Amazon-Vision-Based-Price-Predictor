"use client";

import { useEffect } from "react";

// Registers public/sw.js, which lets the installed app open without a
// connection. Production only: in `next dev` a service worker would serve stale
// files while you edit.
export function ServiceWorkerRegistration() {
  useEffect(() => {
    if (process.env.NODE_ENV !== "production" || !("serviceWorker" in navigator)) return;
    navigator.serviceWorker
      .register("/sw.js")
      .then(() => navigator.serviceWorker.ready)
      .then((registration) => {
        // On a first visit the page's scripts and styles loaded before the
        // worker existed, so it never saw them. Hand over their addresses so
        // the app can open offline from now on, not only from the next visit.
        const urls = performance
          .getEntriesByType("resource")
          .map((entry) => entry.name)
          .filter((url) => url.startsWith(`${location.origin}/_next/static/`));
        registration.active?.postMessage({ type: "cache-urls", urls });
      })
      .catch(() => {
        // Not fatal: the app works without offline support.
      });
  }, []);
  return null;
}
