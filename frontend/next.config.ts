import type { NextConfig } from "next";

// Where the FastAPI backend runs. The browser never calls it directly: it calls
// /api/... on this app and Next.js forwards the request. One address for the
// browser means no CORS setup, and it works the same on a laptop, on a phone
// over the local network, or through an HTTPS tunnel. Rewrites are fixed when
// the app is built (or when `next dev` starts), so set API_URL before that.
const API_URL = process.env.API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  // Don't let `next dev` write AI-agent instruction files into the project.
  agentRules: false,

  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
  },

  // From the Next.js PWA guide: basic hardening for every page, and a service
  // worker that is never cached, so an update reaches users on their next visit.
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
      {
        source: "/sw.js",
        headers: [
          { key: "Content-Type", value: "application/javascript; charset=utf-8" },
          { key: "Cache-Control", value: "no-cache, no-store, must-revalidate" },
          { key: "Content-Security-Policy", value: "default-src 'self'; script-src 'self'" },
        ],
      },
    ];
  },
};

export default nextConfig;
