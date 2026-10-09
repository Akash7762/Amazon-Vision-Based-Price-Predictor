import type { MetadataRoute } from "next";

// Served at /manifest.webmanifest. A valid manifest plus HTTPS (or localhost)
// is what lets browsers offer "Install app" / "Add to Home Screen".
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Vision Price Predictor",
    short_name: "Price Predictor",
    description: "Estimate a product's price from a photo.",
    start_url: "/",
    display: "standalone",
    background_color: "#ffffff",
    theme_color: "#0f766e",
    icons: [
      { src: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
      { src: "/icons/icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
