// Talks to the backend through this app's /api/... route (see next.config.ts).
import { MAX_UPLOAD_BYTES, MAX_UPLOAD_MB } from "@/lib/limits";

export type Prediction = {
  price: number;
  currency: "USD";
  range: { low: number; high: number; coverage: number };
  model_version: string;
};

export type FailureKind = "unreadable" | "too_large" | "unsupported" | "offline" | "unavailable" | "timeout";

export class PredictionError extends Error {
  constructor(public kind: FailureKind, message: string) {
    super(message);
    this.name = "PredictionError";
  }
}

const MESSAGES: Record<FailureKind, string> = {
  unreadable: "We couldn't read that image. Try another photo.",
  too_large: "That photo is too large. Try a smaller one.",
  unsupported: "That file type isn't supported. Use a JPEG, PNG or WebP photo.",
  offline: "You're offline. Connect to the internet and try again.",
  unavailable: "The price service isn't responding right now. Try again in a moment.",
  timeout: "That took too long. Check your connection and try again.",
};

function kindForStatus(status: number): FailureKind {
  if (status === 413) return "too_large";
  if (status === 415) return "unsupported";
  if (status === 400 || status === 422) return "unreadable";
  return "unavailable"; // 5xx, or the backend not running behind the /api route
}

export async function predictPrice(image: Blob, filename: string, timeoutMs = 30_000): Promise<Prediction> {
  // Only a file the browser couldn't shrink gets here this big (lib/image.ts).
  // Say so now rather than after uploading 10 MB the API would refuse.
  if (image.size > MAX_UPLOAD_BYTES) {
    throw new PredictionError("too_large", `That file is larger than ${MAX_UPLOAD_MB} MB. Try a smaller photo.`);
  }

  const form = new FormData();
  form.append("file", image, filename);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let res: Response;
  try {
    res = await fetch("/api/predict", { method: "POST", body: form, signal: controller.signal });
  } catch {
    if (controller.signal.aborted) throw new PredictionError("timeout", MESSAGES.timeout);
    const kind = navigator.onLine ? "unavailable" : "offline";
    throw new PredictionError(kind, MESSAGES[kind]);
  } finally {
    clearTimeout(timer);
  }

  if (res.ok) return (await res.json()) as Prediction;

  const kind = kindForStatus(res.status);
  // The API's messages for upload problems are written for people; use them.
  let detail: string | undefined;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    // not JSON: the proxy's own error page, for example
  }
  throw new PredictionError(kind, kind !== "unavailable" && detail ? detail : MESSAGES[kind]);
}

export async function serviceIsUp(timeoutMs = 5_000): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch("/api/health", { signal: controller.signal, cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}
