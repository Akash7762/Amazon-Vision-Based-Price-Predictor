import type { NextRequest } from "next/server";
import { FORM_OVERHEAD_BYTES, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB } from "@/lib/limits";

// Uploads reach the API through the /api rewrite in next.config.ts. Next.js
// forwards at most `proxyClientMaxBodySize` of a request body but keeps the
// original Content-Length, so for a bigger upload the API waits for bytes that
// never arrive and then reads the next request on that connection as garbage:
// the next photo, possibly another user's, failed with a 500. Uploads the API
// would refuse anyway are refused here instead, before anything is forwarded.
export function proxy(request: NextRequest) {
  const length = Number(request.headers.get("content-length"));
  if (length > MAX_UPLOAD_BYTES + FORM_OVERHEAD_BYTES) {
    return Response.json({ detail: `File is larger than ${MAX_UPLOAD_MB} MB.` }, { status: 413 });
  }
}

export const config = {
  matcher: "/api/predict",
};
