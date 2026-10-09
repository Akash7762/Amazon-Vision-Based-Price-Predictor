// Prepares a chosen photo for upload.
//
// Phone photos are often 3-12 MB, and the model only looks at 224x224 pixels,
// so large photos are shrunk to at most 1600 px on the long side and sent as
// JPEG: much faster on mobile data, and well under the API's 10 MB limit. It
// also converts formats the browser can read but the API doesn't accept, such
// as HEIC from iPhones (Safari can decode it).
//
// Small JPEG/PNG/WebP files are sent untouched; the API turns them upright and
// handles transparency itself. Anything the browser can't decode is sent as it
// is, and the API's answer explains the problem.

const MAX_SIDE = 1600;
const SEND_AS_IS = new Set(["image/jpeg", "image/png", "image/webp"]);
const SMALL_ENOUGH = 2 * 1024 * 1024;

export async function prepareImage(file: File): Promise<Blob> {
  let bitmap: ImageBitmap;
  try {
    // "from-image" applies the EXIF rotation, so the pixels drawn are upright.
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    return file;
  }

  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  if (scale === 1 && SEND_AS_IS.has(file.type) && file.size <= SMALL_ENOUGH) {
    bitmap.close();
    return file;
  }

  const width = Math.round(bitmap.width * scale);
  const height = Math.round(bitmap.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    bitmap.close();
    return file;
  }
  // White first: JPEG has no transparency, and the model learned from
  // white-background catalogue photos (the API does the same for PNGs).
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
  return blob ?? file;
}

export function uploadName(file: File, prepared: Blob): string {
  if (prepared === file) return file.name || "photo";
  const base = (file.name || "photo").replace(/\.[^.]*$/, "");
  return `${base}.jpg`;
}
