"""
Checking and opening uploaded photos.

Every rejection carries the HTTP status the API returns for it:

    413  file bigger than MAX_UPLOAD_MB, or more pixels than MAX_IMAGE_PIXELS
    415  not a JPEG, PNG or WebP (by the declared type, or by what the bytes are)
    400  empty, or claims to be an image but can't be decoded (corrupt, truncated)

The pixel count is checked from the image header, before decoding, so a small
file that expands into a huge image is refused without using the memory.
The declared content type is only a first filter; what counts is what the
bytes actually decode as.
"""
import io

from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")
# Some clients send a generic type or "image/jpg"; the decoded format is checked anyway.
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "application/octet-stream"}
SEND_THIS = "Send a JPEG, PNG or WebP photo."


class UploadError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def check_declared_type(content_type):
    base = (content_type or "").split(";")[0].strip().lower()
    if base and base not in ALLOWED_TYPES:
        raise UploadError(415, f"Unsupported file type '{base}'. {SEND_THIS}")


def read_upload(fileobj, max_bytes):
    data = fileobj.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise UploadError(413, f"File is larger than {max_bytes / 1024 / 1024:g} MB.")
    if not data:
        raise UploadError(400, "The file is empty.")
    return data


def open_image(data, max_pixels):
    try:
        img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise UploadError(413, f"Image has more than {max_pixels:,} pixels.")
    except (UnidentifiedImageError, OSError):
        raise UploadError(400, f"The file is not a readable image. {SEND_THIS}")

    if img.format not in ALLOWED_FORMATS:
        raise UploadError(415, f"Unsupported image format '{img.format}'. {SEND_THIS}")
    width, height = img.size
    if width * height > max_pixels:
        raise UploadError(413, f"Image is {width}x{height} pixels; the limit is {max_pixels:,}.")
    try:
        img.load()
    except (OSError, ValueError, SyntaxError):
        raise UploadError(400, "The image is corrupt or incomplete.")
    return img
