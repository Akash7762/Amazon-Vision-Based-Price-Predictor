// The API's upload limit (MAX_UPLOAD_MB in backend/app/config.py). The app
// checks it before uploading (lib/api.ts), and proxy.ts turns away bigger
// requests before they reach the API.
export const MAX_UPLOAD_MB = 10;
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

// Room for the multipart form's own headers around the file, as the API allows.
export const FORM_OVERHEAD_BYTES = 64 * 1024;
