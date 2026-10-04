# Frontend (Phase 5)

A Next.js 16 web app that installs like a native app (a PWA) on phones and
desktops. Take or choose a product photo, and it shows the estimated price
with its range, from the [backend API](../backend/README.md).

## Run it

The backend has to be running first (see [`backend/README.md`](../backend/README.md)):

```bash
uvicorn backend.app.main:app            # from the repo root, in the Python venv
```

Then, in another terminal:

```bash
cd frontend
npm install                             # once
npm run dev                             # http://localhost:3000
```

`npm run dev` is for working on the code. For the real thing, including
offline support and installing, use the production build:

```bash
npm run build
npm start                               # http://localhost:3000
```

## How it talks to the backend

The browser only ever talks to this app. Requests to `/api/...` are forwarded
by Next.js to the backend (`next.config.ts`), so there's no CORS to set up,
and a phone or an HTTPS tunnel only needs to reach one address.

| Setting | Default | When it's read |
|---|---|---|
| `API_URL` | `http://localhost:8000` | when `npm run dev` starts, or at `npm run build` |

See [`.env.example`](.env.example). Put overrides in `.env.local` (not committed).

## What's in it

| File | What it does |
|---|---|
| `components/PricePredictor.tsx` | the screen: choose or take a photo, the result, errors |
| `lib/image.ts` | shrinks big photos to 1600 px JPEG before upload; small JPEG/PNG/WebP files go as they are |
| `lib/api.ts` | calls `/api/predict` and `/api/health`, and turns failures into plain messages |
| `app/manifest.ts` | the web app manifest (name, icons, colours) that makes it installable |
| `public/sw.js` | the service worker: the app opens offline; prices are never cached |
| `components/ServiceWorkerRegistration.tsx` | registers the worker (production builds only) |
| `scripts/make-icons.py` | draws the icons (`python frontend/scripts/make-icons.py`, needs Pillow) |

Behaviour worth knowing:

- **"Take a photo"** opens the back camera on phones and is hidden on
  desktops. On a desktop you can also drag a photo in, or paste one.
- **Errors** say what happened: not an image, wrong file type, the service
  unreachable, a timeout. "Try again" only appears when trying again can help.
- If the backend can't be reached when the page opens, a banner says so.
- The result explains the range ("about 8 in 10 products that got a similar
  estimate were priced in that range") and warns that multipacks and bulk
  sizes usually cost more than a photo suggests (Phase 3's main finding).

## Installing it

Browsers offer "Install" when the app has a valid manifest and is served
over **HTTPS**. `localhost` counts as secure.

**Desktop (Chrome or Edge):** run the production build, open
http://localhost:3000, and click the install icon at the right of the
address bar (or the browser menu → *Install Vision Price Predictor*). It opens
in its own window, with its own icon.

**Phone, on the same Wi-Fi, without installing:** run the production build
on the PC and open `http://<the PC's local IP>:3000` on the phone (`npm start`
prints a "Network" address; `ipconfig` shows it too). Windows may ask whether Node.js can use private
networks; allow it. This is plain HTTP, so the phone won't offer a real
install or offline support, but the camera, the upload and the layout can all
be checked.

**Phone, installed:** needs an HTTPS address. Deploying (Phase 7) gives one.
To try it sooner, an HTTPS tunnel to the PC such as Cloudflare's
`cloudflared` works: `cloudflared tunnel --url http://localhost:3000` prints an
`https://…trycloudflare.com` address to open on the phone. Then Chrome offers
*Install app* and Safari has *Share → Add to Home Screen*.

## Checked in Phase 5

In a Chromium browser, against the real model:

- a product photo gets the same price and range as calling the API directly
- a file that isn't an image gets the API's message and no pointless "Try again"
- a 4000×3000 image is shrunk and sent as JPEG; a small JPEG is sent untouched
- backend stopped: banner on load, a clear message, "Try again" recovers once
  it's back
- phone-sized screen with touch: "Take a photo" appears and asks for the back
  camera; the result fits a 375 px screen
- production build: manifest, all three icons (including maskable), active
  service worker, security headers
- offline: after one visit, with the server stopped, the app still opens
  fully styled and working, and says the price service is unreachable

Not yet automated: these checks are manual. End-to-end tests belong to
Phase 6.
