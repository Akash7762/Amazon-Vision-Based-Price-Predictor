# Deployment (Phase 7)

**Live:** https://amazon-vision-based-price-predictor.vercel.app

The app runs in two places, both on free plans, both rebuilt automatically
when `main` changes:

```
phone or PC ──HTTPS──▶ Vercel: the web app (Next.js)  ──HTTPS──▶ Render: the price API (Docker)
                       amazon-vision-based-             /api/*    amazon-vision-based-price-
                       price-predictor.vercel.app                 predictor-api.onrender.com
```

The browser only ever talks to Vercel. Vercel forwards `/api/...` to the API
on Render, exactly as `next.config.ts` does on a laptop, so there's no CORS to
set up, and HTTPS on both ends means the app can be installed on a phone.

| | Render: the API | Vercel: the web app |
|---|---|---|
| Plan | Free | Hobby (free; personal, non-commercial use) |
| Built from | [`api.Dockerfile`](api.Dockerfile) | [`frontend/`](../frontend/README.md) |
| Limits | 512 MB of memory, a tenth of a CPU; sleeps after 15 minutes without visitors and takes up to a minute to wake | requests up to 4.5 MB; up to 2 minutes for each one it forwards to the API |

## Checked on the live site

| | |
|---|---|
| [`smoke_test.py`](smoke_test.py), on Render and through Vercel | the released model (sha256), the three test photos at their exact reference prices, a text file refused |
| One photo, through Vercel | about 1.0 s for the API (median of 20), 1.4 to 2.4 s from choosing it to the price on screen |
| 12 photo-sized uploads at once | all 12 answered, the last after about 13 s (they're priced one at a time); before the fix below, this crashed the server |
| The end-to-end tests, run against the live site | 92 passed in desktop Chrome, desktop Edge and an emulated Android phone; 19 skipped where they don't apply |
| A real phone and PC | photo from the camera and from the gallery priced, the app installed on the phone and on the PC ([checklist](../docs/testing.md#2-cross-platform-mobile-and-desktop-parity)) |

## What's here

| File | What it does |
|---|---|
| [`api.Dockerfile`](api.Dockerfile) | the API image: Python 3.12, the backend, and the model, downloaded from the `v0.1-model` release and checked against its sha256 while the image is built; settings for the free plan (below) |
| [`smoke_test.py`](smoke_test.py) | checks a running API: it answers, it's the released model, the three test photos get their reference prices, a text file is refused; optionally a burst of uploads at once (`--burst`) and extra photos (`--also`) |
| [`../.github/workflows/api-container.yml`](../.github/workflows/api-container.yml) | on every change to the API, GitHub builds the image, runs it with the free plan's limits (512 MB, 0.1 CPU) and runs the smoke test against it, with 12 photo-sized uploads at once and a 16 MP image |

## Found on the live site, and fixed

**12 photos at once crashed the API.** The first end-to-end run against the
live site failed partway through, and 12 uploads of 1600 px photos at once
reproduced it: 4 answered, then 502 errors and the API was down for over 30
seconds. Render's events said why: "Ran out of memory (used over 512MB)",
three times. Every photo being priced holds its decoded image and the model's
working memory, and they all ran at once.

The image now sets, for the free plan:

| Setting | Why |
|---|---|
| `MAX_CONCURRENT_PREDICTIONS=1` | photos are priced one at a time and the rest wait their turn. A tenth of a CPU can't price two at once any faster anyway |
| `MALLOC_ARENA_MAX=2` | glibc otherwise keeps a memory pool per thread and holds on to memory that has been freed |
| `MAX_IMAGE_PIXELS=16000000` | one 48 MP image would fill the memory by itself; photos from the web app are 1600 px at most (2.6 MP) |
| `ORT_THREADS=1` | one inference thread uses the tenth of a CPU fully; more would only wait for it |

After the fix, the same 12 photos at once all got their prices and the API
stayed up. GitHub's container check didn't reproduce the crash, even under the
same 512 MB limit and the same load. The likely reason is that Docker frees
cached file data (such as the 112 MB model file it has read) before giving up,
while Render seems to count some of it as used. The check keeps sending that
load, so a change that makes things much worse would still show.

**Vercel refuses requests over 4.5 MB** with a plain "Request Entity Too
Large", before the app sees them. Photos the browser can read are shrunk to a
1600 px JPEG first, well under that. A file it can't read (an iPhone HEIC on a
Windows PC, say) between 4.5 and 10 MB gets "That photo is too large. Try a
smaller one."; over 10 MB the app refuses before uploading.

## First-time setup

Everything is done in a web browser; nothing to install. Render and Vercel
deploy from `main`.

### 1. Render: the API

1. Open https://dashboard.render.com and sign up with **GitHub**.
2. Click **New +** → **Web Service**.
3. Pick the repository **Amazon-Vision-Based-Price-Predictor**. If it isn't
   listed, use the link to configure Render's GitHub access and allow it.
4. Fill in the form:

   | Field | Value |
   |---|---|
   | Name | it becomes the address, e.g. `amazon-vision-based-price-predictor-api` |
   | Language | **Docker** |
   | Branch | `main` |
   | Region | **Singapore** (the closest to India) |
   | Root Directory | leave empty |
   | Instance Type | **Free** |
   | Dockerfile Path (under Advanced) | `./deploy/api.Dockerfile` |
   | Docker Build Context Directory | `.` |
   | Health Check Path | `/health` |
   | Auto-Deploy | **On Commit** |

   No environment variables are needed: the image sets its own. If the
   Dockerfile Path is left at its default, the build fails at once with
   `open Dockerfile: no such file or directory`; set it under **Settings** →
   **Build & Deploy** and choose **Manual Deploy** → **Deploy latest commit**.
5. Click **Deploy Web Service**. The build takes 2 to 5 minutes. It's ready
   when the log shows `Loaded price_model.onnx (v0.1-model, sha256 b7055048f0f0...)`
   and `Your service is live`.
6. Open `https://<your-api>.onrender.com/health`. It should show
   `{"status":"ok","model_version":"v0.1-model",...}`. Copy the address
   (without `/health`) for the next part.

### 2. Vercel: the web app

1. Open https://vercel.com/signup, choose **Hobby**, and continue with **GitHub**.
2. Click **Add New…** → **Project**, and **Import** the repository (if it
   isn't listed, adjust Vercel's GitHub access to include it).
3. On the configure screen:

   | Field | Value |
   |---|---|
   | Project Name | it becomes the address |
   | Framework Preset | Next.js (detected) |
   | Root Directory | click **Edit**, choose `frontend` |
   | Environment Variables | `API_URL` = the Render address from step 1.6 (no `/` at the end) |

   Without `API_URL` the build stops with a message saying so, rather than
   building a site that sends photos to "localhost".
4. Click **Deploy** (1 to 2 minutes).
5. Check the production branch is `main` (the project overview says "To
   update your Production Deployment, push to the main branch"); if not,
   set it in **Settings**.

### 3. Check it

From the repo root:

```bash
python deploy/smoke_test.py https://amazon-vision-based-price-predictor-api.onrender.com --wait 120
```

```bash
python deploy/smoke_test.py https://amazon-vision-based-price-predictor.vercel.app/api --wait 120
```

The first checks the API on its own, the second the whole route through
Vercel. Then the end-to-end tests against the live site (PowerShell; in Git
Bash put `E2E_BASE_URL=... ` in front of the command instead):

```bash
cd e2e; $env:E2E_BASE_URL = "https://amazon-vision-based-price-predictor.vercel.app"; npx playwright test --workers=2
```

They wake the API first, then run everything that applies to a deployed site.

## Updating

Merge into `main`: Render rebuilds the API (about 3 minutes) and Vercel the
web app (about a minute). Pull requests also get a check from the
**API container** workflow, so a broken image is caught before it's merged.
Vercel also makes a preview deployment for each pull request.

## Living with the free plans

- **Sleeping.** After 15 minutes without visitors Render stops the API; the
  next visit wakes it, which takes up to a minute. The app says "Waking up the
  price service" meanwhile, and waits up to 90 seconds for a photo. Before a
  demo, open the app a minute early.
- **Hours.** Render's free plan includes 750 hours a month per account, enough
  for one service running all month.
- **One photo at a time.** Each takes about a second; when several arrive
  together the later ones wait (12 at once: the last after about 13 s).
- **No login and no rate limit.** Anyone with the link can use it. Uploads
  are capped at 4.5 MB by Vercel (10 MB on the API itself) and images at 16
  million pixels.
