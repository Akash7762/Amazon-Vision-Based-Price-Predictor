# Deployment (Phase 7)

The app runs in two places, both on free plans, both rebuilt automatically
when `main` changes:

```
phone or PC ──HTTPS──▶ Vercel: the web app (Next.js)  ──HTTPS──▶ Render: the price API (Docker)
                       https://<app>.vercel.app          /api/*    https://<api>.onrender.com
```

The browser only ever talks to Vercel. Vercel forwards `/api/...` to the API
on Render, exactly as `next.config.ts` does on a laptop, so there's no CORS to
set up, and HTTPS on both ends means the app can be installed on a phone.

| | Render: the API | Vercel: the web app |
|---|---|---|
| Plan | Free | Hobby (free; personal, non-commercial use) |
| Built from | [`api.Dockerfile`](api.Dockerfile) | [`frontend/`](../frontend/README.md) |
| Limits | 512 MB of memory, a tenth of a CPU; sleeps after 15 minutes without visitors and takes up to a minute to wake | up to 2 minutes for each request it forwards to the API |
| Measured | about 200 MB in use, 266 MB at most (8 photos at once); about 1.3 s of work per photo | |

## What's here

| File | What it does |
|---|---|
| [`api.Dockerfile`](api.Dockerfile) | the API image: Python 3.12, the backend, and the model, downloaded from the `v0.1-model` release and checked against its sha256 while the image is built |
| [`smoke_test.py`](smoke_test.py) | checks a running API: it answers, it's the released model, the three test photos get their reference prices, a text file is refused |
| [`../.github/workflows/api-container.yml`](../.github/workflows/api-container.yml) | on every change to the API, GitHub builds the image, runs it with the free plan's limits (512 MB, 0.1 CPU) and runs the smoke test against it |

## First-time setup

Everything is done in a web browser; nothing to install. The steps use
`main`, so merge the deployment branch into `dev` and `dev` into `main` first.

### 1. Render: the API

1. Open https://dashboard.render.com and sign up with **GitHub**.
2. Click **New +** → **Web Service**.
3. Pick the repository **Amazon-Vision-Based-Price-Predictor**. If it isn't
   listed, use the link to configure Render's GitHub access and allow it.
4. Fill in the form:

   | Field | Value |
   |---|---|
   | Name | `vision-price-api` (it becomes the address; Render adds a suffix if it's taken) |
   | Language | **Docker** |
   | Branch | `main` |
   | Region | **Singapore** (the closest to India) |
   | Root Directory | leave empty |
   | Instance Type | **Free** |
   | Dockerfile Path (under Advanced) | `deploy/api.Dockerfile` |
   | Docker Build Context Directory | `.` |
   | Health Check Path | `/health` |
   | Auto-Deploy | **On Commit** |

   No environment variables are needed: the image sets its own.
5. Click **Deploy Web Service**. The first build takes 5 to 10 minutes. It's
   ready when the log shows `Loaded price_model.onnx (v0.1-model, sha256 b7055048f0f0...)`
   and Render marks the service **Live**.
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
   | Project Name | `vision-price-predictor` (the address; Vercel adds a suffix if it's taken) |
   | Framework Preset | Next.js (detected) |
   | Root Directory | click **Edit**, choose `frontend` |
   | Environment Variables | `API_URL` = the Render address from step 1.6, e.g. `https://vision-price-api.onrender.com` (no `/` at the end) |

   Without `API_URL` the build stops with a message saying so, rather than
   building a site that sends photos to "localhost".
4. Click **Deploy** (1 to 2 minutes).
5. In the project's **Settings**, find the **production branch** (under
   **Git** or **Environments**) and set it to `main`. Vercel starts with the
   repository's default branch, which is `dev`.

### 3. Check it

From the repo root on your PC:

```bash
python deploy/smoke_test.py https://<your-api>.onrender.com --wait 120
```

```bash
python deploy/smoke_test.py https://<your-app>.vercel.app/api --wait 120
```

The first checks the API on its own, the second the whole route through
Vercel. Then run the end-to-end tests against the live site:

```bash
cd e2e
$env:E2E_BASE_URL = "https://<your-app>.vercel.app"; npx playwright test
```

(PowerShell syntax; in Git Bash, `E2E_BASE_URL=https://<your-app>.vercel.app npx playwright test`.)
They wake the API first, then run everything except the test that switches
its own server off.

## Updating

Merge into `main`: Render rebuilds the API (about 5 minutes) and Vercel the
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
- **Speed.** A tenth of a CPU makes each photo about 1.3 s of work, and
  several photos at once wait for each other.
- **No login and no rate limit.** Anyone with the link can use it. Uploads
  are capped at 10 MB and 50 million pixels.
