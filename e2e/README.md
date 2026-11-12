# End-to-end tests (Phase 6)

Playwright tests that run the whole app, the FastAPI backend with the real
model and a production build of the web app, and use it the way a person
would, in desktop Chrome, desktop Edge and an emulated Android phone.

```bash
cd e2e
npm install          # once
npx playwright test  # builds and starts the app itself, on ports 8100 and 3100
npx playwright show-report
```

Needs the project venv and the downloaded model (`python backend/download_model.py`).
What's covered, the bugs these tests found and the results are in
[`docs/testing.md`](../docs/testing.md).

Against the deployed site instead (nothing is started locally; see
[`deploy/README.md`](../deploy/README.md)):

```bash
$env:E2E_BASE_URL = "https://amazon-vision-based-price-predictor.vercel.app"; npx playwright test --workers=2
```

The README's screenshots and demo GIF come from the same running app
([`media/screenshots.spec.ts`](media/screenshots.spec.ts)):

```bash
npx playwright test -c media.config.ts
```

```bash
cd .. && python scripts/make_demo_gif.py
```
