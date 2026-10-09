# Phase 6 — Integration and end-to-end testing

The roadmap asks for three things: a full pipeline test (real image → model →
API → UI), a cross-platform checklist (mobile and desktop parity), and a
bug-fixing pass. All three are below, with the numbers from the runs that
produced them.

| | |
|---|---|
| Backend tests | **54**: 31 with a stand-in model, 23 with the real one |
| End-to-end tests | **34**, each in desktop Chrome, desktop Edge and an emulated Android phone: 86 runs, 16 skipped where they don't apply |
| Stability | every end-to-end test run 3 times over: **258 of 258 passed** |
| Accessibility | 0 WCAG 2.2 A/AA violations on 4 screens, light and dark |
| Bugs found and fixed | **6**, each with a test or a measurement showing the fix |

## How to run them

```bash
python -m pytest                 # repo root: backend and model tests
cd e2e && npm install            # once
npx playwright test              # end-to-end
npx playwright show-report       # the HTML report
```

The backend's real-model tests run when `backend/models/price_model.onnx` has
been downloaded and are skipped otherwise. The end-to-end tests start the API
and a production build of the web app themselves, on ports 8100 and 3100 with
their own build folder, so an app already running on 8000/3000 is left alone.
They use the Chrome and Edge installed on the PC; no browser download.

## 1. Full pipeline: real image → model → API → UI

**The photos.** Three product photos from the validation set (the model never
trained on them), at three price levels. Their expected answers come from the
Phase 3 reference code, `model/predict_onnx.py` and `model/price_range.py`,
not from the backend, so each test compares two separate routes to the same
number ([`e2e/fixtures/make_expected.py`](../e2e/fixtures/make_expected.py)).

| Photo | Listed price | Expected estimate | Expected range |
|---|---|---|---|
| French's Honey Dijon mustard, 12 oz | $3.24 | $3.23 | $1.83–$18.15 |
| Filippo Berio olive oil, 750 ml | $10.97 | $10.70 | $4.46–$29.43 |
| Rani ground garlic, 16 oz | $19.99 | $19.71 | $7.18–$44.88 |

These three were picked as clear photos at different prices, and the model
happens to price them well. Its typical error is the Phase 3 one: $11.27 on
average ([`evaluation.md`](evaluation.md)).

**API with the real model** ([`backend/tests/test_real_model.py`](../backend/tests/test_real_model.py)):

- each photo gets the reference price and range
- the same pixels get the same price as JPEG, PNG and WebP
- a sideways photo with an EXIF rotation tag (as phones save them) gets the
  upright photo's price
- a cut-out product on a transparent background gets the white-background price
- 16-bit greyscale gets the 8-bit price
- 15 unusual images (a 1×1 pixel, a 5000×20 strip, CMYK, 1-bit, palette,
  animated, progressive, a bad rotation tag, not a product at all...) each get
  a price and range inside the training range, never an error

Switching off rotation, transparency or 16-bit handling makes exactly the
matching test fail.

**In the browser** ([`e2e/tests/pipeline.spec.ts`](../e2e/tests/pipeline.spec.ts)):

- choosing each photo in turn shows the reference price, range and model
  version, and the small JPEGs leave the browser byte for byte as they are
- pasting a photo and dropping one on the box both work
- a 3000×3000 photo (6.3 MB) is shrunk to a 1600×1600 JPEG before upload. Its
  price is within 2.0% of sending it whole ($9.48 against $9.29)
- a text file, a broken JPEG and an 11 MB file the browser can't read each get
  a clear message and no pointless "Try again"
- 12 uploads at once each get their own photo's price

## 2. Cross-platform: mobile and desktop parity

Every end-to-end test runs in desktop Chrome, desktop Edge and Chrome
emulating a Pixel 7 phone (its screen size, touch input and user agent). The
same photo gets the same price in all three. On top of that
([`devices.spec.ts`](../e2e/tests/devices.spec.ts), [`pwa.spec.ts`](../e2e/tests/pwa.spec.ts), [`a11y.spec.ts`](../e2e/tests/a11y.spec.ts)):

| Check | Desktop | Phone |
|---|---|---|
| "Take a photo" opens the back camera (`capture="environment"`) | hidden | ✅ |
| Drop-or-paste hint | ✅ | hidden |
| Nothing scrolls sideways, start and result screens | ✅ | ✅ |
| Layout from 320 to 1920 px wide; photo and result side by side from 720 px | ✅ | — |
| Buttons at least 48 px tall | — | ✅ |
| Usable with the keyboard alone, focus visible | ✅ | — |
| Accessibility (axe-core, WCAG 2.2 A/AA), 4 screens, light and dark | ✅ 0 issues | ✅ 0 issues |
| Manifest and all three icons valid | ✅ | ✅ |
| Chrome reports nothing stopping an install¹ | ✅ | ✅ |
| After one visit, opens with the server switched off | ✅ | — |

¹ Apart from "in incognito": test browsers run like incognito windows, where
Chrome never offers to install.

**On real devices** (can't be automated: a real camera, real phone browsers,
the install prompt). Run the app with `start-app.bat` and fill in:

| # | Check | Result |
|---|---|---|
| D1 | Desktop Chrome or Edge: install from the address bar; the app opens in its own window and prices a photo | not run yet |
| D2 | Desktop: drag a photo from File Explorer onto the box | not run yet |
| P1 | Phone on the same Wi-Fi, `http://<PC's IP>:3000`: the page fits the screen | not run yet |
| P2 | Phone: "Take a photo" opens the back camera; a product photo gets a price | not run yet |
| P3 | Phone: "Choose a photo" picks from the gallery; the photo shows upright and gets a price | not run yet |
| P4 | Phone: a photo gets about the same price as the same file priced on the PC | not run yet |
| P5 | Phone: install to the home screen | needs HTTPS: Phase 7 |

## 3. When things go wrong

Simulated by intercepting the app's requests
([`failures.spec.ts`](../e2e/tests/failures.spec.ts)):

- service unreachable when the page opens: a banner; "Check again" clears it
- service stops mid-way: a plain message, and "Try again" works once it's back
- a 502 page or a server error: the same message, never the server's own text
- no answer at all: gives up after 30 seconds with "Try again" (a fake clock,
  so the test takes a second)
- offline: says so, and works again once back online

## Bugs found and fixed

| # | Bug | How it was found | Fix | Now |
|---|---|---|---|---|
| 1 | JPEGs that carry extra images (Pillow calls them MPO), such as the depth or HDR maps some phone cameras add, were refused: "Unsupported image format" | sending unusual but valid images to the real model | accept MPO; the main photo is the first image | backend test |
| 2 | 16-bit greyscale PNGs came out as a white square: the olive oil photo got $39.10, the blank-white price, instead of $11.24 | same | scale 16 bits down to 8 instead of clipping (`model/preprocessing.py`) | backend tests |
| 3 | Uploads over 10 MB: Next.js passed only the first 10 MB on to the API but kept the full length, so the API's connection got out of step. The next request through it failed with a 500, possibly someone else's. A file just under the limit hung until it timed out, and in the app an 11 MB file said "The price service isn't responding" | probing the size limits through the web app | `frontend/proxy.ts` refuses oversized uploads before forwarding them; Next.js passes on up to 11 MB; the app checks the size before uploading | 3 end-to-end tests, each seen to fail without the fix |
| 4 | Now and then a 500: Node.js (inside Next.js) and uvicorn both drop idle connections after 5 s, so a request could go down a connection the API was just closing. **8 of 432** requests failed in a stress test | a single "socket hang up" in one end-to-end run, then reproduced | uvicorn `--timeout-keep-alive 75`, longer than anything in front of it | **0 of 432** |
| 5 | The page's first service check could answer after an upload and overwrite it: the "service unreachable" banner disappeared after a failed upload, or could appear next to a price | 1 failure in 258 during a 3× repeat run | an upload's outcome now beats an older check | end-to-end test, 258/258 since |
| 6 | `/health` waited in the same queue as photo pricing: with 48 uploads running, median **592 ms**, 95th percentile 1.4 s, enough to time out the page's check on a busy server | looking into bug 5 | `/health` answers on the event loop | median **44 ms**, 95th 337 ms |

## What the testing showed about the model

None of these are bugs in the app; they're how the model behaves, worth
knowing for the report and the demo.

- **Orientation matters.** On its side, the mustard prices at $6.23 instead
  of $3.23 and the olive oil at $21.34 instead of $10.70 (the garlic hardly
  moves: $20.61 for $19.71). Phone photos carry an EXIF rotation tag and are
  turned upright first (tested), but a product lying on its side is priced as
  it lies. The start screen now says "One item, upright, in good light, works
  best."
- **Anything gets a price.** A blank white image: $39.10; blank black:
  $32.84; random noise: $17.42. Nothing tells a product from a non-product.
- **Fine detail matters, compression doesn't.** Blowing the olive oil photo
  up from 224 to 3000 px and back moves it from $10.70 to $9.35 (−13%).
  JPEG compression at photo size barely does: quality 92 down to 40 at
  1600 px moves the three test photos by at most 4.1%. (On a 224 px thumbnail
  quality 60 moves the olive oil +46%, but the app never sends thumbnails.)

## Timings

Development PC, CPU only, desktop Chrome, one test at a time:

| | |
|---|---|
| Small photo, from choosing it to the price on screen | 0.27–0.42 s |
| 3000×3000 photo (6.3 MB), including shrinking it in the browser | 1.7 s |
| API through the web app, 20 photos one after another | median 95 ms, 95th percentile 119 ms |

## Not covered

- Real phone browsers, the camera and the install prompt: the manual
  checklist above.
- Safari's engine (WebKit) and Firefox: Playwright can test them but needs its
  own browser builds downloaded (`npx playwright install webkit firefox`).
  The emulated phone is Chrome's engine.
