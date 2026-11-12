# Contributing

How work on this project is done. Setup is in the README's
[Installation](README.md#installation) section.

## Branches

| Branch | What it is |
|---|---|
| `main` | released code. Render and Vercel deploy it, and each release is tagged (`v1.0`, `v1.1`, ...) |
| `dev` | where finished work is collected before a release |
| `feature/...`, `fix/...`, `docs/...` | one piece of work each, branched from `dev` |

1. Branch from `dev`: `git switch -c feature/short-name dev`.
2. Open a pull request into `dev`. Wait for the checks to go green before
   merging, and merge with **Create a merge commit**.
3. To release, open a pull request from `dev` into `main`, merge it the same
   way, and publish a GitHub release with a new tag on `main`. Never delete
   `dev`.

Merge commits rather than squashing keep `dev` and `main` sharing one
history, so later merges between them stay clean.

## Before a pull request

GitHub runs all of these on every pull request (the **Tests** workflow), but
they're quicker to fix locally:

```bash
python -m pytest
```

```bash
cd frontend && npm run lint && npx tsc --noEmit && npm run build
```

```bash
cd e2e && npx playwright test
```

Changes to `backend/`, `model/` or `deploy/` also get the **API container**
check: the Docker image is built and run with the free server's limits
(512 MB, a tenth of a CPU), and must still give the reference prices with 12
uploads arriving at once.

## Commits

One change per commit, with a short lowercase summary of what changed
("api: price photos one at a time on the free server"), and a body saying
why when it isn't obvious.

## Things to keep in step

- **Image preprocessing** (`model/preprocessing.py`) is shared by training
  and serving. A change there changes what the model sees: the real-model
  tests compare the API's prices with the evaluation code's.
- **A new model** gets a new `v0.x-model` release, with its sha256 pinned in
  `backend/download_model.py`, a new `backend/calibration.json`
  (`model/calibrate_interval.py`), and new expected prices
  (`python e2e/fixtures/make_expected.py`).
- **The upload limit** (10 MB) is in `backend/app/config.py` and
  `frontend/lib/limits.ts`.
- **The README's images** come from the running app:
  `cd e2e && npx playwright test -c media.config.ts`, then
  `python scripts/make_demo_gif.py`.

## Numbers in the docs

Every figure in the docs comes from a run that was actually executed: a test
score, a measurement, a log. If a change moves one, update it from a new
run, never by estimate.
