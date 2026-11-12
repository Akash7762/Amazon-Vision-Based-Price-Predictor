# Demo script (10 minutes)

Eight slides and a live demo, timed to finish at about 9:30, which leaves
half a minute of slack. The speaker notes on each slide repeat the key line.

## Ten minutes before

| | Check |
|---|---|
| 1 | Open https://amazon-vision-based-price-predictor.vercel.app on the laptop and price one photo. The free server sleeps after 15 minutes without visitors and takes up to a minute to wake: wake it now, not during the talk. |
| 2 | Keep that tab open, so it stays awake (open it again if the talk starts more than 15 minutes later). |
| 3 | Have 2–3 product photos ready on the laptop. The test photos in `e2e/fixtures/photos/` are safe choices: mustard, olive oil, garlic. |
| 4 | Phone: the installed app open, one real product nearby to photograph (upright, in good light). |
| 5 | **Backup:** double-click `start-app.bat` on the laptop, so a local copy runs at http://localhost:3000. It needs no internet. |
| 6 | Slides open in present mode on slide 1. |

## Timeline

| Time | Slide | What to say |
|---|---|---|
| 0:00–0:30 | 1. Cover | Name, project, one sentence: "a product's price from a photo alone." |
| 0:30–1:30 | 2. The problem | Why it's hard: a photo often doesn't show pack size. The goal is an honest measurement plus a usable app. |
| 1:30–2:30 | 3. How it was built | Data → model → evaluation → serving → app. One controlled experiment: log(price) didn't beat raw price, judged by a rule set before the run. |
| 2:30–4:00 | 4. Results | $11.27 vs $14.51 baseline (22% lower). The range held 79.6% against an 80% target. Test set used once, for no decision. |
| 4:00–6:30 | 5. Live demo | Below. |
| 6:30–8:00 | 6. Testing and deployment | 57 + 37 tests, 7 bugs found and fixed. Tell the memory-crash story (one minute). |
| 8:00–9:00 | 7. Limits and future work | Say the limits before you're asked. |
| 9:00–9:30 | 8. Thank you | Leave it up during questions. |

## The live demo (2.5 minutes)

1. **Desktop:** choose the olive oil photo. It should show **$10.70, likely
   between $4.46 and $29.43** (its listed price is $10.97). Read the
   explanation line out loud: about 8 in 10 products with a similar estimate
   were priced in that range.
2. **Phone:** take a photo of the real product with the installed app. Say
   that it runs in a phone browser, installed from the website, with no app
   store.
3. **If there's time:** drop a non-image file on the desktop app to show a
   clear error message with no pointless "Try again".

While a photo is being priced (about a second), say what is happening: the
browser shrinks the photo, Vercel passes it to the API on Render, and the
model runs on a tenth of a CPU.

## If something goes wrong

| Problem | What to do |
|---|---|
| "Waking up the price service…" | Expected on free hosting. Say so, keep talking; it takes up to a minute. |
| No internet | Switch to http://localhost:3000 (the backup from step 5). Same model, same prices. |
| Phone can't be shown on the projector | Show the phone screenshot on slide 5 instead. |
| Everything fails | Stay on slide 5: its screenshots are real results from the app. |
