# Likely viva questions, with honest answers

Short answers you can say in a sentence or two, each with the evidence
behind it. Where the honest answer is "it doesn't", say so: the evidence
for the limits is as solid as for the results.

## The model

**Why ConvNeXt-Tiny?**
A modern CNN pretrained on ImageNet-22k, small enough (28M parameters) to
fine-tune on Kaggle's free T4 GPU in about 4 hours, and fast enough to serve
on a CPU. A much bigger model wouldn't fit the free compute.

**Why regression, not classification into price bands?**
Price is a number, and a range on it is more useful than a band. The
calibrated range gives the "band" feeling while keeping the estimate.

**Why SmoothL1 loss?**
It behaves like squared error for small mistakes and like absolute error for
big ones, so a few very expensive or mislabelled products don't dominate
training.

**Why not train on log(price)?**
It was tried (run2). It did worse: $11.50 against $11.23 on validation, a
difference of +$0.27 with a 95% interval of +$0.13 to +$0.40. The decision
rule was written down before the run. ([`docs/experiments.md`](../experiments.md))

**Is the model overfitting?**
Partly. Validation error stopped improving after about epoch 4 while training
error kept falling, so the later epochs mostly memorised the training
photos. More regularisation is in the future work.

**Why letterbox the images instead of cropping or stretching?**
Amazon photos sit on white backgrounds, so white padding blends in, and the
product keeps its shape (long and thin vs square), which says something about
the category.

## The results

**Is $11.27 average error good?**
It's 22% better than always guessing the median ($14.51), and the median
error is $5.74. It's a clear but modest improvement: a photo alone carries
limited price information. That's the honest finding.

**How do you know the test score isn't inflated?**
The test set was used once and for no decision. Validation predicted $11.23
and the test gave $11.27. Removing all 565 test items whose photo also
appears in training gives $11.23, so duplicates don't inflate it.
([`docs/evaluation.md`](../evaluation.md))

**What does the range mean?**
About 8 in 10 products that got a similar estimate were priced inside it. It
was fitted on validation by price level (split-conformal calibration) and
then held 79.6% of real test prices against an 80% target.

**Why are the ranges so wide?**
The median width is about $31. That's what the photo can honestly support;
a narrower range would claim more certainty than the model has.

## What it gets wrong

**What are its biggest mistakes?**
It hedges toward typical prices: items under $5 are guessed $5.67 too high
on average, items over $50 $40.33 too low. And it can't see pack size: the
worst misses are cases and bulk packs photographed as one unit.

**Any evidence for the pack-size explanation?**
Yes: products sold in packs are guessed 4.5% lower than single items at the
same price. One brand (2,952 products) whose variants differ only in small
label text has a $21.25 error against $10.84 for everything else.

**Are there errors in the data?**
Some of the worst "over-predictions" look like label errors, for example a
24-pack of water listed at $1.79. They can't be checked against live
listings, so they're reported as likely, not certain.

**What happens with a photo that isn't a product?**
It still gets a price; a blank white image gets $39.10. Nothing detects "not
a product". That's a known limit, and future work.

**Does orientation matter?**
Yes: on its side, a product can get twice its upright price. Phone photos'
rotation tags are applied, and the app asks for upright photos.

## The system

**Why ONNX instead of serving PyTorch?**
The server doesn't need PyTorch (a much smaller install), and the exported
model matches PyTorch within $0.0002 on 256 test images.

**How do you make sure the app prices photos the same way as in training?**
One preprocessing module (`model/preprocessing.py`) is shared by training and
serving, and 23 tests compare the API's prices with the evaluation code's.

**How was it tested?**
57 backend tests and 37 end-to-end tests in Chrome, Edge and an emulated
phone, run by GitHub Actions on every pull request, plus checks on a real
phone and PC. The testing found 7 real bugs, all fixed.
([`docs/testing.md`](../testing.md))

**Tell us about one bug.**
On the live site, 12 photos at once crashed the free server: Render's events
said "Ran out of memory (used over 512MB)". Each photo being priced held its
decoded image and the model's working memory. The fix prices photos one at a
time; afterwards 12 at once all got their prices.

**Why free hosting, and what does it cost you?**
It costs nothing for a student project. The trade-offs: the server sleeps
after 15 minutes (up to a minute to wake) and prices photos one at a time,
about a second each.

**Is it secure?**
Uploads are limited in size and pixels and checked by content, not by name,
and the model file is checked by sha256 at startup. There's no login or rate
limit: it's a public demo.

## Future work

**What would you do next?**
1. Use the product's text (title, pack size) together with the photo: the
   biggest errors come from what the photo doesn't show.
2. Train with rotation augmentation, so orientation matters less.
3. Add a "not a product" check before pricing.
4. Regularise more or stop earlier, since validation stopped improving
   around epoch 4.
5. Paid hosting: no sleeping, and several photos priced at once.
