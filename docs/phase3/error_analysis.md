# Error analysis: run1 on run1_test_predictions.csv

11028 images. Errors in dollars unless stated.

| | value |
|---|---|
| MAE | $11.268 |
| median absolute error | $5.737 |
| SMAPE | 53.99% |
| bias (mean of predicted - actual) | $-4.273 |
| predictions below $0 | 0 |
| predictions below $1 | 8 |
| products sold as packs of 2+ | 3582 (32.5%) |

## Predicted vs actual, by price level

![calibration](calibration.png)

| actual price | n | median actual | median predicted | MAE |
|---|---|---|---|---|
| $1-2 | 296 | $1.74 | $5.59 | $6.73 |
| $2-3 | 467 | $2.54 | $5.64 | $5.85 |
| $3-5 | 1078 | $4.06 | $6.98 | $5.58 |
| $5-8 | 1441 | $6.50 | $8.87 | $5.44 |
| $8-13 | 1923 | $10.24 | $11.42 | $5.26 |
| $13-21 | 2044 | $16.77 | $15.16 | $6.77 |
| $21-34 | 1666 | $26.56 | $19.15 | $10.76 |
| $34-55 | 1203 | $42.00 | $25.25 | $18.58 |
| $55-89 | 658 | $67.99 | $33.10 | $36.45 |
| $89-146 | 252 | $107.98 | $38.37 | $69.31 |

## Pack-size check

Mean of log(predicted / actual), shown as a percentage: -20% means predictions sit 20% below the real price on average.

| price range | single items | multipacks | single: predicted vs actual | multipack: predicted vs actual | single MAE | multipack MAE |
|---|---|---|---|---|---|---|
| $0-5 | 1186 | 655 | +112.6% | +137.2% | $5.52 | $6.41 |
| $5-10 | 1671 | 655 | +32.1% | +36.5% | $5.03 | $6.19 |
| $10-20 | 2187 | 726 | -2.4% | -15.2% | $5.89 | $6.78 |
| $20-50 | 1819 | 1025 | -34.2% | -43.9% | $12.69 | $13.98 |
| $50+ | 583 | 521 | -62.9% | -60.3% | $43.70 | $39.09 |

**Multipacks vs single items at the same price level:** -4.5% (95% interval -7.0% to -1.9%). Reading: multipacks are guessed lower than single items at the same price level, and the interval excludes 0. Consistent with the photo not showing pack size.

## Priced furthest too low

![Priced furthest too low](worst_under.jpg)

| sample_id | product | actual | predicted |
|---|---|---|---|
| 48275 | Manischewitz Fish Gefilte Jel | $137.99 | $6.15 |
| 26569 | Sugar Free Sports Drink Mix Powder, Lemonade 0.6 oz., PK500 | $142.31 | $10.64 |
| 160981 | Ortega Diced Green Chiles - 26 oz. can, 12 cans per case | $129.13 | $2.47 |
| 95334 | Dolores Tuna In Water 10z Wholesale, (24 - Pack) | $131.71 | $5.06 |
| 22821 | Laffy Taffy Rope Strawberry .81 oz. (288 count) | $128.88 | $3.33 |
| 66625 | World Honey Market Orange Blossom Honey /100% Pure Raw Orange [...] | $130.54 | $6.40 |
| 277711 | Kellogg's Low Fat Granola Cereal, With Raisins, 2.22oz (70 Count) | $129.99 | $6.24 |
| 157564 | General Mills Pillsbury Strawberry Creme Cheese Puff Pastry [...] | $142.58 | $21.56 |
| 99454 | Its Delish Dark Chili Powder (20 Lbs Bulk) | $135.03 | $18.96 |
| 169150 | Ocean Spray Gluten Free Whole Berry Cranberry Sauce, 117 Ounce [...] | $120.32 | $4.35 |
| 260506 | CLIF BAR - Variety Pack - Made with Organic Oats - 10-11g [...] | $142.99 | $28.38 |
| 206188 | TrueSeaMoss Sea Moss Gel – Made in USA – Wildcrafted Seamoss, [...] | $134.99 | $20.90 |

## Priced most too high (by ratio)

![Priced most too high (by ratio)](worst_over.jpg)

| sample_id | product | actual | predicted |
|---|---|---|---|
| 47473 | BUSH'S BEST Pinto Beans, 111 oz | $1.39 | $92.18 |
| 133712 | La Preferida, Rice Long Grain, 80 Ounce | $1.58 | $71.65 |
| 159724 | Justin's Nut Butter - Almond Butter Squeeze Pack Classic - 1.15 [...] | $1.49 | $33.92 |
| 215879 | Champagne Glitter Leaves Stem by Ashland® - Christmas Florals [...] | $1.99 | $39.52 |
| 296171 | Larabar Bar Apple Pie 1.6 Oz | $1.57 | $30.45 |
| 139364 | Bon Appetit Lemon Cake, 4 Ounce (Pack of 8) | $1.92 | $35.00 |
| 179298 | Surf Sweets Organic Fruity Bears, Non GMO Project Verified, [...] | $2.35 | $41.60 |
| 54082 | Iberia Passion Fruit Juice Drink, 16.57 Fl Oz (Pack of 12) | $1.72 | $29.76 |
| 27049 | Castillo Amor Black Label Hot Sauce, 33-Ounce Bottles (Pack of 12) | $2.98 | $48.34 |
| 146221 | Hint Water Crisp Apple (Pack of 24) 16 Ounce Bottles, Pure Water [...] | $1.79 | $28.55 |
| 270904 | Jell-O No Bake Oreo Dessert Kit (12.6 oz Box) | $4.32 | $67.62 |
| 205752 | Knorr Rice Sides Cajun Chicken Flavor Rice 8 ct for a Delicious [...] | $1.43 | $22.02 |
