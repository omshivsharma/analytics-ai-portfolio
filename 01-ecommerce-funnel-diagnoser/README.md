# 🔎 E-commerce Funnel Drop-off Diagnoser

**Problem.** When a conversion metric drops, an analyst spends hours manually slicing the
funnel by device, browser, city and channel to find *where* it broke. This automates that
root-cause hunt.

**What it does.** Given session-level funnel data across two periods, it:
1. builds the `view → cart → checkout → purchase` funnel for each period,
2. finds the step with the biggest conversion drop,
3. **decomposes that drop across every dimension and ranks the segments** by how much each
   contributed, and
4. confirms the culprit at the intersection of the two strongest dimensions.

## Result (on the sample data)
A payment bug was *planted* into one segment. The tool rediscovered it unaided:

```
Step conversion — baseline vs current:
  checkout_to_purchase    69.4% -> 61.7%   (-7.6 pp)   <- biggest drop

CONCLUSION: 'checkout_to_purchase' fell mainly in device=mobile + browser=Safari:
            70.3% -> 34.5% (-35.8 pp) — check payment flow for that combo.
```
City and channel showed only diffuse, small contributions — correctly *not* flagged.

## Run it
```bash
pip install -r requirements.txt
python generate_data.py   # writes sessions.csv (realistic, with a hidden root cause)
python diagnose.py        # prints the ranked root-cause report
```

## Method (the interesting bit)
For each dimension, each segment's contribution to the overall rate change is
`(rate_current − rate_baseline) × current_volume_share`. Summing these ≈ the total change,
so the most-negative contributions are the segments actually responsible — separating a
*big rate drop in a tiny segment* from a *small drop in a huge one*. The final step
intersects the top device × top browser to localise the issue precisely.

## Why it matters for a business
This is the analytical move behind real conversion recovery (I used the same thinking to
lift conversion 12% at a previous role). Turning it into a repeatable script means the
"why did the metric move?" question gets a first answer in seconds, not an afternoon.

## Swap in real data
Replace `sessions.csv` with any table that has boolean funnel-step columns
(`viewed, carted, checkout, purchased`) + dimension columns (`device, browser, city,
channel`) + a `period` column. Public option: the *eCommerce behavior data* dataset on Kaggle.
