"""Generate a realistic e-commerce session dataset with a *planted* root cause.

Two 14-day periods (baseline vs current). Everything is stable EXCEPT a payment
bug we inject into `Safari` on `mobile` during the current period, which quietly
tanks checkout->purchase conversion for that one segment. The diagnoser's job is
to rediscover this without being told.

Run:  python generate_data.py  ->  writes sessions.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_P = [0.62, 0.32, 0.06]
BROWSERS = ["Chrome", "Safari", "Firefox", "Edge"]
BROWSER_P = [0.55, 0.28, 0.10, 0.07]
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Pune", "Kolkata"]
CHANNELS = ["organic", "paid", "social", "email", "direct"]
CHANNEL_P = [0.34, 0.24, 0.18, 0.10, 0.14]

# Baseline conversion rates for each funnel step (probabilities of advancing).
BASE = {"view_to_cart": 0.34, "cart_to_checkout": 0.55, "checkout_to_purchase": 0.68}

N_PER_DAY = 1600


def _pick(options, p, n):
    return RNG.choice(options, size=n, p=p)


def _period(start: str, days: int, bug: bool) -> pd.DataFrame:
    rows = []
    dates = pd.date_range(start, periods=days, freq="D")
    for d in dates:
        n = N_PER_DAY + int(RNG.normal(0, 80))
        device = _pick(DEVICES, DEVICE_P, n)
        browser = _pick(BROWSERS, BROWSER_P, n)
        city = _pick(CITIES, [1 / len(CITIES)] * len(CITIES), n)
        channel = _pick(CHANNELS, CHANNEL_P, n)

        # step 1: view -> cart (paid traffic converts a bit worse)
        p_cart = np.where(channel == "paid", BASE["view_to_cart"] - 0.04, BASE["view_to_cart"])
        carted = RNG.random(n) < p_cart

        # step 2: cart -> checkout
        p_co = np.full(n, BASE["cart_to_checkout"])
        checkout = carted & (RNG.random(n) < p_co)

        # step 3: checkout -> purchase  (<-- inject the bug here)
        p_pur = np.full(n, BASE["checkout_to_purchase"])
        if bug:
            hit = (device == "mobile") & (browser == "Safari")
            p_pur = np.where(hit, 0.34, p_pur)   # payment bug halves conversion
        purchased = checkout & (RNG.random(n) < p_pur)

        rows.append(pd.DataFrame({
            "date": d.date().isoformat(),
            "device": device, "browser": browser, "city": city, "channel": channel,
            "viewed": True, "carted": carted, "checkout": checkout, "purchased": purchased,
        }))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    baseline = _period("2026-06-01", 14, bug=False)
    baseline["period"] = "baseline"
    current = _period("2026-06-15", 14, bug=True)
    current["period"] = "current"
    df = pd.concat([baseline, current], ignore_index=True)
    df.insert(0, "session_id", [f"s{i:07d}" for i in range(len(df))])
    df.to_csv("sessions.csv", index=False)
    print(f"wrote sessions.csv  ({len(df):,} sessions, planted bug: mobile+Safari checkout->purchase)")


if __name__ == "__main__":
    main()
