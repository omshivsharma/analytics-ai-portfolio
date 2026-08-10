"""Automated funnel root-cause analysis.

Given session-level funnel data split into two periods, this:
  1. builds the view -> cart -> checkout -> purchase funnel for each period,
  2. finds the funnel STEP with the biggest conversion drop,
  3. decomposes that drop across each dimension (device, browser, city, channel)
     and RANKS the segments by how much they contributed to the overall fall,
  4. prints a plain-English root-cause report.

This is the automated version of the manual "slice-and-dice" an analyst does when
a metric moves. Run:  python diagnose.py
"""
from __future__ import annotations

import sys

import pandas as pd

try:  # Windows consoles default to cp1252 and mangle unicode.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

STEPS = [
    ("view_to_cart", "viewed", "carted"),
    ("cart_to_checkout", "carted", "checkout"),
    ("checkout_to_purchase", "checkout", "purchased"),
]
DIMENSIONS = ["device", "browser", "city", "channel"]


def conv(df: pd.DataFrame, num: str, den: str) -> float:
    d = df[den].sum()
    return df[num].sum() / d if d else 0.0


def funnel_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, den, num in STEPS:
        rows.append({"step": name, "rate": conv(df, num, den)})
    return pd.DataFrame(rows)


def biggest_drop_step(base: pd.DataFrame, cur: pd.DataFrame) -> tuple[str, str, str, float, float]:
    worst = None
    for name, den, num in STEPS:
        rb, rc = conv(base, num, den), conv(cur, num, den)
        delta = rc - rb
        if worst is None or delta < worst[3]:
            worst = (name, den, num, delta, rb, rc)
    name, den, num, delta, rb, rc = worst
    return name, den, num, rb, rc


def decompose(base: pd.DataFrame, cur: pd.DataFrame, den: str, num: str, dim: str) -> pd.DataFrame:
    """Attribute the overall rate change to each segment of `dim`.

    contribution_s = (rate_current_s - rate_baseline_s) * (denominator_share_s)
    Summing contributions ~ the overall rate change (ignoring small mix effects).
    """
    out = []
    total_den_cur = cur[den].sum()
    for seg in sorted(set(base[dim]) | set(cur[dim])):
        b, c = base[base[dim] == seg], cur[cur[dim] == seg]
        rb, rc = conv(b, num, den), conv(c, num, den)
        share = c[den].sum() / total_den_cur if total_den_cur else 0
        contribution = (rc - rb) * share
        out.append({
            "segment": f"{dim}={seg}", "rate_baseline": rb, "rate_current": rc,
            "rate_delta": rc - rb, "volume_share": share, "contribution": contribution,
        })
    return pd.DataFrame(out).sort_values("contribution")


def main() -> None:
    df = pd.read_csv("sessions.csv")
    base = df[df.period == "baseline"]
    cur = df[df.period == "current"]

    print("=" * 68)
    print("FUNNEL ROOT-CAUSE REPORT")
    print("=" * 68)

    ft = funnel_table(base).merge(funnel_table(cur), on="step", suffixes=("_base", "_cur"))
    ft["delta_pp"] = (ft.rate_cur - ft.rate_base) * 100
    print("\nStep conversion — baseline vs current:")
    for _, r in ft.iterrows():
        print(f"  {r.step:22} {r.rate_base:6.1%} -> {r.rate_cur:6.1%}   ({r.delta_pp:+.1f} pp)")

    name, den, num, rb, rc = biggest_drop_step(base, cur)
    print(f"\n>> Biggest drop: '{name}'  {rb:.1%} -> {rc:.1%}  ({(rc-rb)*100:+.1f} pp)")

    print("\nWhat's driving it (ranked by contribution to the drop):")
    best_culprit = None
    for dim in DIMENSIONS:
        dec = decompose(base, cur, den, num, dim)
        top = dec.iloc[0]
        print(f"\n  by {dim}:")
        for _, r in dec.head(3).iterrows():
            flag = "  <== main culprit" if r.contribution == dec.contribution.min() and r.contribution < -0.005 else ""
            print(f"    {r.segment:22} {r.rate_baseline:5.1%} -> {r.rate_current:5.1%} "
                  f"| share {r.volume_share:4.0%} | contrib {r.contribution*100:+.2f} pp{flag}")
        if best_culprit is None or top.contribution < best_culprit[1]:
            best_culprit = (top.segment, top.contribution)

    # cross-segment confirmation: intersect the two strongest single-dim culprits
    print("\n" + "-" * 68)
    dec_dev = decompose(base, cur, den, num, "device").iloc[0]["segment"].split("=")[1]
    dec_br = decompose(base, cur, den, num, "browser").iloc[0]["segment"].split("=")[1]
    sub_b = base[(base.device == dec_dev) & (base.browser == dec_br)]
    sub_c = cur[(cur.device == dec_dev) & (cur.browser == dec_br)]
    print(f"CONCLUSION: '{name}' fell mainly in  device={dec_dev} + browser={dec_br}:")
    print(f"            {conv(sub_b, num, den):.1%} -> {conv(sub_c, num, den):.1%} "
          f"({(conv(sub_c,num,den)-conv(sub_b,num,den))*100:+.1f} pp) — check payment flow for that combo.")
    print("=" * 68)


if __name__ == "__main__":
    main()
