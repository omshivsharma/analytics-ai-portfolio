"""Predict appointment no-show risk and produce a 'who to call today' list.

The business move: clinics can't call everyone, so rank tomorrow's appointments by
no-show risk and have staff reminder-call the top slice — recovering revenue that
would otherwise walk out the door.

Run:  python noshow_model.py
"""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

FEATURES = ["lead_time_days", "age", "sms_reminder", "prior_no_shows",
            "appt_hour", "is_new_patient", "chronic_condition"]


def main() -> None:
    df = pd.read_csv("appointments.csv")
    X, y = df[FEATURES], df["no_show"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

    rf = RandomForestClassifier(n_estimators=250, max_depth=8, random_state=1, n_jobs=1)
    rf.fit(Xtr, ytr)
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)

    auc_rf = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
    auc_lr = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
    print("=" * 64)
    print("NO-SHOW RISK MODEL")
    print("=" * 64)
    print(f"ROC-AUC — RandomForest: {auc_rf:.3f} | LogReg: {auc_lr:.3f}")

    imp = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=1, n_jobs=1)
    order = imp.importances_mean.argsort()[::-1]
    print("\nTop drivers of no-show (permutation importance):")
    for i in order[:5]:
        print(f"  {FEATURES[i]:18} {imp.importances_mean[i]:.3f}")

    # Score everyone, build the priority call-list.
    df["risk"] = rf.predict_proba(X)[:, 1]
    call = df.sort_values("risk", ascending=False).head(10)
    print("\n'Call these first' — highest-risk upcoming appointments:")
    print("  appt_id   risk  lead  prior_ns  sms  new")
    for _, r in call.iterrows():
        print(f"  {r.appointment_id}  {r.risk:4.0%}  {int(r.lead_time_days):4}  "
              f"{int(r.prior_no_shows):8}  {int(r.sms_reminder):3}  {int(r.is_new_patient):3}")

    # Quantify the opportunity: if reminder-calling recovers ~30% of would-be no-shows
    top_decile = df.sort_values("risk", ascending=False).head(len(df) // 10)
    recoverable = int(top_decile["no_show"].sum() * 0.30)
    print(f"\nOpportunity: calling the top 10% risk ({len(top_decile)} appts) targets "
          f"{int(top_decile['no_show'].sum())} likely no-shows; ~{recoverable} recoverable at 30% save rate.")
    print("=" * 64)


if __name__ == "__main__":
    main()
