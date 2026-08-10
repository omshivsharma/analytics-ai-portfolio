"""Credit-default risk model + per-applicant, regulator-friendly explanations.

The twist that makes this analyst-grade: it doesn't just output a score, it explains
*why* each applicant is risky using standardized logistic-regression contributions
(the push each feature adds to the log-odds), then an LLM turns that into a plain-English
adverse-action reason a customer or auditor can read.

Run:  python risk_model.py       (set GEMINI_API_KEY for AI narratives)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

FEATURES = ["annual_income", "loan_amount", "dti", "credit_util",
            "credit_history_yrs", "num_delinquencies", "num_open_accounts"]
NICE = {
    "annual_income": "low income", "loan_amount": "large loan",
    "dti": "high debt-to-income", "credit_util": "high credit utilisation",
    "credit_history_yrs": "short credit history",
    "num_delinquencies": "past delinquencies", "num_open_accounts": "many open accounts",
}
MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")


def contributions(scaler, model, row_scaled) -> list[tuple[str, float]]:
    """Per-feature push to log-odds = coef * standardized_value. Signed."""
    contrib = model.coef_[0] * row_scaled
    pairs = sorted(zip(FEATURES, contrib), key=lambda kv: kv[1], reverse=True)
    return pairs


def ai_reason(top_factors: list[str]) -> str | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    prompt = (
        "Write a single, plain-English adverse-action sentence (max 25 words) explaining a "
        "loan decline to the applicant, based ONLY on these top risk drivers: "
        f"{', '.join(top_factors)}. Be factual and non-judgemental. Return only the sentence."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={key}"
    try:
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=40)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    df = pd.read_csv("applications.csv")
    X, y = df[FEATURES], df["default"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
    auc = roc_auc_score(yte, model.predict_proba(scaler.transform(Xte))[:, 1])

    print("=" * 70)
    print("CREDIT-DEFAULT RISK MODEL + EXPLANATIONS")
    print("=" * 70)
    print(f"ROC-AUC: {auc:.3f}")

    df["risk"] = model.predict_proba(scaler.transform(X))[:, 1]
    Xs = scaler.transform(X)

    riskiest = df.sort_values("risk", ascending=False).head(3).index
    print("\nSample high-risk applicants with explanations:")
    for idx in riskiest:
        pairs = contributions(scaler, model, Xs[idx])
        top = [NICE[f] for f, c in pairs if c > 0][:3]
        print(f"\n  {df.loc[idx,'app_id']} | risk {df.loc[idx,'risk']:.0%} "
              f"| income ₹{df.loc[idx,'annual_income']:,.0f}, util {df.loc[idx,'credit_util']:.0%}, "
              f"dti {df.loc[idx,'dti']:.2f}, delinq {df.loc[idx,'num_delinquencies']}")
        print(f"    top risk drivers: {', '.join(top)}")
        reason = ai_reason(top)
        if reason:
            print(f"    AI adverse-action notice: {reason}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
