"""Generate a synthetic loan-application dataset with a learnable default signal.

Run:  python generate_data.py  ->  applications.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)
N = 8000


def main() -> None:
    annual_income = np.clip(RNG.normal(9, 4, N), 1.5, 60) * 1e5      # INR
    loan_amount = np.clip(RNG.normal(6, 3, N), 0.5, 40) * 1e5
    dti = np.clip(loan_amount / annual_income, 0.02, 3)             # debt-to-income
    credit_util = np.clip(RNG.beta(2, 3, N), 0, 1)                  # 0..1
    credit_history_yrs = np.clip(RNG.normal(6, 3.5, N), 0, 25)
    num_delinquencies = RNG.poisson(0.4, N)
    num_open_accounts = np.clip(RNG.normal(6, 3, N), 1, 20).astype(int)

    logit = (
        -1.4
        + 1.8 * credit_util
        + 0.9 * dti
        + 0.55 * num_delinquencies
        - 0.09 * credit_history_yrs
        - 0.0000015 * annual_income
        + 0.03 * num_open_accounts
    )
    p = 1 / (1 + np.exp(-logit))
    default = RNG.random(N) < p

    df = pd.DataFrame({
        "app_id": [f"L{i:06d}" for i in range(N)],
        "annual_income": annual_income.round(0),
        "loan_amount": loan_amount.round(0),
        "dti": dti.round(3),
        "credit_util": credit_util.round(3),
        "credit_history_yrs": credit_history_yrs.round(1),
        "num_delinquencies": num_delinquencies,
        "num_open_accounts": num_open_accounts,
        "default": default.astype(int),
    })
    df.to_csv("applications.csv", index=False)
    print(f"wrote applications.csv ({N} rows, default rate {default.mean():.1%})")


if __name__ == "__main__":
    main()
