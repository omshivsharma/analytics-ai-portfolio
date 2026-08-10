# 💳 AI Credit-Risk Explainer  *(Finance · AI)*

**Problem.** Lenders can build a risk score easily — the hard part is **explaining** each
decision in language a customer, loan officer, or auditor accepts. "The model said no" is
not a valid adverse-action reason.

**What it does.**
1. Trains a credit-default risk model on applicant features.
2. For each applicant, computes **per-feature contributions** to the risk (standardized
   logistic-regression push to the log-odds) — a transparent, defensible attribution.
3. An **LLM turns those drivers into a plain-English adverse-action notice**, grounded
   strictly in the model's actual top factors (no hallucinated reasons).

## Real output
```
ROC-AUC: 0.79

L002544 | risk 97% | income ₹150,000, util 69%, dti 3.00, delinq 3
  top risk drivers: high debt-to-income, past delinquencies, low income
  AI adverse-action notice: Your loan application was declined due to your low income,
  high debt-to-income ratio, and past delinquencies.
```

## Run it
```bash
pip install -r requirements.txt
python generate_data.py          # applications.csv
setx GEMINI_API_KEY "your_key"   # (optional) for the AI notice; runs without it too
python risk_model.py             # AUC + high-risk applicants + explanations
```

## Why it's analyst/fintech-grade
Most credit projects stop at AUC. This one pairs the model with **explainability +
regulator-friendly narratives** — the actual pain point in BFSI/fintech analytics. The
explanation is tied to real feature contributions, so the AI can't invent a reason.

**Real data swap:** use the *German Credit* or *Lending Club* datasets — map columns to the
seven features in `risk_model.py`. (SHAP can replace the coefficient attribution for
non-linear models with one function swap.)
