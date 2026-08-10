# Omshiv Sharma — Analytics & AI Portfolio

Four projects across **different sectors** (analyst roles span industries), each solving a
problem a real company has, using public/synthetic data you can defend in an interview, and
shipping a **decision** — not just a model. **Two are AI projects** (high demand right now).
None are Swiggy/food-delivery, so they read as independent work, not day-job spillover.

| # | Sector | Project | AI? | Status | Core skills shown |
|---|--------|---------|-----|--------|-------------------|
| 1 | E-commerce | [Funnel Drop-off Diagnoser](01-ecommerce-funnel-diagnoser/) | — | ✅ built | SQL, Python, RCA, segmentation |
| 3 | Healthcare | [AI No-Show & Feedback Engine](03-healthcare-noshow-ai/) | 🤖 | ✅ built | LLM, NLP, classification |
| 4 | Finance | [AI Credit-Risk Explainer](04-finance-credit-risk-ai/) | 🤖 | ✅ built | ML, explainability, LLM |
| 2 | Retail / CPG | Demand Forecaster + BI | — | ⏳ planned | Forecasting, Power BI, ops decisions |

Three of four are built and runnable. #2 (Power BI) is a GUI build — I can generate the
forecast model + dashboard spec, but you'll assemble the `.pbix` in Power BI Desktop.

---

## 1. E-commerce — Funnel Drop-off Diagnoser 🔎  *(building first)*
**Problem:** "Conversion dropped last week — why?" costs analysts hours of manual slicing.
**What it is:** ingests session/event data, builds the view→cart→checkout→purchase funnel,
compares two periods, and **automatically ranks which segment (device, browser, city,
channel) explains the drop** — a mini automated root-cause report.
**Not generic because:** it automates RCA (exactly the analytical thinking that got you a
12% conversion lift), and shows real SQL/pandas depth, not a plotting exercise.
**Stack:** Python, pandas, DuckDB/SQL. Runs offline on generated realistic data.
**Deliverable:** `python diagnose.py` prints a ranked "what caused the drop" report.

## 2. Retail / CPG — Demand Forecaster + BI 📈
**Problem:** stores over/under-stock because demand swings by day, promo, and season.
**What it is:** hourly/daily demand forecast per store-SKU + a **Power BI dashboard** that
turns the forecast into a stock/staffing recommendation and quantifies the cost of error
("what does 10% forecast error cost?").
**Not generic because:** it ends at a **dashboard a manager would use**, not an RMSE number —
your Matter Energy demand-forecasting + Power BI strength, in a retail context.
**Stack:** Python (Prophet / gradient boosting), Power BI. Public retail sales data (cited).

## 3. Healthcare — AI Patient-Feedback & No-Show Engine 🤖
**Problem:** clinics lose revenue to no-shows and drown in unstructured patient feedback.
**What it is:** (a) a model predicting appointment **no-show risk**, and (b) an **LLM layer**
that reads free-text patient reviews/feedback and returns structured themes + sentiment +
suggested action — so ops can prioritise outreach.
**Not generic because:** combines a classic classification task with a modern LLM insight
layer on messy text, tied to a clear ROI (recovered appointments).
**Stack:** Python, scikit-learn, Gemini/OpenAI for the NLP layer. Public no-show dataset.

## 4. Finance — AI Credit-Risk Explainer 🤖
**Problem:** loan/credit teams need risk scores they can actually explain to customers & auditors.
**What it is:** a credit-default risk model **plus an LLM that turns each decision into a
plain-English "why"** ("declined mainly due to high utilisation + short history"), grounded
in the model's feature contributions (SHAP).
**Not generic because:** pairs ML with **explainability + regulator-friendly narratives** — a
real pain point in fintech/BFSI analyst roles.
**Stack:** Python, scikit-learn/XGBoost, SHAP, Gemini/OpenAI. Public credit dataset (e.g. German Credit / Lending Club).

---

## How to present it
- A **GitHub profile README** linking all four: each with a 2-line "problem → result" + one screenshot.
- Every repo README leads with **the problem and the decision**, then the how.
- Replace the Housing/Music-Store lines on your resume with #1 and #3 headline results.
- Optional single-page site (GitHub Pages) — can be generated from these four.
