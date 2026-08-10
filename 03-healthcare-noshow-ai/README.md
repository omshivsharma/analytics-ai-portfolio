# 🏥 AI Patient No-Show & Feedback Engine  *(Healthcare · AI)*

Two problems clinics actually have, solved together:

1. **No-shows** quietly burn revenue and waste doctor time.
2. **Free-text patient feedback** piles up unread, so nobody fixes what's driving churn.

## Part A — No-show risk model
Predicts each appointment's probability of a no-show, then produces a **"call these
first" list** so staff reminder-call the highest-risk patients instead of everyone.

```
ROC-AUC — RandomForest: 0.70 | LogReg: 0.71
Top drivers: lead_time_days, prior_no_shows, appt_hour, sms_reminder
Opportunity: calling the top 10% risk (600 appts) targets 426 likely no-shows;
             ~127 recoverable at a 30% save rate.
```
The point isn't a fancy AUC — it's turning the score into a **prioritised action list
with a rupee-value opportunity**, which is what an ops lead actually wants.

## Part B — AI feedback structuring 🤖
An LLM reads each raw patient review and returns **theme + sentiment + a concrete
action**, then aggregates to show where to focus. Real output:

```
[negative] cost        | Prices were higher than quoted...
           action: Provide accurate pre-service estimates and advance notice of changes.
[negative] scheduling  | Rescheduling online is impossible...
           action: Fix online patient portal self-rescheduling.
Theme volume: positive 3 · communication 2 · wait time 1 · cost 1 · scheduling 1 ...
```
Falls back to a keyword classifier if no API key, so it always runs.

## Run it
```bash
pip install -r requirements.txt
python generate_data.py          # appointments.csv + reviews.csv
python noshow_model.py           # risk model + call-list + opportunity
setx GEMINI_API_KEY "your_key"   # (optional) for the AI feedback layer
python feedback_ai.py            # structured, actionable feedback
```

## Why it's not a generic ML demo
It pairs a classic classification task with a modern LLM insight layer on **messy real
text**, and both halves end in a **decision** (who to call; what to fix) tied to ROI —
exactly the analyst mindset, in a healthcare context.

**Real data swap:** the well-known *Medical Appointment No Shows* dataset (Kaggle) drops
straight into `noshow_model.py` with minor column renames.
