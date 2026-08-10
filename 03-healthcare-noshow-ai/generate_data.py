"""Generate synthetic-but-realistic clinic data for the no-show engine.

Writes:
  appointments.csv  — one row per booked appointment, with a no_show label whose
                      probability genuinely depends on the features (so a model can
                      learn it and we can defend the drivers in an interview).
  reviews.csv       — free-text patient feedback for the AI insight layer.

Run:  python generate_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N = 6000


def main() -> None:
    lead_time = RNG.integers(0, 45, N)            # days between booking and appt
    age = np.clip(RNG.normal(41, 17, N), 1, 95).astype(int)
    sms_reminder = RNG.random(N) < 0.6
    prior_no_shows = RNG.poisson(0.5, N)
    hour = RNG.integers(8, 20, N)
    is_new_patient = RNG.random(N) < 0.35
    chronic = RNG.random(N) < 0.25

    # True underlying log-odds of a no-show (the signal the model should recover).
    logit = (
        -1.9
        + 0.045 * lead_time            # longer wait -> more no-shows
        - 0.9 * sms_reminder           # reminders help a lot
        + 0.6 * prior_no_shows         # habit is the strongest signal
        + 0.4 * is_new_patient
        - 0.012 * age                  # older patients show up more
        + 0.05 * (hour >= 17)          # late-evening slots slightly worse
        - 0.3 * chronic                # chronic patients are more committed
    )
    p = 1 / (1 + np.exp(-logit))
    no_show = RNG.random(N) < p

    df = pd.DataFrame({
        "appointment_id": [f"a{i:06d}" for i in range(N)],
        "lead_time_days": lead_time, "age": age,
        "sms_reminder": sms_reminder.astype(int),
        "prior_no_shows": prior_no_shows, "appt_hour": hour,
        "is_new_patient": is_new_patient.astype(int),
        "chronic_condition": chronic.astype(int),
        "no_show": no_show.astype(int),
    })
    df.to_csv("appointments.csv", index=False)

    reviews = [
        "Waited over an hour past my slot. Doctor was kind but the front desk is chaos.",
        "Booking on the app was smooth and I got an SMS reminder, really appreciated it.",
        "Could not reach anyone on the phone to reschedule, so I just didn't go.",
        "Great experience, staff explained everything clearly and billing was transparent.",
        "The clinic is too far and parking is a nightmare, thinking of switching.",
        "Reminder came too late, only an hour before. Almost missed it.",
        "Very clean facility and short wait. Nurse was excellent.",
        "Prices were higher than quoted and no one warned me in advance.",
        "Rescheduling online is impossible, had to call three times.",
        "Doctor rushed the consultation, felt like a number not a patient.",
    ]
    pd.DataFrame({"review_id": range(len(reviews)), "text": reviews}).to_csv("reviews.csv", index=False)
    print(f"wrote appointments.csv ({N} rows, no-show rate {no_show.mean():.1%}) and reviews.csv ({len(reviews)} reviews)")


if __name__ == "__main__":
    main()
