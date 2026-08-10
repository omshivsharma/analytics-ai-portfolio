"""AI layer: turn messy free-text patient feedback into structured, actionable rows.

For each review the LLM returns: theme, sentiment, and a concrete suggested action —
so operations can see, at a glance, what's driving dissatisfaction and what to fix.
Falls back to a keyword classifier if no GEMINI_API_KEY is set, so it always runs.

Run:  python feedback_ai.py     (set GEMINI_API_KEY for the AI version)
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")


def classify_ai(reviews: list[str]) -> list[dict] | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(reviews))
    prompt = (
        "You are a healthcare operations analyst. For each patient review below, return a "
        "JSON array; each element = {\"id\": <int>, \"theme\": <one of "
        "'wait time','scheduling','staff','cost','facility/location','communication','positive'>, "
        "\"sentiment\": <'positive'|'neutral'|'negative'>, \"action\": <short concrete fix>}. "
        "Return ONLY the JSON array.\n\n" + numbered
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={key}"
    try:
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=45)
        r.raise_for_status()
        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(txt)
    except Exception as e:  # noqa: BLE001
        print(f"  ! Gemini failed, using keyword fallback: {e}")
        return None


def classify_fallback(reviews: list[str]) -> list[dict]:
    kw = {
        "wait time": ["wait", "hour", "late", "delay"],
        "scheduling": ["reschedul", "book", "app", "phone", "call"],
        "cost": ["price", "billing", "cost", "quoted", "expensive"],
        "facility/location": ["far", "parking", "clean", "facility"],
        "staff": ["staff", "nurse", "doctor", "front desk", "rushed"],
    }
    out = []
    for i, t in enumerate(reviews):
        low = t.lower()
        theme = next((th for th, words in kw.items() if any(w in low for w in words)), "communication")
        pos = any(w in low for w in ["great", "smooth", "excellent", "clean", "appreciated", "clearly"])
        out.append({"id": i, "theme": "positive" if pos and theme == "staff" else theme,
                    "sentiment": "positive" if pos else "negative",
                    "action": "review this theme with ops"})
    return out


def main() -> None:
    reviews = pd.read_csv("reviews.csv")["text"].tolist()
    rows = classify_ai(reviews) or classify_fallback(reviews)
    df = pd.DataFrame(rows).merge(
        pd.DataFrame({"id": range(len(reviews)), "text": reviews}), on="id")

    print("=" * 74)
    print("PATIENT FEEDBACK — STRUCTURED BY AI")
    print("=" * 74)
    for _, r in df.iterrows():
        print(f"[{r['sentiment']:8}] {r['theme']:18} | {r['text'][:52]}")
        print(f"           action: {r['action']}")
    print("\nTheme volume (where to focus):")
    print(df["theme"].value_counts().to_string())
    print("=" * 74)


if __name__ == "__main__":
    main()
