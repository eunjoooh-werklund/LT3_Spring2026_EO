"""
generate_data.py
-----------------
Generates realistic mock/simulated data for the AI-in-K12-Classrooms
Principal Dashboard assignment.

Run this once to produce CSV files in /data, OR import generate_all()
directly from app.py (the app calls this automatically if /data is empty).

Why mock data is structured this way:
- One row per (teacher, week) for the time-series / usage tables, so a
  principal can filter by date range.
- A separate "comments" table for free-text teacher feedback, which is
  what gets fed into the sentiment analysis (VADER) in the dashboard.
- Teacher attributes (department, years experience, school) are kept in
  a lookup table to avoid repeating strings across thousands of rows.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)  # fixed seed -> reproducible mock data

DEPARTMENTS = ["Math", "Science", "English", "Social Studies", "Elective/Arts", "Special Education"]
SCHOOLS = ["Lincoln Middle School"]  # single school; change if you want multi-school comparison
N_TEACHERS = 42
N_WEEKS = 16  # one semester, weekly cadence

PD_TOPICS = [
    "AI Ethics & Academic Integrity",
    "Prompt Engineering for Lesson Design",
    "AI-Assisted Differentiation",
    "Grading & Feedback with AI",
    "Data Privacy & Student Safety",
    "AI for IEP/504 Accommodations",
]

SUPPORT_TYPES = [
    "Need ed-tech coach in classroom",
    "Need more planning time",
    "Need clearer AI use policy",
    "Need training on a specific tool",
    "Need help with student misuse/cheating concerns",
    "Need help integrating AI with curriculum standards",
]

# Pools of realistic free-text comments, biased by sentiment bucket.
# In a real deployment these would be actual open-ended survey responses.
POSITIVE_COMMENTS = [
    "The AI lesson-planning tool has cut my prep time significantly and I finally have time to focus on differentiation.",
    "I feel much more confident about the district's AI policy after the last training session.",
    "Students are more engaged when I use AI-generated practice problems tailored to their level.",
    "I appreciate having clear guidance on what counts as acceptable AI use for assignments.",
    "The micro-credential course on prompt engineering was genuinely useful and easy to apply right away.",
    "I trust the new assessment rubric because it clearly separates AI-assisted drafts from final work.",
    "My team meetings are more productive now that we share AI-generated differentiation strategies.",
    "I feel supported by administration when questions about AI use come up with parents.",
]
NEUTRAL_COMMENTS = [
    "I'm still getting used to the new AI tools, it's a mixed bag depending on the week.",
    "Some students use the AI tools well, others just copy outputs without thinking.",
    "I have questions about how to grade work that used AI assistance fairly.",
    "Training has been okay but I'd like more subject-specific examples.",
    "Not sure yet if this is saving me time overall once I factor in fixing AI mistakes.",
    "I use it for some tasks but not others, depends on the lesson.",
]
NEGATIVE_COMMENTS = [
    "I don't feel like I understand the expectations for AI use well enough to enforce them confidently.",
    "Planning time hasn't really gone down, I still have to heavily edit what the AI produces.",
    "I'm worried about students relying on AI instead of building foundational skills.",
    "The policy on academic integrity and AI feels unclear and inconsistent across departments.",
    "I haven't had time to attend any of the AI training sessions offered this semester.",
    "I need an ed-tech specialist in my room because I don't feel confident troubleshooting these tools alone.",
    "I'm skeptical this is actually helping engagement, it feels like a novelty that's wearing off.",
]


def _teacher_lookup():
    teacher_ids = [f"T{idx:03d}" for idx in range(1, N_TEACHERS + 1)]
    dept = RNG.choice(DEPARTMENTS, size=N_TEACHERS, p=[0.20, 0.18, 0.18, 0.16, 0.14, 0.14])
    years_exp = RNG.integers(1, 31, size=N_TEACHERS)
    # AI adoption tendency: newer teachers + electives skew slightly higher adopters (mock assumption)
    base_adoption = RNG.normal(0.55, 0.18, size=N_TEACHERS)
    adoption_propensity = np.clip(base_adoption, 0.05, 0.95)
    school = RNG.choice(SCHOOLS, size=N_TEACHERS)

    df = pd.DataFrame({
        "teacher_id": teacher_ids,
        "teacher_name": [f"Teacher {i}" for i in range(1, N_TEACHERS + 1)],  # anonymized by design
        "department": dept,
        "years_experience": years_exp,
        "school": school,
        "adoption_propensity": adoption_propensity,
    })
    return df


def _weekly_usage_and_planning(teachers: pd.DataFrame):
    """One row per teacher per week: AI usage frequency, planning time, achievement & engagement proxies."""
    start = datetime(2026, 1, 12)  # semester start (a Monday)
    rows = []
    for _, t in teachers.iterrows():
        # Each teacher has a slowly increasing adoption trend over the semester + noise
        trend = np.linspace(0, 0.25, N_WEEKS) * t["adoption_propensity"]
        weekly_usage_rate = np.clip(t["adoption_propensity"] + trend + RNG.normal(0, 0.07, N_WEEKS), 0, 1)

        # AI sessions per week (count of distinct AI-assisted planning/teaching actions)
        ai_sessions = RNG.poisson(lam=weekly_usage_rate * 8)

        # Planning time WITHOUT AI baseline ~ 5-9 hrs/week depending on experience (less exp = more time)
        baseline_planning_hours = np.clip(RNG.normal(7.5 - 0.05 * t["years_experience"], 1.0, N_WEEKS), 3, 12)
        # Time saved scales with usage rate (diminishing returns) + noise
        time_saved_hours = weekly_usage_rate * RNG.normal(2.2, 0.6, N_WEEKS)
        time_saved_hours = np.clip(time_saved_hours, 0, baseline_planning_hours * 0.6)
        actual_planning_hours = np.clip(baseline_planning_hours - time_saved_hours, 1.5, None)

        # Engagement score (1-100), achievement delta (% change vs prior unit test, can be negative)
        engagement_score = np.clip(
            55 + weekly_usage_rate * 25 + RNG.normal(0, 6, N_WEEKS), 0, 100
        )
        achievement_delta_pct = weekly_usage_rate * RNG.normal(4.5, 2.0, N_WEEKS) + RNG.normal(0, 1.8, N_WEEKS)

        for w in range(N_WEEKS):
            rows.append({
                "teacher_id": t["teacher_id"],
                "department": t["department"],
                "week_start": (start + timedelta(weeks=w)).date().isoformat(),
                "week_number": w + 1,
                "ai_sessions": int(ai_sessions[w]),
                "ai_usage_rate": round(float(weekly_usage_rate[w]), 3),
                "planning_hours_baseline": round(float(baseline_planning_hours[w]), 2),
                "planning_hours_actual": round(float(actual_planning_hours[w]), 2),
                "time_saved_hours": round(float(time_saved_hours[w]), 2),
                "engagement_score": round(float(engagement_score[w]), 1),
                "achievement_delta_pct": round(float(achievement_delta_pct[w]), 2),
            })
    return pd.DataFrame(rows)


def _pd_records(teachers: pd.DataFrame):
    """Self-reported professional learning: completed vs. desired micro-credentials/courses."""
    rows = []
    for _, t in teachers.iterrows():
        n_completed = RNG.integers(0, 4)
        completed = list(RNG.choice(PD_TOPICS, size=n_completed, replace=False)) if n_completed > 0 else []
        n_desired = RNG.integers(0, 3)
        remaining_pool = [p for p in PD_TOPICS if p not in completed]
        desired = list(RNG.choice(remaining_pool, size=min(n_desired, len(remaining_pool)), replace=False)) if remaining_pool and n_desired > 0 else []
        goal_progress_pct = int(np.clip(RNG.normal(55, 25), 0, 100))
        rows.append({
            "teacher_id": t["teacher_id"],
            "department": t["department"],
            "credentials_completed": "; ".join(completed) if completed else "None",
            "n_completed": n_completed,
            "credentials_desired": "; ".join(desired) if desired else "None",
            "n_desired": len(desired),
            "professional_goal_progress_pct": goal_progress_pct,
        })
    return pd.DataFrame(rows)


def _support_requests(teachers: pd.DataFrame):
    """Self-reported needs / support requests (the 'I need an ed-tech specialist' tab)."""
    rows = []
    request_id = 1
    today = datetime(2026, 6, 21)
    for _, t in teachers.iterrows():
        n_requests = RNG.integers(0, 3)
        if n_requests == 0:
            continue
        chosen = RNG.choice(SUPPORT_TYPES, size=n_requests, replace=False)
        for req in chosen:
            urgency = RNG.choice(["Low", "Medium", "High"], p=[0.35, 0.4, 0.25])
            status = RNG.choice(["Open", "In Progress", "Resolved"], p=[0.45, 0.30, 0.25])
            days_ago = int(RNG.integers(0, 90))
            rows.append({
                "request_id": f"R{request_id:04d}",
                "teacher_id": t["teacher_id"],
                "department": t["department"],
                "request_type": req,
                "urgency": urgency,
                "status": status,
                "date_submitted": (today - timedelta(days=days_ago)).date().isoformat(),
                "days_open": days_ago if status != "Resolved" else 0,
            })
            request_id += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # --- Priority scoring methodology (Tab 7) ---
    # priority_score = weighted sum of urgency + how long it's been open + how many
    # similar (same-type) requests are active, so principals see what to triage first.
    urgency_weight = {"Low": 1, "Medium": 2, "High": 3}
    df["urgency_weight"] = df["urgency"].map(urgency_weight)
    df["recency_weight"] = np.clip(df["days_open"] / 30, 0, 3)  # caps at 3 (90+ days)
    type_freq = df[df["status"] != "Resolved"]["request_type"].value_counts()
    df["frequency_weight"] = df["request_type"].map(type_freq).fillna(0) / max(type_freq.max(), 1) * 2

    raw_score = df["urgency_weight"] * 2 + df["recency_weight"] + df["frequency_weight"]
    df["priority_score"] = (raw_score / raw_score.max() * 100).round(1)
    df.loc[df["status"] == "Resolved", "priority_score"] = 0.0

    return df



def _sentiment_comments(teachers: pd.DataFrame):
    """Free-text comments for sentiment/trust analysis (Tab 6 — the methodology centerpiece)."""
    rows = []
    comment_id = 1
    start = datetime(2026, 1, 12)
    for _, t in teachers.iterrows():
        # Comfort/trust level skews with adoption_propensity but isn't identical to it
        trust_lean = np.clip(t["adoption_propensity"] + RNG.normal(0, 0.2), 0, 1)
        n_comments = RNG.integers(1, 5)
        weeks_commented = RNG.choice(range(N_WEEKS), size=n_comments, replace=False)
        for w in sorted(weeks_commented):
            roll = RNG.random()
            if roll < trust_lean - 0.15:
                text = RNG.choice(POSITIVE_COMMENTS)
                bucket = "positive_source"
            elif roll > trust_lean + 0.15:
                text = RNG.choice(NEGATIVE_COMMENTS)
                bucket = "negative_source"
            else:
                text = RNG.choice(NEUTRAL_COMMENTS)
                bucket = "neutral_source"
            rows.append({
                "comment_id": f"C{comment_id:05d}",
                "teacher_id": t["teacher_id"],
                "department": t["department"],
                "week_start": (start + timedelta(weeks=int(w))).date().isoformat(),
                "comment_text": text,
                "_source_bucket": bucket,  # for sanity-checking VADER later; not shown in dashboard
            })
            comment_id += 1
    return pd.DataFrame(rows)


def _trust_readiness_survey(teachers: pd.DataFrame):
    """Likert-scale (1-5) survey: trust & readiness around AI policy/expectations.

    Used for Tab 5 (Trust & Readiness) and as clustering features for Tab 6
    (Professional Learning readiness groups).
    """
    rows = []
    for _, t in teachers.iterrows():
        lean = t["adoption_propensity"]  # reuse propensity to keep responses internally consistent

        def likert(center_shift=0.0, noise=0.7):
            val = 3 + (lean - 0.5) * 4 + center_shift + RNG.normal(0, noise)
            return int(np.clip(round(val), 1, 5))

        rows.append({
            "teacher_id": t["teacher_id"],
            "department": t["department"],
            "policy_clarity": likert(),              # "I understand the AI use policy"
            "assessment_confidence": likert(-0.2),    # "I'm confident assessing AI-assisted work fairly"
            "comfort_with_ai_tools": likert(0.3),     # "I'm comfortable using AI tools in my classroom"
            "perceived_admin_support": likert(0.1),   # "I feel supported by administration on AI questions"
            "pedagogy_understanding": likert(-0.1),   # "I understand how to pedagogically integrate AI"
        })
    df = pd.DataFrame(rows)
    likert_cols = ["policy_clarity", "assessment_confidence", "comfort_with_ai_tools",
                   "perceived_admin_support", "pedagogy_understanding"]
    df["trust_index"] = df[likert_cols].mean(axis=1).round(2)  # composite 1-5 trust/readiness index
    return df


def generate_all(out_dir="data"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    teachers = _teacher_lookup()
    usage = _weekly_usage_and_planning(teachers)
    pd_records = _pd_records(teachers)
    support = _support_requests(teachers)
    comments = _sentiment_comments(teachers)
    trust = _trust_readiness_survey(teachers)

    teachers.to_csv(f"{out_dir}/teachers.csv", index=False)
    usage.to_csv(f"{out_dir}/weekly_usage.csv", index=False)
    pd_records.to_csv(f"{out_dir}/professional_learning.csv", index=False)
    support.to_csv(f"{out_dir}/support_requests.csv", index=False)
    comments.to_csv(f"{out_dir}/comments.csv", index=False)
    trust.to_csv(f"{out_dir}/trust_readiness.csv", index=False)

    print(f"Generated mock data in ./{out_dir}/:")
    print(f"  teachers.csv              {len(teachers)} rows")
    print(f"  weekly_usage.csv          {len(usage)} rows")
    print(f"  professional_learning.csv {len(pd_records)} rows")
    print(f"  support_requests.csv      {len(support)} rows")
    print(f"  comments.csv              {len(comments)} rows")
    print(f"  trust_readiness.csv       {len(trust)} rows")

    return teachers, usage, pd_records, support, comments, trust


if __name__ == "__main__":
    generate_all()
