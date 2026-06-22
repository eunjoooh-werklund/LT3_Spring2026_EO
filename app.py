"""
app.py — AI in the Classroom: Principal Dashboard (mock data demo)
====================================================================
A 7-tab Streamlit dashboard demonstrating how a K-12 principal could
monitor teacher AI adoption and its connection to student achievement,
engagement, planning workload, teacher needs, trust/readiness, professional
learning, and resource-allocation priorities.

Each tab maps to one row of the advisor-approved tab plan:

  1. Overview                  -> frequency counts, adoption rate, trend summaries
  2. Achievement & Engagement  -> correlation, before/after comparison, engagement scoring
  3. Teacher Planning Time     -> min/max/mean, boxplots, outlier detection
  4. Teacher Needs             -> text mining, topic frequency, sentiment
  5. Trust & Readiness         -> sentiment analysis, Likert-scale scoring
  6. Professional Learning     -> recommendation logic, clustering
  7. Classroom Support Requests-> priority scoring

Run with:
    pip install -r requirements.txt
    streamlit run app.py

All data is SIMULATED for demonstration purposes only (see generate_data.py).
"""

import os
import re
from collections import Counter

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from generate_data import generate_all

# ----------------------------------------------------------------------------
# Page config & data loading
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI in the Classroom — Principal Dashboard",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = "data"


@st.cache_data
def load_data():
    if not os.path.exists(f"{DATA_DIR}/teachers.csv"):
        generate_all(DATA_DIR)
    teachers = pd.read_csv(f"{DATA_DIR}/teachers.csv")
    usage = pd.read_csv(f"{DATA_DIR}/weekly_usage.csv", parse_dates=["week_start"])
    pd_records = pd.read_csv(f"{DATA_DIR}/professional_learning.csv")
    support = pd.read_csv(f"{DATA_DIR}/support_requests.csv", parse_dates=["date_submitted"])
    comments = pd.read_csv(f"{DATA_DIR}/comments.csv", parse_dates=["week_start"])
    trust = pd.read_csv(f"{DATA_DIR}/trust_readiness.csv")
    return teachers, usage, pd_records, support, comments, trust


@st.cache_data
def score_sentiment(comments: pd.DataFrame):
    """Applies VADER sentiment analysis to free-text teacher comments — the
    'principals wouldn't have time to run this themselves' methodology piece."""
    analyzer = SentimentIntensityAnalyzer()
    scores = comments["comment_text"].apply(analyzer.polarity_scores)
    comments = comments.copy()
    comments["compound"] = scores.apply(lambda s: s["compound"])
    comments["sentiment_label"] = pd.cut(
        comments["compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["Negative", "Neutral", "Positive"],
    )
    return comments


STOPWORDS = set("""
the a an and or but if is are was were be been being to of in on for with
as at by from this that it its i my we our you your they their he she
not no so it's i'm i've don't doesn't didn't isn't more most still very
have has had do does did just about than then them than feel feels felt
""".split())


def top_keywords(texts: pd.Series, n=12):
    """Very small keyword-frequency text-mining utility (no extra NLP deps):
    lowercases, strips punctuation, removes stopwords, counts remaining words."""
    counter = Counter()
    for t in texts.dropna():
        words = re.findall(r"[a-zA-Z']+", t.lower())
        counter.update(w for w in words if w not in STOPWORDS and len(w) > 2)
    return counter.most_common(n)


teachers, usage, pd_records, support, comments, trust = load_data()
comments = score_sentiment(comments)
usage = usage.merge(teachers[["teacher_id", "years_experience"]], on="teacher_id", how="left")

# ----------------------------------------------------------------------------
# Sidebar — global filters
# ----------------------------------------------------------------------------
st.sidebar.title("🎛️ Filters")
st.sidebar.caption("Filter every tab below by department and date range.")

all_depts = sorted(teachers["department"].unique())
selected_depts = st.sidebar.multiselect("Department", all_depts, default=all_depts)

min_week, max_week = int(usage["week_number"].min()), int(usage["week_number"].max())
week_range = st.sidebar.slider("Week of semester", min_week, max_week, (min_week, max_week))

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **All data on this dashboard is simulated** for demonstration purposes "
    "(see `generate_data.py`). No real student or teacher data is used."
)

# Apply filters
usage_f = usage[
    usage["department"].isin(selected_depts)
    & usage["week_number"].between(week_range[0], week_range[1])
]
teacher_ids_in_scope = usage_f["teacher_id"].unique()
pd_f = pd_records[pd_records["department"].isin(selected_depts)]
support_f = support[support["department"].isin(selected_depts)]
comments_f = comments[comments["department"].isin(selected_depts)]
trust_f = trust[trust["department"].isin(selected_depts)]

# ----------------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------------
st.title("📊 AI in the Classroom — Principal Dashboard")
st.caption(
    "Demo dashboard built with Python (pandas, NumPy, scikit-learn, VADER NLP, Plotly, "
    "Streamlit) showing how AI-adoption data, achievement/engagement signals, planning "
    "time, teacher needs, trust/readiness, professional learning, and resource priorities "
    "could be surfaced for a school principal."
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Overview",
    "🎯 Achievement & Engagement",
    "⏱️ Teacher Planning Time",
    "🙋 Teacher Needs",
    "🤝 Trust & Readiness",
    "📚 Professional Learning",
    "🧰 Classroom Support Requests",
])

# ============================================================================
# TAB 1 — OVERVIEW
# Principal question: How is AI being used across classrooms?
# Method: frequency counts, adoption rate, trend summaries
# ============================================================================
with tab1:
    st.subheader("How Is AI Being Used Across Classrooms?")

    col1, col2, col3, col4, col5 = st.columns(5)
    avg_usage = usage_f["ai_usage_rate"].mean()
    total_sessions = usage_f["ai_sessions"].sum()
    avg_engagement = usage_f["engagement_score"].mean()
    avg_achievement = usage_f["achievement_delta_pct"].mean()
    open_requests = (support_f["status"] == "Open").sum()

    col1.metric("Adoption Rate", f"{avg_usage:.0%}")
    col2.metric("Total AI Sessions", f"{int(total_sessions):,}")
    col3.metric("Avg. Engagement Score", f"{avg_engagement:.1f} / 100")
    col4.metric("Avg. Achievement Δ", f"{avg_achievement:+.1f}%")
    col5.metric("Open Support Requests", int(open_requests))

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weekly Adoption Trend (by Department)**")
        weekly_dept = usage_f.groupby(["week_number", "department"], as_index=False)["ai_usage_rate"].mean()
        fig = px.line(weekly_dept, x="week_number", y="ai_usage_rate", color="department",
                       labels={"week_number": "Week", "ai_usage_rate": "Avg. Usage Rate"})
        fig.update_layout(yaxis_tickformat=".0%", height=380, legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**AI Session Frequency by Department**")
        freq = usage_f.groupby("department", as_index=False)["ai_sessions"].sum().sort_values("ai_sessions", ascending=False)
        fig2 = px.bar(freq, x="department", y="ai_sessions", color="department")
        fig2.update_layout(height=380, showlegend=False, xaxis_title="", yaxis_title="Total AI Sessions")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Adoption Rate Cards — by Department**")
    cards = usage_f.groupby("department", as_index=False)["ai_usage_rate"].mean().sort_values("ai_usage_rate", ascending=False)
    card_cols = st.columns(len(cards)) if len(cards) > 0 else [st]
    for col, (_, row) in zip(card_cols, cards.iterrows()):
        col.metric(row["department"], f"{row['ai_usage_rate']:.0%}")

# ============================================================================
# TAB 2 — ACHIEVEMENT & ENGAGEMENT
# Principal question: Is AI helping student outcomes?
# Method: correlation, before/after comparison, engagement scoring
# ============================================================================
with tab2:
    st.subheader("Is AI Helping Student Outcomes?")
    st.caption("Correlational view only — useful for spotting patterns, not proving causation.")

    metric_choice = st.radio("Compare AI usage against:", ["Engagement Score", "Achievement Δ%"], horizontal=True)
    y_col = "engagement_score" if metric_choice == "Engagement Score" else "achievement_delta_pct"

    fig = px.scatter(
        usage_f, x="ai_usage_rate", y=y_col, color="department",
        trendline="ols", opacity=0.55,
        labels={"ai_usage_rate": "AI Usage Rate", y_col: metric_choice},
    )
    fig.update_layout(xaxis_tickformat=".0%", height=440)
    st.plotly_chart(fig, use_container_width=True)

    corr = usage_f["ai_usage_rate"].corr(usage_f[y_col])
    st.info(f"Correlation (AI usage rate vs. {metric_choice}): **r = {corr:.2f}** "
            f"({'weak' if abs(corr) < 0.3 else 'moderate' if abs(corr) < 0.6 else 'strong'} relationship in this mock data)")

    st.markdown("---")
    st.markdown("**Before / After Comparison — Low- vs. High-AI-Usage Weeks**")
    st.caption("Splits teacher-weeks into low-usage vs. high-usage halves (median split) as a simple before/after proxy.")
    median_usage = usage_f["ai_usage_rate"].median()
    usage_f_split = usage_f.copy()
    usage_f_split["usage_group"] = np.where(usage_f_split["ai_usage_rate"] >= median_usage, "High AI Use", "Low AI Use")
    comp = usage_f_split.groupby("usage_group", as_index=False).agg(
        avg_engagement=("engagement_score", "mean"),
        avg_achievement=("achievement_delta_pct", "mean"),
    ).round(2)

    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.bar(comp, x="usage_group", y="avg_engagement", color="usage_group",
                       color_discrete_map={"Low AI Use": "#9E9E9E", "High AI Use": "#1f77b4"})
        fig2.update_layout(height=320, showlegend=False, xaxis_title="", yaxis_title="Avg. Engagement Score")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.bar(comp, x="usage_group", y="avg_achievement", color="usage_group",
                       color_discrete_map={"Low AI Use": "#9E9E9E", "High AI Use": "#2E7D32"})
        fig3.update_layout(height=320, showlegend=False, xaxis_title="", yaxis_title="Avg. Achievement Δ%")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Engagement Heatmap — Department × Week**")
    heat = usage_f.pivot_table(index="department", columns="week_number", values="engagement_score", aggfunc="mean")
    fig4 = px.imshow(heat, color_continuous_scale="YlGnBu", aspect="auto",
                      labels=dict(x="Week", y="Department", color="Engagement"))
    fig4.update_layout(height=320)
    st.plotly_chart(fig4, use_container_width=True)

# ============================================================================
# TAB 3 — TEACHER PLANNING TIME
# Principal question: Is AI reducing planning workload?
# Method: min/max/mean, boxplots, outlier detection
# ============================================================================
with tab3:
    st.subheader("Is AI Reducing Planning Workload?")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min Planning Hrs/Wk", f"{usage_f['planning_hours_actual'].min():.1f}")
    c2.metric("Max Planning Hrs/Wk", f"{usage_f['planning_hours_actual'].max():.1f}")
    c3.metric("Mean Planning Hrs/Wk", f"{usage_f['planning_hours_actual'].mean():.1f}")
    c4.metric("Mean Time Saved/Wk", f"{usage_f['time_saved_hours'].mean():.1f} hrs")

    st.markdown("**Boxplot — Planning Hours by Department**")
    fig = px.box(usage_f, x="department", y="planning_hours_actual", color="department", points="outliers")
    fig.update_layout(height=420, showlegend=False, xaxis_title="", yaxis_title="Planning Hours / Week")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Outlier Detection — Teachers with Unusually High/Low Planning Time**")
    st.caption("Outliers flagged using the IQR rule (1.5×IQR beyond Q1/Q3) on each teacher's average planning hours.")
    teacher_avg = usage_f.groupby("teacher_id", as_index=False)["planning_hours_actual"].mean()
    q1, q3 = teacher_avg["planning_hours_actual"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    teacher_avg["flag"] = np.where(
        teacher_avg["planning_hours_actual"] > upper, "High outlier",
        np.where(teacher_avg["planning_hours_actual"] < lower, "Low outlier", "Typical")
    )
    outliers = teacher_avg[teacher_avg["flag"] != "Typical"].merge(
        teachers[["teacher_id", "department"]], on="teacher_id"
    ).sort_values("planning_hours_actual", ascending=False)

    if outliers.empty:
        st.success("No outliers detected in the current filter — planning times are fairly consistent.")
    else:
        st.dataframe(
            outliers[["teacher_id", "department", "planning_hours_actual", "flag"]].round(2),
            use_container_width=True, hide_index=True,
        )

    with st.expander("See department summary table (min / max / mean)"):
        agg = usage_f.groupby("department", as_index=False).agg(
            min_hrs=("planning_hours_actual", "min"),
            max_hrs=("planning_hours_actual", "max"),
            mean_hrs=("planning_hours_actual", "mean"),
            mean_saved=("time_saved_hours", "mean"),
        ).round(2)
        st.dataframe(agg, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 4 — TEACHER NEEDS
# Principal question: What support do teachers ask for?
# Method: text mining, topic frequency, sentiment
# ============================================================================
with tab4:
    st.subheader("What Support Do Teachers Ask For?")
    st.caption("Combines structured support-request tickets with text-mined keywords from open-ended comments.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Support Request Topics (Frequency)**")
        type_counts = support_f["request_type"].value_counts().reset_index()
        type_counts.columns = ["Request Type", "Count"]
        fig = px.bar(type_counts, x="Count", y="Request Type", orientation="h")
        fig.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Top Keywords in Teacher Comments (Text Mining)**")
        kw = top_keywords(comments_f["comment_text"], n=12)
        if kw:
            kw_df = pd.DataFrame(kw, columns=["Keyword", "Frequency"])
            fig2 = px.bar(kw_df, x="Frequency", y="Keyword", orientation="h", color_discrete_sequence=["#6A5ACD"])
            fig2.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No comments in this filter.")

    st.markdown("**Sentiment of Need-Related Comments**")
    sent_by_dept = comments_f.groupby("department", as_index=False)["compound"].mean().sort_values("compound")
    fig3 = px.bar(sent_by_dept, x="compound", y="department", orientation="h", color="compound",
                  color_continuous_scale="RdYlGn", range_color=[-0.5, 0.5])
    fig3.update_layout(height=300, coloraxis_showscale=False, xaxis_title="Avg. Sentiment (compound)", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("See all open support requests"):
        st.dataframe(
            support_f[support_f["status"] != "Resolved"][
                ["teacher_id", "department", "request_type", "urgency", "status", "date_submitted"]
            ].sort_values("date_submitted", ascending=False),
            use_container_width=True, hide_index=True,
        )

# ============================================================================
# TAB 5 — TRUST & READINESS
# Principal question: Are teachers comfortable with AI policy and expectations?
# Method: sentiment analysis, Likert-scale scoring
# ============================================================================
with tab5:
    st.subheader("Are Teachers Comfortable With AI Policy & Expectations?")
    st.caption(
        "Combines a 5-point Likert survey (policy clarity, assessment confidence, comfort with "
        "tools, admin support, pedagogy understanding) with VADER sentiment scoring of open-ended comments."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg. Trust Index", f"{trust_f['trust_index'].mean():.2f} / 5")
    c2.metric("Avg. Comment Sentiment", f"{comments_f['compound'].mean():+.2f}", help="-1 (very negative) to +1 (very positive)")
    pos_pct = (comments_f["sentiment_label"] == "Positive").mean() if len(comments_f) else 0
    c3.metric("% Positive Comments", f"{pos_pct:.0%}")

    st.markdown("**Likert Scores — Average by Department**")
    likert_cols = ["policy_clarity", "assessment_confidence", "comfort_with_ai_tools",
                   "perceived_admin_support", "pedagogy_understanding"]
    likert_labels = {
        "policy_clarity": "Policy Clarity",
        "assessment_confidence": "Assessment Confidence",
        "comfort_with_ai_tools": "Comfort with AI Tools",
        "perceived_admin_support": "Perceived Admin Support",
        "pedagogy_understanding": "Pedagogy Understanding",
    }
    radar_data = trust_f.groupby("department")[likert_cols].mean().reset_index()

    fig = go.Figure()
    for _, row in radar_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[c] for c in likert_cols] + [row[likert_cols[0]]],
            theta=[likert_labels[c] for c in likert_cols] + [likert_labels[likert_cols[0]]],
            fill="toself", name=row["department"], opacity=0.5,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1, 5])), height=480)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sentiment Trend Over the Semester**")
        weekly_sent = comments_f.groupby(pd.Grouper(key="week_start", freq="W"))["compound"].mean().reset_index()
        fig2 = px.line(weekly_sent, x="week_start", y="compound", markers=True)
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(height=350, yaxis_title="Avg. Compound Sentiment", xaxis_title="Week")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("**Sample Comments by Sentiment**")
        label_pick = st.selectbox("Show examples of:", ["Negative", "Neutral", "Positive"])
        examples = comments_f[comments_f["sentiment_label"] == label_pick][["teacher_id", "comment_text", "compound"]]
        if examples.empty:
            st.info("No comments in this category for the current filters.")
        else:
            st.dataframe(examples.sample(min(5, len(examples))).reset_index(drop=True), use_container_width=True, hide_index=True)

    with st.expander("See raw Likert-scale data by teacher"):
        st.dataframe(
            trust_f[["teacher_id", "department"] + likert_cols + ["trust_index"]].rename(columns=likert_labels),
            use_container_width=True, hide_index=True,
        )

# ============================================================================
# TAB 6 — PROFESSIONAL LEARNING
# Principal question: Who needs AI training or micro-credentials?
# Method: recommendation logic, clustering
# ============================================================================
with tab6:
    st.subheader("Who Needs AI Training or Micro-Credentials?")
    st.caption(
        "K-means clustering groups teachers into readiness profiles based on trust index, AI usage "
        "rate, credentials completed, and goal progress — then a simple recommendation rule suggests "
        "next steps for each group."
    )

    # --- Build feature set & cluster ---
    usage_avg = usage.groupby("teacher_id", as_index=False)["ai_usage_rate"].mean()
    features = (
        trust[["teacher_id", "department", "trust_index"]]
        .merge(usage_avg, on="teacher_id")
        .merge(pd_records[["teacher_id", "n_completed", "professional_goal_progress_pct"]], on="teacher_id")
    )
    features = features[features["department"].isin(selected_depts)].reset_index(drop=True)

    if len(features) < 6:
        st.info("Not enough teachers in this filter to run clustering (need at least ~6). Try selecting more departments.")
    else:
        feature_cols = ["trust_index", "ai_usage_rate", "n_completed", "professional_goal_progress_pct"]
        X = StandardScaler().fit_transform(features[feature_cols])
        k = st.slider("Number of readiness groups (k)", 2, 5, 3)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        features["cluster"] = km.fit_predict(X)

        # Label clusters by readiness level using trust_index + ai_usage_rate as the proxy
        cluster_means = features.groupby("cluster")[["trust_index", "ai_usage_rate"]].mean()
        cluster_means["composite"] = cluster_means["trust_index"].rank() + cluster_means["ai_usage_rate"].rank()
        ranked = cluster_means.sort_values("composite").index.tolist()
        readiness_names = ["🔴 Needs Foundational Support", "🟡 Building Confidence", "🟢 Ready to Lead / Mentor"]
        name_map = {cl: readiness_names[i] if i < len(readiness_names) else f"Group {i}" for i, cl in enumerate(ranked)}
        features["readiness_group"] = features["cluster"].map(name_map)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.scatter(
                features, x="ai_usage_rate", y="trust_index", color="readiness_group",
                size="professional_goal_progress_pct", hover_data=["teacher_id", "department", "n_completed"],
                labels={"ai_usage_rate": "AI Usage Rate", "trust_index": "Trust Index"},
            )
            fig.update_layout(xaxis_tickformat=".0%", height=420)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Group sizes**")
            st.dataframe(features["readiness_group"].value_counts().rename("Teachers"), use_container_width=True)

        st.markdown("**Recommended Next Step — by Readiness Group**")
        recommendations = {
            "🔴 Needs Foundational Support": "Pair with an instructional coach; prioritize 1:1 onboarding to AI tools before group PD.",
            "🟡 Building Confidence": "Enroll in a hands-on micro-credential (e.g. Prompt Engineering for Lesson Design); peer observation.",
            "🟢 Ready to Lead / Mentor": "Invite to lead a PD session or mentor colleagues; consider for pilot programs.",
        }
        for grp in features["readiness_group"].unique():
            st.markdown(f"- **{grp}**: {recommendations.get(grp, 'Review individually.')}")

        with st.expander("See teacher-level cluster assignments"):
            st.dataframe(
                features[["teacher_id", "department", "trust_index", "ai_usage_rate", "n_completed",
                          "professional_goal_progress_pct", "readiness_group"]].round(2),
                use_container_width=True, hide_index=True,
            )

    st.markdown("---")
    st.markdown("**Most Desired Topics (Unmet Demand)**")
    desired_series = pd_f["credentials_desired"].str.split("; ").explode()
    desired_series = desired_series[desired_series != "None"]
    if desired_series.empty:
        st.info("No outstanding requests in this filter.")
    else:
        counts = desired_series.value_counts().reset_index()
        counts.columns = ["Topic", "Teachers Requesting"]
        fig = px.bar(counts, x="Teachers Requesting", y="Topic", orientation="h")
        fig.update_layout(height=320, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 7 — CLASSROOM SUPPORT REQUESTS
# Principal question: Where should principals allocate resources?
# Method: priority scoring
# ============================================================================
with tab7:
    st.subheader("Where Should Principals Allocate Resources?")
    st.caption(
        "Each request gets a 0–100 **priority score** combining urgency, days open, and how "
        "frequently that request type recurs across the building — so principals can triage "
        "instead of reading every ticket individually."
    )

    active = support_f[support_f["status"] != "Resolved"].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Requests", len(active))
    c2.metric("Avg. Priority Score", f"{active['priority_score'].mean():.1f}" if len(active) else "—")
    c3.metric("Highest Priority", f"{active['priority_score'].max():.1f}" if len(active) else "—")

    st.markdown("**Top Priority Requests Right Now**")
    if active.empty:
        st.success("No active requests in this filter — everything is resolved.")
    else:
        top = active.sort_values("priority_score", ascending=False).head(10)
        fig = px.bar(
            top.sort_values("priority_score"), x="priority_score", y="request_id",
            color="urgency", orientation="h",
            color_discrete_map={"Low": "#9E9E9E", "Medium": "#F9A825", "High": "#C62828"},
            hover_data=["teacher_id", "department", "request_type", "days_open"],
        )
        fig.update_layout(height=420, xaxis_title="Priority Score (0–100)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            top[["request_id", "teacher_id", "department", "request_type", "urgency", "days_open", "priority_score"]],
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.markdown("**Resource Allocation View — Priority by Request Type**")
    by_type = active.groupby("request_type", as_index=False)["priority_score"].mean().sort_values("priority_score", ascending=False)
    if not by_type.empty:
        fig2 = px.bar(by_type, x="priority_score", y="request_type", orientation="h", color_discrete_sequence=["#1f77b4"])
        fig2.update_layout(height=350, yaxis={"categoryorder": "total ascending"}, xaxis_title="Avg. Priority Score")
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "💡 **Methodology note:** priority_score = 2×urgency_weight + recency_weight (days open, capped) "
        "+ frequency_weight (how common that request type is among active tickets), rescaled to 0–100. "
        "Weights are illustrative — in a real deployment these would be calibrated with the principal's input."
    )
