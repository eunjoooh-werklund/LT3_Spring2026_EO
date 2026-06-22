"""
app.py — AI in the Classroom: Principal Dashboard (mock data demo)
====================================================================
A 7-page Streamlit dashboard with sidebar navigation demonstrating how
a K-12 principal could monitor teacher AI adoption and its connection to
student achievement, engagement, planning workload, teacher needs,
mediating factors, professional learning, and resource-allocation priorities.

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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI in the Classroom — Principal Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme colors
# ---------------------------------------------------------------------------
PRIMARY = "#1B2A4A"
ACCENT = "#3B82F6"
ACCENT_LIGHT = "#EFF6FF"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
SURFACE = "#F8FAFC"
TEXT_MUTED = "#64748B"

# ---------------------------------------------------------------------------
# Custom CSS for app-like look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #1B2A4A;
        min-width: 280px;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption p {
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #FFFFFF !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
        color: #FFFFFF !important;
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px 12px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {
        color: #64748B !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1B2A4A !important;
        font-weight: 700 !important;
    }

    /* Page header */
    .page-header {
        background: linear-gradient(135deg, #1B2A4A 0%, #2D4A7A 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .page-header h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }
    .page-header p {
        margin: 0.3rem 0 0 0;
        color: #94A3B8;
        font-size: 0.9rem;
    }

    /* Clean expander styling */
    details {
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }

    /* Info/success/warning boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    /* Hide default hamburger and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    generate_all(DATA_DIR)
    teachers = pd.read_csv(f"{DATA_DIR}/teachers.csv")
    usage = pd.read_csv(f"{DATA_DIR}/weekly_usage.csv", parse_dates=["week_start"])
    pd_records = pd.read_csv(f"{DATA_DIR}/professional_learning.csv")
    support = pd.read_csv(f"{DATA_DIR}/support_requests.csv", parse_dates=["date_submitted"])
    comments = pd.read_csv(f"{DATA_DIR}/comments.csv", parse_dates=["week_start"])
    trust = pd.read_csv(f"{DATA_DIR}/trust_readiness.csv")
    return teachers, usage, pd_records, support, comments, trust


@st.cache_data
def score_sentiment(comments_df: pd.DataFrame):
    analyzer = SentimentIntensityAnalyzer()
    scores = comments_df["comment_text"].apply(analyzer.polarity_scores)
    comments_df = comments_df.copy()
    comments_df["compound"] = scores.apply(lambda s: s["compound"])
    comments_df["sentiment_label"] = pd.cut(
        comments_df["compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["Negative", "Neutral", "Positive"],
    )
    return comments_df


STOPWORDS = set("""
the a an and or but if is are was were be been being to of in on for with
as at by from this that it its i my we our you your they their he she
not no so it's i'm i've don't doesn't didn't isn't more most still very
have has had do does did just about than then them than feel feels felt
""".split())


def top_keywords(texts: pd.Series, n=12):
    counter = Counter()
    for t in texts.dropna():
        words = re.findall(r"[a-zA-Z']+", t.lower())
        counter.update(w for w in words if w not in STOPWORDS and len(w) > 2)
    return counter.most_common(n)


def page_header(title: str, subtitle: str = ""):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="page-header"><h2>{title}</h2>{sub}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load & prepare data
# ---------------------------------------------------------------------------
teachers, usage, pd_records, support, comments, trust = load_data()
comments = score_sentiment(comments)
usage = usage.merge(teachers[["teacher_id", "years_experience"]], on="teacher_id", how="left")

# ---------------------------------------------------------------------------
# Sidebar — Navigation + Filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 AI in the Classroom")
    st.caption("Principal Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🎯 Achievement & Engagement",
            "⏱️ Planning Time",
            "🙋 Teacher Needs",
            "🤝 Mediating Factors",
            "📚 Professional Learning",
            "🧰 Support Requests",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("##### 🎛️ Filters")

    all_depts = sorted(teachers["department"].unique())
    selected_depts = st.multiselect("Department", all_depts, default=all_depts)

    min_week, max_week = int(usage["week_number"].min()), int(usage["week_number"].max())
    week_range = st.slider("Week of semester", min_week, max_week, (min_week, max_week))

    st.markdown("---")
    st.caption(
        "⚠️ All data is **simulated** for demonstration purposes. "
        "No real student or teacher data is used."
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

# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
if page == "🏠 Overview":
    page_header("How Is AI Being Used Across Classrooms?",
                "Frequency counts, adoption rate, and trend summaries")

    col1, col2, col3, col4, col5 = st.columns(5)
    avg_usage = usage_f["ai_usage_rate"].mean()
    total_sessions = usage_f["ai_sessions"].sum()
    avg_engagement = usage_f["engagement_score"].mean()
    avg_achievement = usage_f["achievement_delta_pct"].mean()
    open_requests = (support_f["status"] == "Open").sum()

    col1.metric("Adoption Rate", f"{avg_usage:.0%}")
    col2.metric("Total AI Sessions", f"{int(total_sessions):,}")
    col3.metric("Avg. Engagement", f"{avg_engagement:.1f}/100")
    col4.metric("Achievement Δ", f"{avg_achievement:+.1f}%")
    col5.metric("Open Requests", int(open_requests))

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weekly Adoption Trend**")
        weekly_dept = usage_f.groupby(["week_number", "department"], as_index=False)["ai_usage_rate"].mean()
        fig = px.line(weekly_dept, x="week_number", y="ai_usage_rate", color="department",
                      labels={"week_number": "Week", "ai_usage_rate": "Usage Rate"})
        fig.update_layout(yaxis_tickformat=".0%", height=380, legend_title="",
                          plot_bgcolor="white", paper_bgcolor="white")
        fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**AI Sessions by Department**")
        freq = usage_f.groupby("department", as_index=False)["ai_sessions"].sum().sort_values("ai_sessions", ascending=False)
        fig2 = px.bar(freq, x="department", y="ai_sessions", color="department",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=380, showlegend=False, xaxis_title="", yaxis_title="Total Sessions",
                           plot_bgcolor="white", paper_bgcolor="white")
        fig2.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Department Adoption Cards**")
    cards = usage_f.groupby("department", as_index=False)["ai_usage_rate"].mean().sort_values("ai_usage_rate", ascending=False)
    card_cols = st.columns(len(cards)) if len(cards) > 0 else [st]
    for col, (_, row) in zip(card_cols, cards.iterrows()):
        col.metric(row["department"], f"{row['ai_usage_rate']:.0%}")

# ===========================================================================
# PAGE 2 — ACHIEVEMENT & ENGAGEMENT
# ===========================================================================
elif page == "🎯 Achievement & Engagement":
    page_header("Is AI Helping Student Outcomes?",
                "Correlational view — useful for spotting patterns, not proving causation.")

    metric_choice = st.radio("Compare AI usage against:", ["Engagement Score", "Achievement Δ%"], horizontal=True)
    y_col = "engagement_score" if metric_choice == "Engagement Score" else "achievement_delta_pct"

    fig = px.scatter(
        usage_f, x="ai_usage_rate", y=y_col, color="department",
        trendline="ols", opacity=0.55,
        labels={"ai_usage_rate": "AI Usage Rate", y_col: metric_choice},
    )
    fig.update_layout(xaxis_tickformat=".0%", height=440, plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig, use_container_width=True)

    corr = usage_f["ai_usage_rate"].corr(usage_f[y_col])
    strength = 'weak' if abs(corr) < 0.3 else 'moderate' if abs(corr) < 0.6 else 'strong'
    st.info(f"Correlation: **r = {corr:.2f}** ({strength} relationship in this mock data)")

    st.markdown("---")
    st.markdown("**Before / After — Low vs. High AI Usage**")
    st.caption("Median-split comparison as a simple before/after proxy.")
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
                      color_discrete_map={"Low AI Use": "#94A3B8", "High AI Use": ACCENT})
        fig2.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="Avg. Engagement",
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.bar(comp, x="usage_group", y="avg_achievement", color="usage_group",
                      color_discrete_map={"Low AI Use": "#94A3B8", "High AI Use": SUCCESS})
        fig3.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="Avg. Achievement Δ%",
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Engagement Heatmap — Department x Week**")
    heat = usage_f.pivot_table(index="department", columns="week_number", values="engagement_score", aggfunc="mean")
    fig4 = px.imshow(heat, color_continuous_scale="Blues", aspect="auto",
                     labels=dict(x="Week", y="Department", color="Engagement"))
    fig4.update_layout(height=300)
    st.plotly_chart(fig4, use_container_width=True)

# ===========================================================================
# PAGE 3 — TEACHER PLANNING TIME
# ===========================================================================
elif page == "⏱️ Planning Time":
    page_header("Is AI Reducing Planning Workload?",
                "Min/max/mean analysis, boxplots, and IQR outlier detection")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min Hrs/Wk", f"{usage_f['planning_hours_actual'].min():.1f}")
    c2.metric("Max Hrs/Wk", f"{usage_f['planning_hours_actual'].max():.1f}")
    c3.metric("Mean Hrs/Wk", f"{usage_f['planning_hours_actual'].mean():.1f}")
    c4.metric("Avg. Saved/Wk", f"{usage_f['time_saved_hours'].mean():.1f} hrs")

    st.markdown("**Planning Hours by Department**")
    fig = px.box(usage_f, x="department", y="planning_hours_actual", color="department", points="outliers",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400, showlegend=False, xaxis_title="", yaxis_title="Planning Hours / Week",
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**🔎 Individual Teacher Drill-Down**")
    teacher_options = sorted(usage_f["teacher_id"].unique())
    if teacher_options:
        picked_teacher = st.selectbox("Select a teacher:", teacher_options, key="planning_teacher_pick")
        t_data = usage_f[usage_f["teacher_id"] == picked_teacher].sort_values("week_number")
        t_dept = teachers.loc[teachers["teacher_id"] == picked_teacher, "department"].values[0]

        tc1, tc2, tc3 = st.columns(3)
        tc1.metric(f"{picked_teacher}", t_dept)
        tc2.metric("Avg Planning Hrs", f"{t_data['planning_hours_actual'].mean():.1f}")
        tc3.metric("Avg Time Saved", f"{t_data['time_saved_hours'].mean():.1f} hrs")

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=t_data["week_number"], y=t_data["planning_hours_baseline"],
                                   name="Baseline (no AI)", line=dict(dash="dot", color="#94A3B8")))
        fig_t.add_trace(go.Scatter(x=t_data["week_number"], y=t_data["planning_hours_actual"],
                                   name="Actual (with AI)", line=dict(color=ACCENT, width=3)))
        fig_t.update_layout(height=300, xaxis_title="Week", yaxis_title="Hours", legend_title="",
                            plot_bgcolor="white", paper_bgcolor="white")
        fig_t.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        fig_t.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("---")
    st.markdown("**Outlier Detection (IQR Rule)**")
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
        st.success("No outliers detected — planning times are fairly consistent.")
    else:
        st.dataframe(
            outliers[["teacher_id", "department", "planning_hours_actual", "flag"]].round(2),
            use_container_width=True, hide_index=True,
        )

    with st.expander("Department summary table"):
        agg = usage_f.groupby("department", as_index=False).agg(
            min_hrs=("planning_hours_actual", "min"),
            max_hrs=("planning_hours_actual", "max"),
            mean_hrs=("planning_hours_actual", "mean"),
            mean_saved=("time_saved_hours", "mean"),
        ).round(2)
        st.dataframe(agg, use_container_width=True, hide_index=True)

# ===========================================================================
# PAGE 4 — TEACHER NEEDS
# ===========================================================================
elif page == "🙋 Teacher Needs":
    page_header("What Support Do Teachers Ask For?",
                "Text mining, topic frequency, and sentiment analysis of teacher feedback")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Support Request Topics**")
        type_counts = support_f["request_type"].value_counts().reset_index()
        type_counts.columns = ["Request Type", "Count"]
        fig = px.bar(type_counts, x="Count", y="Request Type", orientation="h",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(height=380, yaxis={"categoryorder": "total ascending"},
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Top Keywords in Comments (Text Mining)**")
        kw = top_keywords(comments_f["comment_text"], n=12)
        if kw:
            kw_df = pd.DataFrame(kw, columns=["Keyword", "Frequency"])
            fig2 = px.bar(kw_df, x="Frequency", y="Keyword", orientation="h",
                          color_discrete_sequence=["#8B5CF6"])
            fig2.update_layout(height=380, yaxis={"categoryorder": "total ascending"},
                               plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No comments in this filter.")

    st.markdown("**Sentiment by Department**")
    sent_by_dept = comments_f.groupby("department", as_index=False)["compound"].mean().sort_values("compound")
    fig3 = px.bar(sent_by_dept, x="compound", y="department", orientation="h", color="compound",
                  color_continuous_scale="RdYlGn", range_color=[-0.5, 0.5])
    fig3.update_layout(height=280, coloraxis_showscale=False, xaxis_title="Avg. Sentiment", yaxis_title="",
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("All open support requests"):
        st.dataframe(
            support_f[support_f["status"] != "Resolved"][
                ["teacher_id", "department", "request_type", "urgency", "status", "date_submitted"]
            ].sort_values("date_submitted", ascending=False),
            use_container_width=True, hide_index=True,
        )

# ===========================================================================
# PAGE 5 — MEDIATING FACTORS FOR AI INTEGRATION
# ===========================================================================
elif page == "🤝 Mediating Factors":
    page_header("What Do Teachers Need for AI Integration to Succeed?",
                "Six literature-grounded mediating factors scored 0–100. "
                "Trust in AI Policy is the central factor identified in the leadership literature.")

    FACTOR_COLS = ["trust_in_ai_policy", "understanding_expectations", "institutional_support",
                   "task_technology_fit", "ai_assessment_confidence", "professional_learning_access"]
    FACTOR_LABELS = {
        "trust_in_ai_policy": "Trust in AI Policy",
        "understanding_expectations": "Understanding of Expectations",
        "institutional_support": "Institutional Support",
        "task_technology_fit": "Task-Technology Fit",
        "ai_assessment_confidence": "AI Assessment Confidence",
        "professional_learning_access": "Professional Learning Access",
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Trust in AI Policy", f"{trust_f['trust_in_ai_policy'].mean():.0f}%",
              help="Central mediating factor from the literature review.")
    c2.metric("Avg. Sentiment", f"{comments_f['compound'].mean():+.2f}",
              help="-1 (very negative) to +1 (very positive)")
    pos_pct = (comments_f["sentiment_label"] == "Positive").mean() if len(comments_f) else 0
    c3.metric("% Positive Comments", f"{pos_pct:.0%}")

    st.markdown("**Mediating Factors — Average Scores**")
    factor_means = trust_f[FACTOR_COLS].mean().rename(FACTOR_LABELS).sort_values(ascending=True)
    bar_colors = [DANGER if name == "Trust in AI Policy" else "#94A3B8" for name in factor_means.index]
    fig = go.Figure(go.Bar(
        x=factor_means.values, y=factor_means.index, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.0f}%" for v in factor_means.values], textposition="outside",
    ))
    fig.update_layout(height=360, xaxis_title="Score (0–100)", yaxis_title="",
                      xaxis_range=[0, 100], plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=60))
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "💡 The literature review identifies trust in AI policy as the mediator "
        "that enables the others — low-trust teachers underuse even strong tools."
    )

    st.markdown("---")
    st.markdown("**Scores by Department**")
    dept_factors = trust_f.groupby("department")[FACTOR_COLS].mean().rename(columns=FACTOR_LABELS)
    fig_dept = go.Figure()
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EF4444", "#06B6D4"]
    for i, col in enumerate(dept_factors.columns):
        fig_dept.add_trace(go.Bar(name=col, x=dept_factors.index, y=dept_factors[col],
                                  marker_color=colors[i % len(colors)]))
    fig_dept.update_layout(barmode="group", height=380, yaxis_title="Score (0–100)", legend_title="",
                           plot_bgcolor="white", paper_bgcolor="white")
    fig_dept.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig_dept, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sentiment Trend Over Semester**")
        weekly_sent = comments_f.groupby(pd.Grouper(key="week_start", freq="W"))["compound"].mean().reset_index()
        fig2 = px.line(weekly_sent, x="week_start", y="compound", markers=True,
                       color_discrete_sequence=[ACCENT])
        fig2.add_hline(y=0, line_dash="dash", line_color="#CBD5E1")
        fig2.update_layout(height=320, yaxis_title="Avg. Sentiment", xaxis_title="Week",
                           plot_bgcolor="white", paper_bgcolor="white")
        fig2.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        fig2.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("**Sample Comments**")
        label_pick = st.selectbox("Show:", ["Negative", "Neutral", "Positive"])
        examples = comments_f[comments_f["sentiment_label"] == label_pick][["teacher_id", "comment_text", "compound"]]
        if examples.empty:
            st.info("No comments in this category.")
        else:
            st.dataframe(examples.sample(min(5, len(examples))).reset_index(drop=True),
                         use_container_width=True, hide_index=True)

    with st.expander("Raw factor scores by teacher"):
        st.dataframe(
            trust_f[["teacher_id", "department"] + FACTOR_COLS + ["trust_index"]].rename(columns=FACTOR_LABELS),
            use_container_width=True, hide_index=True,
        )

# ===========================================================================
# PAGE 6 — PROFESSIONAL LEARNING
# ===========================================================================
elif page == "📚 Professional Learning":
    page_header("Who Needs AI Training or Micro-Credentials?",
                "K-means clustering groups teachers into readiness profiles, "
                "then a recommendation rule suggests next steps for each group.")

    usage_avg = usage.groupby("teacher_id", as_index=False)["ai_usage_rate"].mean()
    features = (
        trust[["teacher_id", "department", "trust_index"]]
        .merge(usage_avg, on="teacher_id")
        .merge(pd_records[["teacher_id", "n_completed", "professional_goal_progress_pct"]], on="teacher_id")
    )
    features = features[features["department"].isin(selected_depts)].reset_index(drop=True)

    if len(features) < 6:
        st.info("Not enough teachers to run clustering (need >= 6). Select more departments.")
    else:
        feature_cols = ["trust_index", "ai_usage_rate", "n_completed", "professional_goal_progress_pct"]
        X = StandardScaler().fit_transform(features[feature_cols])
        k = st.slider("Number of readiness groups (k)", 2, 5, 3)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        features["cluster"] = km.fit_predict(X)

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
                size="professional_goal_progress_pct",
                hover_data=["teacher_id", "department", "n_completed"],
                labels={"ai_usage_rate": "AI Usage Rate", "trust_index": "Trust Index"},
                color_discrete_sequence=[DANGER, WARNING, SUCCESS],
            )
            fig.update_layout(xaxis_tickformat=".0%", height=420,
                              plot_bgcolor="white", paper_bgcolor="white")
            fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
            fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Group Sizes**")
            st.dataframe(features["readiness_group"].value_counts().rename("Teachers"),
                         use_container_width=True)

        st.markdown("**Recommendations**")
        recommendations = {
            "🔴 Needs Foundational Support": "Pair with an instructional coach; prioritize 1:1 onboarding before group PD.",
            "🟡 Building Confidence": "Enroll in a micro-credential (e.g. Prompt Engineering for Lesson Design); peer observation.",
            "🟢 Ready to Lead / Mentor": "Invite to lead a PD session or mentor colleagues; consider for pilot programs.",
        }
        for grp in sorted(features["readiness_group"].unique()):
            st.markdown(f"- **{grp}**: {recommendations.get(grp, 'Review individually.')}")

        with st.expander("Teacher-level assignments"):
            st.dataframe(
                features[["teacher_id", "department", "trust_index", "ai_usage_rate", "n_completed",
                          "professional_goal_progress_pct", "readiness_group"]].round(2),
                use_container_width=True, hide_index=True,
            )

    st.markdown("---")
    st.markdown("**Most Desired PD Topics (Unmet Demand)**")
    desired_series = pd_f["credentials_desired"].str.split("; ").explode()
    desired_series = desired_series[desired_series != "None"]
    if desired_series.empty:
        st.info("No outstanding requests in this filter.")
    else:
        counts = desired_series.value_counts().reset_index()
        counts.columns = ["Topic", "Teachers Requesting"]
        fig = px.bar(counts, x="Teachers Requesting", y="Topic", orientation="h",
                     color_discrete_sequence=["#06B6D4"])
        fig.update_layout(height=300, yaxis={"categoryorder": "total ascending"},
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# PAGE 7 — CLASSROOM SUPPORT REQUESTS
# ===========================================================================
elif page == "🧰 Support Requests":
    page_header("Where Should Principals Allocate Resources?",
                "Priority scoring: urgency + days open + request frequency → 0–100 score for triage")

    active = support_f[support_f["status"] != "Resolved"].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Requests", len(active))
    c2.metric("Avg. Priority", f"{active['priority_score'].mean():.1f}" if len(active) else "—")
    c3.metric("Highest Priority", f"{active['priority_score'].max():.1f}" if len(active) else "—")

    st.markdown("**Top Priority Requests**")
    if active.empty:
        st.success("No active requests — everything is resolved.")
    else:
        top = active.sort_values("priority_score", ascending=False).head(10)
        fig = px.bar(
            top.sort_values("priority_score"), x="priority_score", y="request_id",
            color="urgency", orientation="h",
            color_discrete_map={"Low": "#94A3B8", "Medium": WARNING, "High": DANGER},
            hover_data=["teacher_id", "department", "request_type", "days_open"],
        )
        fig.update_layout(height=400, xaxis_title="Priority Score (0–100)", yaxis_title="",
                          plot_bgcolor="white", paper_bgcolor="white")
        fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            top[["request_id", "teacher_id", "department", "request_type", "urgency", "days_open", "priority_score"]],
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.markdown("**Priority by Request Type**")
    by_type = active.groupby("request_type", as_index=False)["priority_score"].mean().sort_values("priority_score", ascending=False)
    if not by_type.empty:
        fig2 = px.bar(by_type, x="priority_score", y="request_type", orientation="h",
                      color_discrete_sequence=[ACCENT])
        fig2.update_layout(height=320, yaxis={"categoryorder": "total ascending"},
                           xaxis_title="Avg. Priority Score", plot_bgcolor="white", paper_bgcolor="white")
        fig2.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "💡 priority_score = 2×urgency + recency (days open) + frequency (how common that type is), "
        "rescaled to 0–100."
    )
