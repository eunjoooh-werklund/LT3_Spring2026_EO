# AI in the Classroom — Principal Dashboard (Demo)

A multi-tab Streamlit dashboard demonstrating how a K-12 principal could monitor
teacher AI adoption — and its connection to student achievement, engagement,
planning time, professional learning, support needs, and teacher sentiment/trust.

**All data is simulated.** This is a methodology demo, not a deployed tool —
built to show that Python (pandas, NumPy, an NLP sentiment model, Plotly,
Streamlit) can turn qualitative + quantitative inputs into something a busy
principal could actually use day-to-day.

## Why these 7 tabs

| Tab | Principal question | Python / AI method |
|---|---|---|
| 🏠 Overview | How is AI being used across classrooms? | Frequency counts, adoption rate, weekly trend |
| 🎯 Achievement & Engagement | Is AI helping student outcomes? | Correlation, before/after comparison, engagement heatmap |
| ⏱️ Teacher Planning Time | Is AI reducing planning workload? | Min/max/mean, boxplots, IQR outlier detection |
| 🙋 Teacher Needs | What support do teachers ask for? | Text mining (keyword frequency), topic frequency, sentiment |
| 🤝 Mediating Factors for AI Integration | What do teachers need for AI integration to succeed? | Literature-grounded mediating-factor scoring (0–100), VADER sentiment analysis |
| 📚 Professional Learning | Who needs AI training or micro-credentials? | K-means clustering (readiness groups), rule-based recommendations |
| 🧰 Classroom Support Requests | Where should principals allocate resources? | Weighted priority scoring (urgency + recency + frequency) |

## Setup & Run

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open in your browser (usually `http://localhost:8501`).
Mock data is generated automatically into `/data` on first run
(or run `python3 generate_data.py` manually to regenerate it).

## Project Structure

```
ai_dashboard/
├── app.py                 # Main Streamlit app (6 tabs)
├── generate_data.py        # Mock data generator (pandas/numpy)
├── requirements.txt
├── README.md
└── data/                   # Generated CSVs (created on first run)
    ├── teachers.csv
    ├── weekly_usage.csv
    ├── professional_learning.csv
    ├── support_requests.csv
    ├── comments.csv
    └── trust_readiness.csv
```

## The Methodology Piece (for your write-up)

This dashboard demonstrates several applied AI/data-science methods, each tied to a
specific tab:

- **Sentiment analysis (VADER):** free-text teacher comments are scored from -1
  (very negative) to +1 (very positive) using a lexicon-based NLP model, then
  aggregated by week and department (Mediating Factors, Teacher Needs tabs).
- **Mediating-factor scoring (0–100):** six factors drawn from the leadership literature on
  AI integration — Trust in AI Policy, Understanding of Expectations, Institutional Support,
  Task-Technology Fit, AI Assessment Confidence, and Professional Learning Access — each
  scored per teacher and aggregated by department. Trust in AI Policy is highlighted as the
  factor your advisor identified as central; the literature suggests it mediates how
  effectively the other five factors translate into actual classroom AI use.
- **Text mining / keyword frequency:** open-ended comments are tokenized and
  stop-words removed to surface the most common terms teachers use when describing
  their needs (Teacher Needs tab) — a lightweight alternative to full topic modeling.
- **Outlier detection (IQR method):** teacher planning-time averages are flagged as
  high/low outliers using the standard 1.5×IQR rule (Teacher Planning Time tab).
- **K-means clustering:** teachers are grouped into data-driven "readiness profiles"
  (Needs Support / Building Confidence / Ready to Lead) based on trust index, AI
  usage rate, credentials completed, and goal progress — with a simple recommendation
  rule attached to each group (Professional Learning tab).
- **Priority scoring:** support requests get a 0–100 score combining urgency weight,
  days-open (recency), and how frequently that request type recurs — so principals
  can triage instead of reading every ticket (Classroom Support Requests tab).
- **Correlation & before/after comparison:** AI usage rate is correlated against
  engagement/achievement proxies, plus a median-split before/after view (Achievement
  & Engagement tab).

All of these are "the kind of analysis principals would not have time to run
themselves" — which is the methodological argument for why a dashboard like this adds
value, per your advisor's feedback.

## Customizing

- **Change mock data realism:** edit the distributions in `generate_data.py`
  (e.g. `N_TEACHERS`, `N_WEEKS`, the `RNG.normal(...)` parameters).
- **Swap in real data:** replace the CSVs in `/data` with real survey/usage
  exports using the same column names, and the dashboard will work unchanged.
- **Add a tab:** copy one of the `with tabN:` blocks in `app.py` as a template.

## Deploying (optional, for sharing a live link)

Push this folder to a public GitHub repo, then deploy free at
[share.streamlit.io](https://share.streamlit.io) (Streamlit Community Cloud) —
point it at `app.py` and it will install `requirements.txt` automatically.
