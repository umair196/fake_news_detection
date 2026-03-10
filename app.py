import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from predict_helper import predict_news

st.set_page_config(
    page_title="Fake News Detection Pro",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Session State ----------------
if "headline_input" not in st.session_state:
    st.session_state.headline_input = ""

if "article_input" not in st.session_state:
    st.session_state.article_input = ""

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Helpers ----------------
def get_result_color(prediction: str) -> str:
    return "#ff4b4b" if prediction == "Fake News" else "#00c853"

def get_result_bg(prediction: str) -> str:
    return "rgba(255, 75, 75, 0.12)" if prediction == "Fake News" else "rgba(0, 200, 83, 0.12)"

def load_sample(headline, article):
    st.session_state.headline_input = headline
    st.session_state.article_input = article

def clear_inputs():
    st.session_state.headline_input = ""
    st.session_state.article_input = ""

def clear_history():
    st.session_state.history = []

def add_to_history(headline, article, result):
    st.session_state.history.insert(
        0,
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Headline": headline,
            "Article Preview": article[:100] + ("..." if len(article) > 100 else ""),
            "Prediction": result["prediction"],
            "Confidence": result["confidence"],
            "Interpretation": result["interpretation"]
        }
    )
    st.session_state.history = st.session_state.history[:10]

def build_export_text(headline, article, result):
    return f"""
Fake News Detection Report
==========================

Headline:
{headline}

Article:
{article}

Model Used:
{result['model_used']}

Prediction:
{result['prediction']}

Raw Prediction:
{result['raw_prediction']}

Decision Score:
{result['score']}

Confidence:
{result['confidence']}

Interpretation:
{result['interpretation']}

Word Count:
{result['word_count']}

Character Count:
{result['char_count']}

Input Quality:
{result['input_quality']}

Suspicious Terms:
{', '.join(result['suspicious_terms']) if result['suspicious_terms'] else 'None'}

Cleaned Text:
{result['cleaned_text']}
""".strip()

def make_input_chart(word_count, char_count):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Word Count", "Character Count"], [word_count, char_count])
    ax.set_title("Input Size Overview")
    return fig

def make_confidence_chart(confidence):
    if confidence is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Confidence"], [confidence])
    ax.set_ylim(0, 100)
    ax.set_title("Confidence Indicator")
    return fig

# ---------------- Style ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 30%, #1e293b 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

.hero-card {
    background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(124,58,237,0.22));
    border: 1px solid rgba(255,255,255,0.10);
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    margin-bottom: 18px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-size: 16px;
    color: #dbeafe;
    margin-bottom: 12px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 14px;
    margin-right: 8px;
    margin-top: 8px;
    border-radius: 999px;
    background: rgba(255,255,255,0.10);
    color: #f8fafc;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.12);
}

.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(12px);
    border-radius: 22px;
    padding: 20px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.18);
    margin-bottom: 16px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}

.metric-box {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 16px;
    text-align: center;
    margin-bottom: 12px;
}

.metric-label {
    color: #cbd5e1;
    font-size: 13px;
    margin-bottom: 4px;
}

.metric-value {
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
}

.result-card {
    border-radius: 24px;
    padding: 22px;
    margin-top: 10px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 8px 25px rgba(0,0,0,0.20);
}

.result-title {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 6px;
}

.info-chip {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    color: #e2e8f0;
    font-size: 13px;
    font-weight: 600;
    margin-right: 8px;
    margin-top: 6px;
    border: 1px solid rgba(255,255,255,0.08);
}

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.06) !important;
    color: #ffffff !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}

.stButton > button {
    width: 100%;
    border-radius: 16px;
    border: none;
    padding: 0.75rem 1rem;
    font-weight: 700;
    font-size: 15px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    box-shadow: 0 8px 18px rgba(37,99,235,0.30);
}

.footer-box {
    margin-top: 18px;
    text-align: center;
    color: #cbd5e1;
    font-size: 13px;
    padding: 14px 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.info("This app predicts whether a news item appears Real or Fake using a trained NLP model.")

    st.markdown("### 📌 Load Examples")
    sample_headline_1 = "Government approves major economic reform"
    sample_article_1 = "The government announced a new policy after an official cabinet meeting and parliamentary review."

    sample_headline_2 = "Secret miracle cure hidden for decades"
    sample_article_2 = "Shocking report claims scientists found an impossible cure that was hidden from the public for years."

    sample_headline_3 = "Finance ministry releases budget statement"
    sample_article_3 = "International markets reacted after the finance ministry published an official policy and budget update."

    st.button("Load Example 1", on_click=load_sample, args=(sample_headline_1, sample_article_1))
    st.button("Load Example 2", on_click=load_sample, args=(sample_headline_2, sample_article_2))
    st.button("Load Example 3", on_click=load_sample, args=(sample_headline_3, sample_article_3))

    st.markdown("---")
    st.button("🧹 Clear Inputs", on_click=clear_inputs)
    st.button("🗑 Clear History", on_click=clear_history)

# ---------------- Hero ----------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">📰 Fake News Detection Pro</div>
    <div class="hero-subtitle">
        A stronger end-to-end NLP application for analyzing headlines and article text using preprocessing,
        vectorization, and machine learning classification.
    </div>
    <span class="hero-badge">NLP</span>
    <span class="hero-badge">Machine Learning</span>
    <span class="hero-badge">TF-IDF</span>
    <span class="hero-badge">Charts</span>
    <span class="hero-badge">Streamlit</span>
</div>
""", unsafe_allow_html=True)

# ---------------- Top Metrics ----------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-box"><div class="metric-label">Project Type</div><div class="metric-value">NLP App</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-box"><div class="metric-label">Inputs</div><div class="metric-value">Headline + Article</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-box"><div class="metric-label">Prediction</div><div class="metric-value">Real / Fake</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-box"><div class="metric-label">Model</div><div class="metric-value">Saved ML Model</div></div>', unsafe_allow_html=True)

# ---------------- Inputs ----------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔎 Analyze News</div>', unsafe_allow_html=True)
headline = st.text_input("Headline", key="headline_input", placeholder="Enter headline here...")
article = st.text_area("Article Text", key="article_input", height=220, placeholder="Paste article text here...")

c1, c2 = st.columns(2)
with c1:
    predict_clicked = st.button("🚀 Predict Now")
with c2:
    use_demo = st.button("📄 Load Demo Example")

if use_demo:
    load_sample(
        "Official policy approved after cabinet review",
        "According to the ministry statement, the cabinet approved the economic package after formal review and public release."
    )
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

result = None
combined_text = f"{headline} {article}".strip()

if predict_clicked:
    if not combined_text:
        st.warning("Please enter a headline, article text, or both.")
    else:
        result = predict_news(combined_text)
        add_to_history(headline, article, result)

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "🧠 About Model", "📊 Project Insights", "🗂 History"])

with tab1:
    if result is None:
        st.info("Enter a headline/article and click 'Predict Now' to see results.")
    else:
        color = get_result_color(result["prediction"])
        bg = get_result_bg(result["prediction"])

        st.markdown(
            f"""
            <div class="result-card" style="background:{bg}; border:1px solid {color}55;">
                <div class="result-title" style="color:{color};">
                    {"🚨 Fake News Detected" if result["prediction"] == "Fake News" else "✅ Real News Detected"}
                </div>
                <div style="margin-top:10px;">
                    <span class="info-chip">Model: {result["model_used"]}</span>
                    <span class="info-chip">Prediction: {result["prediction"]}</span>
                    <span class="info-chip">Confidence: {result["confidence"]}</span>
                    <span class="info-chip">Interpretation: {result["interpretation"]}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Word Count", result["word_count"])
        with d2:
            st.metric("Character Count", result["char_count"])
        with d3:
            st.metric("Input Quality", result["input_quality"])

        if result["confidence"] is not None:
            st.markdown("### Confidence Meter")
            st.progress(int(result["confidence"]))
            st.caption(f"Confidence-style UI indicator: {result['confidence']}%")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Prediction Details")
        st.write(f"**Prediction:** {result['prediction']}")
        st.write(f"**Raw Prediction Class:** {result['raw_prediction']}")
        st.write(f"**Interpretation:** {result['interpretation']}")
        st.write(f"**Decision Score:** {result['score']}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Suspicious Terms Scan")
        if result["suspicious_terms"]:
            st.warning(", ".join(result["suspicious_terms"]))
        else:
            st.success("No suspicious terms detected from the built-in keyword list.")
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🧼 Show Cleaned Text Used by Model"):
            st.write(result["cleaned_text"])

        export_text = build_export_text(headline, article, result)
        st.download_button(
            label="📥 Download Prediction Report",
            data=export_text,
            file_name="fake_news_prediction_report.txt",
            mime="text/plain"
        )

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 About the Model</div>', unsafe_allow_html=True)
    st.write("""
This app uses a saved machine learning fake-news classifier.

### Pipeline
- text cleaning
- TF-IDF vectorization
- trained classification model
- score interpretation
- confidence-style UI

### Notes
This model predicts from learned text patterns.
It does not fact-check live internet sources.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    if result is None:
        st.info("Run a prediction first to see charts and project insights.")
    else:
        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Confidence Chart")
            fig = make_confidence_chart(result["confidence"])
            if fig is not None:
                st.pyplot(fig)
            else:
                st.warning("Confidence is not available for this model.")
            st.markdown('</div>', unsafe_allow_html=True)

        with chart2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Word Count Visuals")
            fig2 = make_input_chart(result["word_count"], result["char_count"])
            st.pyplot(fig2)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Project Insights")
        st.write("""
This app demonstrates an end-to-end NLP workflow:

- text preprocessing
- vectorization
- trained machine learning model
- confidence and input diagnostics
- history tracking
- downloadable result report

It is a strong portfolio-style project because it behaves like a usable product, not just a notebook result.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗂 Prediction History</div>', unsafe_allow_html=True)

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.caption("No prediction history yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("""
<div class="footer-box">
    Built with ❤️ using <b>Streamlit</b>, <b>TF-IDF</b>, and <b>Machine Learning</b><br>
    Fake News Detection Pro — NLP Portfolio Project
</div>
""", unsafe_allow_html=True)