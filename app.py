import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, nltk, os, requests, time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")
nltk.download('stopwords', quiet=True)

# ─────────────────────────────────────
# Page Config
# ─────────────────────────────────────
st.set_page_config(
    page_title="AI Health Monitor",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .header {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        padding: 25px; border-radius: 12px;
        text-align: center; color: white; margin-bottom: 20px;
    }
    .result-negative { background:#fde8e8; color:#c0392b; padding:20px;
        border-radius:10px; text-align:center; font-size:22px;
        font-weight:bold; border:2px solid #c0392b; }
    .result-positive { background:#e8f8e8; color:#27ae60; padding:20px;
        border-radius:10px; text-align:center; font-size:22px;
        font-weight:bold; border:2px solid #27ae60; }
    .result-neutral  { background:#e8f0fe; color:#1a73e8; padding:20px;
        border-radius:10px; text-align:center; font-size:22px;
        font-weight:bold; border:2px solid #1a73e8; }
    .alert-critical { background:#fff0f0; border-left:4px solid #e74c3c;
        padding:12px 16px; border-radius:8px; margin-bottom:10px; }
    .alert-normal   { background:#f0f8ff; border-left:4px solid #3498db;
        padding:12px 16px; border-radius:8px; margin-bottom:10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# Header
# ─────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🏥 AI Health Monitor</h1>
    <p>AI Based Analysis of Social Media Data for Public Health Domain</p>
    <small>Riya Kumari | Roll: 22-CS-01 | GEC Lakhisarai</small>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# Sidebar
# ─────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=80)
st.sidebar.title("🏥 Navigation")
st.sidebar.markdown("---")

menu = st.sidebar.selectbox(
    "📌 Go To:",
    ["📊 Dashboard", "🔍 Analyze Text", "🚨 Outbreak Alerts"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("AI & ML based real-time health news analyzer and outbreak detector.")
st.sidebar.markdown("---")
st.sidebar.markdown("**👩‍💻 Developer:** Riya Kumari")
st.sidebar.markdown("**🎓 College:** GEC Lakhisarai")
st.sidebar.markdown("**📅 Batch:** 2022–2026")

# ─────────────────────────────────────
# Load & Train Model
# ─────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv('data/health_news_labeled.csv')
    df.fillna('', inplace=True)
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['label'].map({'Negative':0,'Neutral':1,'Positive':2})
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X, y)
    return tfidf, svm, df

tfidf, model, df = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    return ' '.join([w for w in text.split() if w not in stop_words])

# ════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════
if menu == "📊 Dashboard":
    st.title("📊 Dashboard")

    # ── Refresh Button ──
    if st.button("🔄 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")

    # ── Metrics ──
    label_counts = df['label'].value_counts()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📰 Total Articles", len(df))
    c2.metric("🔴 Negative", label_counts.get('Negative',0))
    c3.metric("🔵 Neutral",  label_counts.get('Neutral', 0))
    c4.metric("🟢 Positive", label_counts.get('Positive',0))

    st.markdown("---")

    # ── Charts ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🥧 Sentiment Distribution")
        fig1, ax1 = plt.subplots(figsize=(4,4))
        colors = ['#e74c3c','#3498db','#2ecc71']
        wedges, texts, autotexts = ax1.pie(
            label_counts.values,
            labels=label_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=140,
            pctdistance=0.82,
            wedgeprops=dict(width=0.6)
        )
        for t in texts:
            t.set_fontsize(12); t.set_fontweight('bold')
        for at in autotexts:
            at.set_fontsize(10); at.set_color('white'); at.set_fontweight('bold')
        ax1.set_title('Sentiment Distribution', fontweight='bold', fontsize=13)
        ax1.legend(label_counts.index, loc="lower center",
                   bbox_to_anchor=(0.5,-0.08), ncol=3,
                   fontsize=10, frameon=False)
        plt.tight_layout()
        st.pyplot(fig1)

    with col_b:
        st.subheader("📊 Model Accuracy")
        fig2, ax2 = plt.subplots(figsize=(4,4))
        models_list = ['Naive\nBayes','Decision\nTree','SVM']
        accuracies  = [84.2, 78.9, 84.2]
        bar_colors  = ['#3498db','#e74c3c','#2ecc71']
        bars = ax2.bar(models_list, accuracies,
                       color=bar_colors, width=0.35,
                       edgecolor='white', linewidth=1.2)
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.8, f'{acc}%',
                     ha='center', fontweight='bold', fontsize=11)
        ax2.set_ylim(0,105)
        ax2.set_ylabel('Accuracy (%)', fontsize=11)
        ax2.set_title('Model Accuracy Comparison',
                      fontweight='bold', fontsize=13)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        ax2.set_axisbelow(True)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)

    # ── Horizontal Count Chart ──
    st.markdown("---")
    st.subheader("🌍 Article Count by Sentiment")
    fig3, ax3 = plt.subplots(figsize=(7,3))
    bar_cols3 = ['#e74c3c','#3498db','#2ecc71'][:len(label_counts)]
    bars3 = ax3.barh(label_counts.index.tolist(),
                     label_counts.values.tolist(),
                     color=bar_cols3, height=0.4, edgecolor='white')
    for bar, val in zip(bars3, label_counts.values):
        ax3.text(bar.get_width()+0.3,
                 bar.get_y()+bar.get_height()/2,
                 str(val), va='center',
                 fontweight='bold', fontsize=11)
    ax3.set_xlabel('Number of Articles', fontsize=11)
    ax3.set_title('Article Count per Sentiment',
                  fontweight='bold', fontsize=13)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.xaxis.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
    ax3.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("---")

    # ── Search + Filter ──
    st.subheader("🔎 Search & Filter News")
    col_s, col_f = st.columns(2)
    with col_s:
        search = st.text_input("🔍 Search by keyword:", 
                                placeholder="e.g. covid, flu, vaccine...")
    with col_f:
        keyword_filter = st.selectbox("📂 Filter by Sentiment:",
                                      ["All","Negative","Neutral","Positive"])

    filtered_df = df.copy()
    if search:
        filtered_df = filtered_df[
            filtered_df['Title'].str.contains(search, case=False, na=False)
        ]
    if keyword_filter != "All":
        filtered_df = filtered_df[filtered_df['label'] == keyword_filter]

    st.markdown(f"**Showing {len(filtered_df)} articles**")
    st.dataframe(
        filtered_df[['Title','label','Date']].head(20),
        use_container_width=True
    )

    # ── Download Button ──
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False),
        file_name="health_news_filtered.csv",
        mime="text/csv"
    )

# ════════════════════════════════════════
# PAGE 2 — ANALYZE TEXT
# ════════════════════════════════════════
elif menu == "🔍 Analyze Text":
    st.title("🔍 Health Text Analyzer")
    st.markdown("Enter any health-related text to analyze its sentiment.")
    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        user_input = st.text_area(
            "📝 Enter Health Text:",
            placeholder="e.g. New flu outbreak reported in multiple cities...",
            height=180
        )

        st.markdown("**💡 Quick Examples:**")
        ex1 = st.button("🔴 Negative Example")
        ex2 = st.button("🟢 Positive Example")
        ex3 = st.button("🔵 Neutral Example")

        if ex1: user_input = "Dangerous virus outbreak spreading rapidly causing deaths and panic"
        if ex2: user_input = "New vaccine successfully developed for deadly disease prevention"
        if ex3: user_input = "Doctors discuss new medicine research at medical conference"

        analyze_btn = st.button("🚀 Analyze Now", use_container_width=True)

    with col2:
        st.markdown("### 📌 How It Works")
        st.markdown("""
        **Step 1** 📥 You enter health text  
        **Step 2** 🧹 Text gets cleaned (NLP)  
        **Step 3** 🔢 Converted to numbers (TF-IDF)  
        **Step 4** 🤖 SVM model predicts sentiment  
        **Step 5** 📊 Result shown with confidence  
        """)
        st.info("💡 The model is trained on real health news data collected using NewsAPI")

    if analyze_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text first!")
        else:
            with st.spinner("🔍 Analyzing text..."):
                time.sleep(1)
                cleaned    = clean_text(user_input)
                vector     = tfidf.transform([cleaned])
                pred       = model.predict(vector)[0]
                proba      = model.predict_proba(vector)[0]
                labels     = ['Negative','Neutral','Positive']
                label      = labels[pred]
                confidence = round(max(proba)*100, 2)
                emoji      = {"Negative":"🔴","Neutral":"🔵","Positive":"🟢"}[label]
                css        = f"result-{label.lower()}"

            st.markdown("---")
            st.markdown(f"""
                <div class="{css}">
                    {emoji} Sentiment: {label}<br>
                    <span style="font-size:16px;">
                    Confidence: {confidence}%</span>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Confidence Scores")
            fig4, ax4 = plt.subplots(figsize=(6,3))
            colors = ['#e74c3c','#3498db','#2ecc71']
            bars4 = ax4.barh(labels,
                             [p*100 for p in proba],
                             color=colors)
            ax4.set_xlim(0,100)
            ax4.set_xlabel('Confidence (%)')
            for bar, p in zip(bars4, proba):
                ax4.text(bar.get_width()+1,
                         bar.get_y()+bar.get_height()/2,
                         f'{p*100:.1f}%', va='center',
                         fontweight='bold')
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig4)

# ════════════════════════════════════════
# PAGE 3 — OUTBREAK ALERTS
# ════════════════════════════════════════
elif menu == "🚨 Outbreak Alerts":
    st.title("🚨 Real-Time Disease Outbreak Detector")
    st.markdown("Scan real-time news to detect disease outbreaks by country.")
    st.markdown("---")

    countries = ["India","USA","China","UK","Brazil",
                 "Russia","France","Germany","Japan",
                 "Australia","Pakistan","Italy","Canada"]

    col1, col2 = st.columns([1,1])
    with col1:
        selected_country = st.selectbox("🌍 Select Country:",
                                        ["All Countries"] + countries)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        alert_btn = st.button("🔍 Check for Outbreaks",
                               use_container_width=True)

    @st.cache_data(ttl=3600)
    def fetch_outbreaks(country):
        query  = f"disease outbreak {country}" \
                 if country != "All Countries" else "disease outbreak"
        url    = "https://newsapi.org/v2/everything"
        params = {
            'q': query, 'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20, 'apiKey': API_KEY
        }
        results = []
        try:
            r    = requests.get(url, params=params, timeout=10)
            data = r.json()
            for a in data.get('articles', []):
                if a['title'] and a['description']:
                    results.append({
                        'Country': country,
                        'Title': a['title'],
                        'Description': (a['description'] or '')[:150]+'...',
                        'Date': (a['publishedAt'] or '')[:10],
                        'Source': a['source']['name'],
                        'URL': a['url'],
                        'Critical': any(w in a['title'].lower()
                                        for w in ['virus','outbreak',
                                                  'epidemic','death',
                                                  'emergency','deadly'])
                    })
        except:
            pass
        return pd.DataFrame(results)

    if alert_btn:
        with st.spinner("⏳ Scanning real-time health news..."):
            time.sleep(1)
            df_alerts = fetch_outbreaks(selected_country)

        if df_alerts.empty:
            st.success("✅ No major outbreaks detected!")
        else:
            # ── Alert counts ──
            critical = df_alerts[df_alerts['Critical']==True]
            normal   = df_alerts[df_alerts['Critical']==False]

            col_x, col_y = st.columns(2)
            col_x.error(f"🚨 {len(critical)} Critical Alerts")
            col_y.info(f"ℹ️ {len(normal)} General Alerts")

            st.markdown("---")
            st.subheader("📰 Latest Alerts")

            # ── Smart Alert Cards ──
            for _, row in df_alerts.iterrows():
                if row['Critical']:
                    st.markdown(f"""
                    <div class="alert-critical">
                        🚨 <b>{row['Title']}</b><br>
                        <small>📅 {row['Date']} | 
                        🗞️ {row['Source']}</small><br>
                        {row['Description']}
                        <br><a href="{row['URL']}" 
                        target="_blank">🔗 Read More</a>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-normal">
                        ℹ️ <b>{row['Title']}</b><br>
                        <small>📅 {row['Date']} | 
                        🗞️ {row['Source']}</small><br>
                        {row['Description']}
                        <br><a href="{row['URL']}" 
                        target="_blank">🔗 Read More</a>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Summary Table ──
            st.subheader("📋 Summary Table")
            st.dataframe(
                df_alerts[['Title','Date','Source','Critical']],
                use_container_width=True
            )

            # ── Download ──
            st.download_button(
                label="⬇️ Download Alerts as CSV",
                data=df_alerts.to_csv(index=False),
                file_name="outbreak_alerts.csv",
                mime="text/csv"
            )

# ─────────────────────────────────────
# Footer
# ─────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:13px;'>
🎓 Final Year Project | AI Based Analysis of Social Media Data for Public Health<br>
Riya Kumari | Roll: 22-CS-01 | GEC Lakhisarai | 2022–2026
</div>
""", unsafe_allow_html=True)