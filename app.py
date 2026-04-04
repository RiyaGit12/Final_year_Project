import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Download NLTK data
nltk.download('stopwords', quiet=True)

# ─────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────
st.set_page_config(
    page_title="AI Health Monitor",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .title {
        text-align: center;
        color: #1a73e8;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }
    .negative { background-color: #fde8e8; color: #c0392b; border: 2px solid #c0392b; }
    .positive { background-color: #e8f8e8; color: #27ae60; border: 2px solid #27ae60; }
    .neutral  { background-color: #e8f0fe; color: #1a73e8; border: 2px solid #1a73e8; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# Title
# ─────────────────────────────────────
st.markdown('<div class="title">🏥 AI Health Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Based Analysis of Social Media Data for Public Health Domain</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────
# Load & Train Model
# ─────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv('data/health_news_labeled.csv')
    df.fillna('', inplace=True)

    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(df['cleaned_text'])

    def encode(label):
        return {'Negative': 0, 'Neutral': 1, 'Positive': 2}[label]
    y = df['label'].apply(encode)

    svm = SVC(kernel='linear', probability=True)
    svm.fit(X, y)

    return tfidf, svm, df

tfidf, model, df = load_and_train()

# ─────────────────────────────────────
# Clean Text Function
# ─────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# ─────────────────────────────────────
# Main Layout — Two Columns
# ─────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🔍 Analyze Health Text")
    user_input = st.text_area(
        "Enter any health-related news or text below:",
        placeholder="e.g. New flu outbreak reported in multiple cities...",
        height=160
    )

    analyze_btn = st.button("🚀 Analyze Now", use_container_width=True)

    if analyze_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            cleaned = clean_text(user_input)
            vector = tfidf.transform([cleaned])
            prediction = model.predict(vector)[0]
            proba = model.predict_proba(vector)[0]

            labels = ['Negative', 'Neutral', 'Positive']
            label = labels[prediction]
            confidence = round(max(proba) * 100, 2)

            # Result Box
            emoji = {"Negative": "🔴", "Neutral": "🔵", "Positive": "🟢"}[label]
            css_class = label.lower()

            st.markdown(f"""
                <div class="result-box {css_class}">
                    {emoji} Sentiment: {label}<br>
                    <span style="font-size:16px;">Confidence: {confidence}%</span>
                </div>
            """, unsafe_allow_html=True)

            # Confidence Bar Chart
            st.markdown("#### 📊 Confidence Scores")
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ['#e74c3c', '#3498db', '#2ecc71']
            bars = ax.barh(labels, [p*100 for p in proba], color=colors)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Confidence (%)')
            for bar, p in zip(bars, proba):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{p*100:.1f}%', va='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

with col2:
    st.subheader("📈 Dataset Overview")

    # Pie Chart
    label_counts = df['label'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    ax2.pie(label_counts.values, labels=label_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=140)
    ax2.set_title('Sentiment Distribution', fontweight='bold')
    st.pyplot(fig2)

    # Stats
    st.markdown("#### 📋 Dataset Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Articles", len(df))
    c2.metric("Negative", label_counts.get('Negative', 0))
    c3.metric("Positive", label_counts.get('Positive', 0))

# ─────────────────────────────────────
# Footer
# ─────────────────────────────────────
st.markdown("---")
st.markdown("""
    <div style='text-align:center; color:#888; font-size:14px;'>
    🎓 Final Year Project | AI Based Analysis of Social Media Data for Public Health Domain<br>
    Riya Kumari | Roll: 22-CS-01 | GEC Lakhisarai
    </div>
""", unsafe_allow_html=True)
# ─────────────────────────────────────
# Disease Outbreak Detector Section
# ─────────────────────────────────────
st.markdown("---")
st.markdown("## 🚨 Real-Time Disease Outbreak Detector")
st.markdown("Detecting if any new disease is spreading across countries...")

# Country list to search
countries = [
    "India", "USA", "China", "UK", "Brazil", 
    "Russia", "France", "Germany", "Japan", "Australia",
    "Pakistan", "Bangladesh", "Africa", "Italy", "Canada"
]

# Disease keywords
disease_keywords = [
    "outbreak", "epidemic", "disease spreading", 
    "virus alert", "health emergency", "infection spreading",
    "new disease", "deadly virus", "contagious disease"
]

@st.cache_data(ttl=3600)  # refresh every 1 hour
def fetch_outbreak_news():
    import requests
    
    API_KEY = "fc6988b7aab442d2a9d8f8f6cbec9d3b"  # ← paste your NewsAPI key here
    all_alerts = []

    for country in countries:
        for keyword in disease_keywords[:3]:  # limit requests
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{keyword} {country}',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 3,
                'apiKey': API_KEY
            }
            try:
                response = requests.get(url, params=params, timeout=5)
                data = response.json()
                
                if data.get('articles'):
                    for article in data['articles']:
                        if article['title'] and article['description']:
                            all_alerts.append({
                                'Country': f'🌍 {country}',
                                'Title': article['title'],
                                'Description': article['description'][:150] + '...',
                                'Date': article['publishedAt'][:10],
                                'Source': article['source']['name'],
                                'URL': article['url']
                            })
            except:
                continue

    return pd.DataFrame(all_alerts).drop_duplicates(subset=['Title'])

# ── Search Controls ──
col_a, col_b = st.columns([1, 1])

with col_a:
    selected_country = st.selectbox(
        "🌍 Select Country to Monitor:",
        ["All Countries"] + countries
    )

with col_b:
    alert_btn = st.button("🔍 Check for Outbreaks", use_container_width=True)

if alert_btn:
    with st.spinner("⏳ Scanning real-time health news worldwide..."):
        df_alerts = fetch_outbreak_news()

    if df_alerts.empty:
        st.success("✅ No major disease outbreaks detected right now!")
    else:
        # Filter by country if selected
        if selected_country != "All Countries":
            df_filtered = df_alerts[
                df_alerts['Country'].str.contains(selected_country)
            ]
        else:
            df_filtered = df_alerts

        if df_filtered.empty:
            st.success(f"✅ No outbreaks detected in {selected_country}!")
        else:
            # Show alert count
            st.error(f"🚨 {len(df_filtered)} Potential Outbreak Alerts Found!")

            # Show each alert as a card
            for _, row in df_filtered.iterrows():
                with st.expander(f"{row['Country']} — {row['Title'][:80]}..."):
                    st.markdown(f"**📰 Source:** {row['Source']}")
                    st.markdown(f"**📅 Date:** {row['Date']}")
                    st.markdown(f"**📝 Description:** {row['Description']}")
                    st.markdown(f"**🔗 Read More:** [Click Here]({row['URL']})")

            # Summary Table
            st.markdown("### 📊 Outbreak Summary Table")
            st.dataframe(
                df_filtered[['Country', 'Title', 'Date', 'Source']],
                use_container_width=True
            )

            # Bar chart - alerts per country
            st.markdown("### 📈 Alerts by Country")
            country_counts = df_filtered['Country'].value_counts()
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.bar(country_counts.index, country_counts.values, color='#e74c3c')
            ax3.set_xlabel('Country')
            ax3.set_ylabel('Number of Alerts')
            ax3.set_title('Disease Alerts by Country', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig3)