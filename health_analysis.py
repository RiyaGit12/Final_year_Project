import requests
import pandas as pd
import os
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Load API Key from .env ──
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

# ── Setup ──
os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
nltk.download('stopwords', quiet=True)

# ════════════════════════════════════════
# STEP 1 — Data Collection
# ════════════════════════════════════════
print("⏳ Fetching real-time health news...")

url = "https://newsapi.org/v2/everything"
params = {
    'q': 'health OR disease OR symptoms OR outbreak OR virus',
    'language': 'en',
    'sortBy': 'publishedAt',
    'pageSize': 100,
    'apiKey': API_KEY
}

response = requests.get(url, params=params)
data = response.json()

articles = []
for article in data['articles']:
    articles.append({
        'Date':        article['publishedAt'],
        'Source':      article['source']['name'],
        'Title':       article['title'],
        'Description': article['description'],
        'Content':     article['content']
    })

df = pd.DataFrame(articles)
df.to_csv('data/health_news.csv', index=False)
print(f"✅ Collected {len(df)} articles")

# ════════════════════════════════════════
# STEP 2 — Preprocessing
# ════════════════════════════════════════
print("\n⏳ Preprocessing data...")

df = pd.read_csv('data/health_news.csv')
df.fillna('', inplace=True)
df['text'] = df['Title'] + ' ' + df['Description']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

df['cleaned_text'] = df['text'].apply(clean_text)
df.to_csv('data/health_news_cleaned.csv', index=False)
print("✅ Preprocessing Done!")
print("\nOriginal:", df['text'][0])
print("Cleaned: ", df['cleaned_text'][0])

# ════════════════════════════════════════
# STEP 3 — Feature Extraction + Labeling
# ════════════════════════════════════════
print("\n⏳ Extracting features...")

df = pd.read_csv('data/health_news_cleaned.csv')
df.fillna('', inplace=True)

tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['cleaned_text'])

print(f"✅ Feature Extraction Done!")
print(f"Shape: {X.shape} → {X.shape[0]} articles, {X.shape[1]} features")

def assign_label(text):
    text = text.lower()
    if any(w in text for w in ['outbreak','virus','death','danger',
                                'risk','warning','disease']):
        return 'Negative'
    elif any(w in text for w in ['cure','recovery','healthy','vaccine',
                                  'treatment','prevent']):
        return 'Positive'
    else:
        return 'Neutral'

df['label'] = df['cleaned_text'].apply(assign_label)
df.to_csv('data/health_news_labeled.csv', index=False)

print("\nLabel Distribution:")
print(df['label'].value_counts())

# ════════════════════════════════════════
# STEP 4 — Model Training
# ════════════════════════════════════════
print("\n⏳ Training ML Models...")

df  = pd.read_csv('data/health_news_labeled.csv')
df.fillna('', inplace=True)

tfidf = TfidfVectorizer(max_features=1000)
X     = tfidf.fit_transform(df['cleaned_text'])

le = LabelEncoder()
y  = le.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Naive Bayes
nb      = MultinomialNB()
nb.fit(X_train, y_train)
nb_acc  = accuracy_score(y_test, nb.predict(X_test))

# Decision Tree
dt      = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc  = accuracy_score(y_test, dt.predict(X_test))

# SVM
svm     = SVC(kernel='linear', random_state=42, probability=True)
svm.fit(X_train, y_train)
svm_pred= svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

print("\n" + "="*40)
print("      MODEL COMPARISON RESULTS")
print("="*40)
print(f"✅ Naive Bayes Accuracy:   {nb_acc*100:.2f}%")
print(f"✅ Decision Tree Accuracy: {dt_acc*100:.2f}%")
print(f"✅ SVM Accuracy:           {svm_acc*100:.2f}%")
print("="*40)
print("\n📊 Detailed Report (SVM):")
print(classification_report(y_test, svm_pred,
      target_names=le.classes_))

results = pd.DataFrame({
    'Model':    ['Naive Bayes','Decision Tree','SVM'],
    'Accuracy': [nb_acc*100, dt_acc*100, svm_acc*100]
})
results.to_csv('outputs/model_results.csv', index=False)
print("✅ Results saved!")

# ════════════════════════════════════════
# STEP 5 — Visualizations
# ════════════════════════════════════════
print("\n⏳ Generating visualizations...")

# Graph 1 — Model Accuracy
plt.figure(figsize=(8,5))
bars = plt.bar(['Naive Bayes','Decision Tree','SVM'],
               [nb_acc*100, dt_acc*100, svm_acc*100],
               color=['#3498db','#e74c3c','#2ecc71'], width=0.4)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for bar, acc in zip(bars, [nb_acc*100, dt_acc*100, svm_acc*100]):
    plt.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+1, f'{acc:.1f}%',
             ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/model_comparison.png')
plt.show()
print("✅ Graph 1 saved!")

# Graph 2 — Sentiment Pie
label_counts = df['label'].value_counts()
plt.figure(figsize=(7,7))
plt.pie(label_counts.values, labels=label_counts.index,
        autopct='%1.1f%%',
        colors=['#e74c3c','#3498db','#2ecc71'],
        startangle=140,
        textprops={'fontsize':13})
plt.title('Health News Sentiment Distribution',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/sentiment_distribution.png')
plt.show()
print("✅ Graph 2 saved!")

# Graph 3 — Confusion Matrix
cm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix - SVM', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png')
plt.show()
print("✅ Graph 3 saved!")

# Graph 4 — Top Keywords
feature_names = tfidf.get_feature_names_out()
weights       = np.array(X.sum(axis=0)).flatten()
top_indices   = weights.argsort()[-10:][::-1]
top_words     = [feature_names[i] for i in top_indices]
top_scores    = [weights[i] for i in top_indices]

plt.figure(figsize=(9,5))
plt.barh(top_words[::-1], top_scores[::-1], color='#9b59b6')
plt.title('Top 10 Health Keywords', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/top_keywords.png')
plt.show()
print("✅ Graph 4 saved!")

print("\n" + "="*40)
print("  🎉 PROJECT COMPLETE! ALL DONE! 🎉")
print("="*40)