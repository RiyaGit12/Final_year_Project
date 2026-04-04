import requests
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

print("⏳ Fetching real-time health news... please wait")

# Free News API - no key needed
url = "https://newsapi.org/v2/everything"

params = {
    'q': 'health OR disease OR symptoms OR outbreak OR virus',
    'language': 'en',
    'sortBy': 'publishedAt',
    'pageSize': 100,
    'apiKey': 'fc6988b7aab442d2a9d8f8f6cbec9d3b'  # we'll get this in next step
}

response = requests.get(url, params=params)
data = response.json()

articles = []
for article in data['articles']:
    articles.append({
        'Date': article['publishedAt'],
        'Source': article['source']['name'],
        'Title': article['title'],
        'Description': article['description'],
        'Content': article['content']
    })

df = pd.DataFrame(articles)
df.to_csv('data/health_news.csv', index=False)

print(f"✅ Done! Collected {len(df)} articles")
print(df.head())
import nltk
import re
import string

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords

print("\n⏳ Preprocessing data...")

# Load the saved data
df = pd.read_csv('data/health_news.csv')

# Fill empty values
df.fillna('', inplace=True)

# Combine Title + Description into one column
df['text'] = df['Title'] + ' ' + df['Description']

# ---- Cleaning Function ----
def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'http\S+', '', text)          # remove links
    text = re.sub(r'[^a-z\s]', '', text)         # remove numbers/symbols
    text = re.sub(r'\s+', ' ', text).strip()     # remove extra spaces
    
    # Remove stopwords (common words like 'the', 'is', 'a')
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    return ' '.join(words)

# Apply cleaning to all rows
df['cleaned_text'] = df['text'].apply(clean_text)

# Save cleaned data
df.to_csv('data/health_news_cleaned.csv', index=False)

print("✅ Preprocessing Done!")
print("\nOriginal text sample:")
print(df['text'][0])
print("\nCleaned text sample:")
print(df['cleaned_text'][0])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("\n⏳ Extracting features...")

# Load cleaned data
df = pd.read_csv('data/health_news_cleaned.csv')
df.fillna('', inplace=True)

# ---- TF-IDF Feature Extraction ----
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['cleaned_text'])

print(f"✅ Feature Extraction Done!")
print(f"Shape of feature matrix: {X.shape}")
print(f"This means: {X.shape[0]} articles and {X.shape[1]} features")

# ---- Create Labels (Sentiment) ----
# We'll label articles based on keywords
def assign_label(text):
    text = text.lower()
    if any(word in text for word in ['outbreak', 'virus', 'death', 'danger', 'risk', 'warning', 'disease']):
        return 'Negative'
    elif any(word in text for word in ['cure', 'recovery', 'healthy', 'vaccine', 'treatment', 'prevent']):
        return 'Positive'
    else:
        return 'Neutral'

df['label'] = df['cleaned_text'].apply(assign_label)

# Save with labels
df.to_csv('data/health_news_labeled.csv', index=False)

# Check label distribution
print("\nLabel Distribution:")
print(df['label'].value_counts())
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

print("\n⏳ Training ML Models...")

# Load labeled data
df = pd.read_csv('data/health_news_labeled.csv')
df.fillna('', inplace=True)

# Features and Labels
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['cleaned_text'])

# Encode labels to numbers
le = LabelEncoder()
y = le.fit_transform(df['label'])  # Negative=0, Neutral=1, Positive=2

# Split data - 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ---- Model 1: Naive Bayes ----
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

# ---- Model 2: Decision Tree ----
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# ---- Model 3: SVM ----
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

# ---- Print Results ----
print("\n" + "="*40)
print("        MODEL COMPARISON RESULTS")
print("="*40)
print(f"✅ Naive Bayes Accuracy:     {nb_acc*100:.2f}%")
print(f"✅ Decision Tree Accuracy:   {dt_acc*100:.2f}%")
print(f"✅ SVM Accuracy:             {svm_acc*100:.2f}%")
print("="*40)

# Best model detailed report
print("\n📊 Detailed Report (Best Model - SVM):")
print(classification_report(y_test, svm_pred, 
      target_names=le.classes_))

# Save results
results = pd.DataFrame({
    'Model': ['Naive Bayes', 'Decision Tree', 'SVM'],
    'Accuracy': [nb_acc*100, dt_acc*100, svm_acc*100]
})
results.to_csv('outputs/model_results.csv', index=False)
print("\n✅ Results saved to outputs/model_results.csv")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("\n⏳ Generating visualizations...")

os.makedirs('outputs', exist_ok=True)

# ---- Graph 1: Model Accuracy Comparison ----
models = ['Naive Bayes', 'Decision Tree', 'SVM']
accuracies = [nb_acc*100, dt_acc*100, svm_acc*100]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'], width=0.4)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/model_comparison.png')
plt.show()
print("✅ Graph 1 saved!")

# ---- Graph 2: Label Distribution Pie Chart ----
label_counts = df['label'].value_counts()
colors = ['#e74c3c', '#3498db', '#2ecc71']

plt.figure(figsize=(7, 7))
plt.pie(label_counts.values, labels=label_counts.index,
        autopct='%1.1f%%', colors=colors, startangle=140,
        textprops={'fontsize': 13})
plt.title('Health News Sentiment Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/sentiment_distribution.png')
plt.show()
print("✅ Graph 2 saved!")

# ---- Graph 3: Confusion Matrix ----
cm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix - SVM Model', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png')
plt.show()
print("✅ Graph 3 saved!")

# ---- Graph 4: Top 10 Health Keywords ----
feature_names = tfidf.get_feature_names_out()
svm_weights = np.abs(svm.coef_).mean(axis=0) if hasattr(svm, 'coef_') else \
              np.array(X.sum(axis=0)).flatten()
top_indices = svm_weights.argsort()[-10:][::-1]
top_words = [feature_names[i] for i in top_indices]
top_scores = [svm_weights[i] for i in top_indices]

plt.figure(figsize=(9, 5))
plt.barh(top_words[::-1], top_scores[::-1], color='#9b59b6')
plt.title('Top 10 Health Keywords in Data', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/top_keywords.png')
plt.show()
print("✅ Graph 4 saved!")

print("\n" + "="*40)
print("  🎉 PROJECT COMPLETE! ALL DONE! 🎉")
print("="*40)
print("📁 Check your 'outputs' folder for all graphs!")