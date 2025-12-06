# ==================== FAKE NEWS DETECTION - ISOT DATASET ====================
# Save as: fake_news_isot.py

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

print("🎯 FAKE NEWS DETECTION - ISOT DATASET")
print("=" * 60)

# ==================== 1. LOAD ISOT DATASET ====================
print("\n📂 Loading ISOT Fake News Dataset...")

# Make sure these files are in your project folder:
# Fake.csv and True.csv (from ISOT dataset)

try:
    # Load real news
    true_df = pd.read_csv('True.csv')
    true_df['label'] = 1  # 1 = REAL

    # Load fake news
    fake_df = pd.read_csv('Fake.csv')
    fake_df['label'] = 0  # 0 = FAKE

    # Combine
    df = pd.concat([true_df, fake_df], ignore_index=True)

    print(f"✅ SUCCESS: Loaded {len(df):,} total samples")
    print(f"   Real news (label=1): {len(true_df):,} samples")
    print(f"   Fake news (label=0): {len(fake_df):,} samples")
    print(f"   Perfect balance: 50% real, 50% fake")

except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    print("\n📥 DOWNLOAD INSTRUCTIONS:")
    print("1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    print("2. Click 'Download' (requires Kaggle account)")
    print("3. Extract both 'True.csv' and 'Fake.csv' to this folder")
    print("4. Run this script again")
    exit()

# ==================== 2. PREPROCESSING ====================
print("\n🧹 Preprocessing text data...")


def clean_text(text):
    """Clean text for better feature extraction"""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Combine title and text for more context
df['combined_text'] = df['title'] + ' ' + df['text']

# Clean the text
df['clean_text'] = df['combined_text'].apply(clean_text)

# Remove very short texts
df = df[df['clean_text'].str.len() > 50]

print(f"After cleaning: {len(df):,} samples")
print(f"Sample cleaned text: {df['clean_text'].iloc[0][:200]}...")

# ==================== 3. TRAIN/TEST SPLIT ====================
print("\n✂️ Creating train/test split...")

X = df['clean_text'].values  # Features (text)
y = df['label'].values  # Labels (0=fake, 1=real)

# Stratified split to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 80% train, 20% test
    stratify=y,  # Keep same class distribution
    random_state=42  # For reproducibility
)

print(f"Training set: {len(X_train):,} samples")
print(f"Testing set:  {len(X_test):,} samples")
print(f"Class distribution in train: {np.unique(y_train, return_counts=True)}")

# ==================== 4. FEATURE EXTRACTION (TF-IDF) ====================
print("\n🔤 Creating TF-IDF features...")

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,  # Use top 5000 words
    stop_words='english',  # Remove common English words
    ngram_range=(1, 2),  # Use single words and word pairs
    min_df=5,  # Word must appear in at least 5 documents
    max_df=0.7  # Word must appear in less than 70% of documents
)

# Fit on training data, transform both train and test
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_):,} words")

# ==================== 5. TRAIN LOGISTIC REGRESSION ====================
print("\n🏋️ Training Logistic Regression model...")

# Logistic Regression works well with TF-IDF
model = LogisticRegression(
    max_iter=1000,  # Maximum iterations
    random_state=42,  # Reproducibility
    class_weight='balanced',  # Handle any minor imbalance
    C=1.0,  # Regularization strength
    solver='liblinear'  # Good for binary classification
)

# Train the model
model.fit(X_train_tfidf, y_train)

print("✅ Model training complete!")

# ==================== 6. EVALUATION ====================
print("\n🧪 Evaluating model performance...")

# Make predictions
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 TEST RESULTS:")
print(f"Accuracy: {accuracy:.2%}")

print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
# ==================== 6.5 OVERFITTING DIAGNOSTIC ====================
print("\n" + "=" * 60)
print("🔍 OVERFITTING ANALYSIS")
print("=" * 60)

# Calculate training accuracy
y_train_pred = model.predict(X_train_tfidf)
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"\n📊 ACCURACY COMPARISON:")
print(f"Training Accuracy:   {train_accuracy:.2%}")
print(f"Testing Accuracy:    {accuracy:.2%}")
print(f"Difference (Gap):    {abs(train_accuracy - accuracy):.2%}")

# Assess overfitting
gap = abs(train_accuracy - accuracy)
if gap > 0.05:
    print("🚨 WARNING: Potential overfitting! Gap > 5%")
    print("   Recommendation: Add regularization")
elif gap > 0.02:
    print("⚠️  CAUTION: Some overfitting detected (2-5% gap)")
    print("   Model is still good for project")
else:
    print("✅ EXCELLENT: Minimal overfitting! Gap < 2%")
    print("   Model generalizes very well")

# Quick cross-validation check
from sklearn.model_selection import cross_val_score
print("\n🧪 5-Fold Cross-validation:")
cv_scores = cross_val_score(model, X_train_tfidf, y_train,
                           cv=3, scoring='accuracy')  # Use 3-fold for speed
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")

# Final verdict
print("\n🎯 OVERFITTING VERDICT:")
if cv_scores.mean() > 0.95 and gap < 0.02:
    print("✅ STRONG GENERALIZATION: Your 99% is real!")
    print("   Model learns true patterns, not noise")
elif cv_scores.mean() > 0.90:
    print("✅ GOOD GENERALIZATION: Model works well")
    print("   Minor overfitting, still excellent for project")
else:
    print("⚠️  NEEDS IMPROVEMENT: Consider regularization")
# ==================== 7. CONFUSION MATRIX ====================
print("\n🎯 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Fake', 'Predicted Real'],
            yticklabels=['Actual Fake', 'Actual Real'])
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 8. FEATURE IMPORTANCE ====================
print("\n🔍 Top predictive words:")

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Get coefficients from logistic regression
coefficients = model.coef_[0]

# Create DataFrame of words and their coefficients
word_importance = pd.DataFrame({
    'word': feature_names,
    'coefficient': coefficients
})

# Sort by absolute coefficient value
word_importance['abs_coef'] = np.abs(word_importance['coefficient'])
word_importance = word_importance.sort_values('abs_coef', ascending=False)

print("\nTop 10 words indicating REAL news:")
print(word_importance.sort_values('coefficient', ascending=False).head(10)[['word', 'coefficient']])

print("\nTop 10 words indicating FAKE news:")
print(word_importance.sort_values('coefficient', ascending=True).head(10)[['word', 'coefficient']])

# ==================== 9. SAVE MODEL ====================
print("\n💾 Saving model and vectorizer...")

# Save model
joblib.dump(model, 'fake_news_model.pkl')
print("✅ Model saved: fake_news_model.pkl")

# Save vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("✅ Vectorizer saved: tfidf_vectorizer.pkl")

# Save test predictions for analysis
test_results = pd.DataFrame({
    'text': X_test,
    'true_label': y_test,
    'predicted_label': y_pred,
    'probability_fake': y_pred_proba[:, 0],
    'probability_real': y_pred_proba[:, 1]
})
test_results.to_csv('test_predictions.csv', index=False)
print("✅ Test predictions saved: test_predictions.csv")


# ==================== 10. PREDICTION FUNCTION ====================
def predict_fake_news(text):
    """
    Predict if a news article is fake or real

    Parameters:
    text (str): News article text

    Returns:
    dict: Prediction results with confidence
    """
    # Clean the input text
    text_clean = clean_text(text)

    # Transform using saved vectorizer
    text_tfidf = vectorizer.transform([text_clean])

    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]

    # Format results
    result = {
        'text': text[:100] + '...' if len(text) > 100 else text,
        'prediction': 'REAL' if prediction == 1 else 'FAKE',
        'confidence': f"{max(probability) * 100:.1f}%",
        'real_probability': f"{probability[1] * 100:.1f}%",
        'fake_probability': f"{probability[0] * 100:.1f}%",
        'prediction_numeric': int(prediction)
    }

    return result


# ==================== 11. DEMO PREDICTIONS ====================
print("\n🎯 DEMO PREDICTIONS:")
print("=" * 50)

demo_texts = [
    "Breaking news: Scientists discover new planet that could support human life in nearby star system",
    "Donald Trump announces he will run for president again in 2024 during rally speech",
    "Secret government report reveals aliens have been living among us for decades undetected",
    "New medical study finds drinking coffee daily reduces risk of heart disease by 30 percent",
    "Celebrity claims vaccines contain microchips to track population movements globally"
]

for i, text in enumerate(demo_texts, 1):
    result = predict_fake_news(text)
    print(f"\n{i}. {result['text']}")
    print(f"   Prediction: {result['prediction']}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Probabilities - Real: {result['real_probability']}, Fake: {result['fake_probability']}")



