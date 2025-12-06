import pickle
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os


class FakeNewsClassifier:
    def __init__(self, model_path='model/logistic_regression_model.pkl',
                 vectorizer_path='model/tfidf_vectorizer.pkl'):
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

        # Load or train model
        self.load_or_train_model()

    def preprocess_text(self, text):
        """Clean and preprocess input text"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def load_or_train_model(self):
        """Load pre-trained model or train new one"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                print("Loading pre-trained model...")
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                print("✓ Model loaded successfully")
            else:
                print("No pre-trained model found. Training new model...")
                self.train_model()

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Training new model...")
            self.train_model()

    def train_model(self):
        """Train the model from scratch"""
        try:
            # Try to load dataset
            try:
                data_path = 'dataset/'
                true_file = None
                fake_file = None

                # Look for dataset files
                for file in os.listdir(data_path):
                    if 'true' in file.lower() and file.endswith('.csv'):
                        true_file = os.path.join(data_path, file)
                    elif 'fake' in file.lower() and file.endswith('.csv'):
                        fake_file = os.path.join(data_path, file)

                if true_file and fake_file:
                    true_news = pd.read_csv(true_file)
                    fake_news = pd.read_csv(fake_file)

                    # Add labels
                    true_news['label'] = 1  # Real
                    fake_news['label'] = 0  # Fake

                    # Combine
                    df = pd.concat([true_news, fake_news], ignore_index=True)

                    # Create text column
                    if 'title' in df.columns and 'text' in df.columns:
                        df['combined_text'] = df['title'] + ' ' + df['text']
                    else:
                        df['combined_text'] = df.iloc[:, 0]  # Use first column

                    # Preprocess
                    df['combined_text'] = df['combined_text'].apply(self.preprocess_text)

                    # Remove short texts
                    df = df[df['combined_text'].str.len() > 50]

                    X = df['combined_text']
                    y = df['label']

                    print(f"✓ Loaded dataset with {len(X)} samples")

                else:
                    raise FileNotFoundError("Dataset files not found")

            except Exception as e:
                print(f"Dataset loading failed: {e}. Using synthetic data...")
                # Create synthetic data
                np.random.seed(42)
                real_samples = [
                                   "The government announced new economic policies today according to official sources.",
                                   "Researchers at Harvard University published findings in a peer-reviewed journal.",
                                   "The president will address the nation about new healthcare legislation.",
                                   "Official statistics show economic growth for the third consecutive quarter.",
                                   "According to experts, climate change measures are showing positive results."
                               ] * 100

                fake_samples = [
                                   "BREAKING: Shocking conspiracy revealed by anonymous government source!",
                                   "They don't want you to know this secret about COVID vaccines.",
                                   "Leaked documents prove alien existence is being covered up.",
                                   "This one simple trick can cure cancer but doctors hate it!",
                                   "The truth about the moon landing they never taught in school."
                               ] * 100

                X = real_samples + fake_samples
                y = [1] * 500 + [0] * 500

            # Create TF-IDF Vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.7
            )

            # Transform text
            X_tfidf = self.vectorizer.fit_transform(X)

            # Train model
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                C=1.0,
                solver='liblinear',
                penalty='l2'
            )

            self.model.fit(X_tfidf, y)

            # Save model
            os.makedirs('model', exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)

            print("✓ Model trained and saved successfully")

        except Exception as e:
            print(f"✗ Error training model: {e}")
            raise

    def predict(self, text):
        """Predict if text is real or fake news"""
        try:
            # Preprocess
            cleaned_text = self.preprocess_text(text)

            if len(cleaned_text.split()) < 3:
                return {
                    'prediction': 'Insufficient Text',
                    'confidence': 0,
                    'fake_probability': 50.0,
                    'real_probability': 50.0,
                    'key_indicators': ['Text too short for reliable analysis'],
                    'status': 'warning'
                }

            # Transform
            text_tfidf = self.vectorizer.transform([cleaned_text])

            # Predict
            prediction = self.model.predict(text_tfidf)[0]
            probability = self.model.predict_proba(text_tfidf)[0]

            # Get feature importance
            feature_names = self.vectorizer.get_feature_names_out()
            coefficients = self.model.coef_[0]

            # Analyze words in text
            word_scores = {}
            text_words = cleaned_text.split()

            for word in text_words:
                if word in feature_names:
                    idx = np.where(feature_names == word)[0][0]
                    word_scores[word] = coefficients[idx]

            # Sort by importance
            sorted_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

            # Prepare result
            label = "Real News" if prediction == 1 else "Fake News"
            confidence = probability[prediction] * 100

            # Key indicators
            key_indicators = []
            for word, score in sorted_words[:5]:
                indicator_type = "real" if score > 0 else "fake"
                key_indicators.append({
                    'word': word,
                    'score': round(score, 3),
                    'type': indicator_type,
                    'message': f"'{word}' suggests {indicator_type} news"
                })

            return {
                'prediction': label,
                'confidence': round(confidence, 2),
                'fake_probability': round(probability[0] * 100, 2),
                'real_probability': round(probability[1] * 100, 2),
                'key_indicators': key_indicators,
                'text_length': len(cleaned_text.split()),
                'status': 'success'
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'prediction': 'Error',
                'confidence': 0,
                'fake_probability': 0,
                'real_probability': 0,
                'key_indicators': [],
                'status': 'error',
                'error': str(e)
            }


# Create global classifier instance
classifier = FakeNewsClassifier()