# FAKE NEWS DETECTION USING MACHINE LEARNING

# Step 1. Import Libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Download stopwords (run once)
nltk.download('stopwords')

# Step 2. Load Dataset

df = pd.read_csv("Fake_news.csv")   
print(" Dataset Loaded Successfully!")
print(df.head())


# Step 3. Data Cleaning

df = df.dropna()   

ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)     # keep only letters
    text = text.lower()                       # lowercase
    text = text.split()                       # tokenize
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    return ' '.join(text)

df['cleaned_text'] = df['text'].apply(clean_text)

print("\n Text Cleaning Completed!")
print(df[['text', 'cleaned_text']].head())

# Step 4. Feature Extraction (TF-IDF)

x = df['cleaned_text']
y = df['label']  # label column should be 'FAKE' or 'REAL'

vectorizer = TfidfVectorizer(max_features=5000)
x = vectorizer.fit_transform(x).toarray()

print("\n TF-IDF Vectorization Completed!")
print("Feature shape:", x.shape)

# Step 5. Train/Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Step 6. Model Training (Logistic Regression)

model = LogisticRegression()
model.fit(x_train, y_train)


# Step 7. Model Evaluation

y_pred = model.predict(x_test)

print("\n Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8. Save Model & Vectorizer

pickle.dump(model, open('fake_news_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
print("\n Model and Vectorizer Saved Successfully!")

# Step 9. Test Prediction on Custom Input with History Record

# Initialize a list to store prediction history
prediction_history = []

sample_news = """The government announced a new scheme for youth employment in 2025."""
transformed_input = vectorizer.transform([sample_news]).toarray()
result = model.predict(transformed_input)

# Get confidence if available (assuming LogisticRegression has predict_proba)
if hasattr(model, 'predict_proba'):
    prob = model.predict_proba(transformed_input)[0]
    confidence = max(prob) * 100
else:
    confidence = None

print("\n Prediction for sample input:", result[0])

# Record the prediction in history
history_entry = {
    'input_text': sample_news[:100] + '...' if len(sample_news) > 100 else sample_news,  # Truncate for brevity
    'prediction': result[0],
    'confidence': confidence
}
prediction_history.append(history_entry)

print("\n Prediction History:")
for i, entry in enumerate(prediction_history, 1):
    print(f"{i}. Text: '{entry['input_text']}' -> Prediction: {entry['prediction']}" + (f" (Confidence: {entry['confidence']:.2f}%)" if entry['confidence'] else ""))

# Optional: To persist history across runs, you could save to a file (e.g., JSON)
# import json
# with open('prediction_history.json', 'w') as f:
#     json.dump(prediction_history, f)
