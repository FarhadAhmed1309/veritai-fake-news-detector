import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load datasets
print("Loading dataset...")
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 1   # 1 = Fake
real["label"] = 0   # 0 = Real

# Combine both
df = pd.concat([fake, real], ignore_index=True)

# Combine title + text for better accuracy
df["content"] = df["title"] + " " + df["text"]

# Remove empty rows
df = df[["content", "label"]].dropna()

print(f"Total articles: {len(df)}")
print(f"Fake: {df[df['label']==1].shape[0]}  |  Real: {df[df['label']==0].shape[0]}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)

# Convert text to numbers using TF-IDF
print("Training model...")
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(tfidf,  "model/tfidf.pkl")
print("\nModel saved to model/model.pkl")
print("Vectorizer saved to model/tfidf.pkl")