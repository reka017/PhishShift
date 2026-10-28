
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("messages.csv")

# Clean the data
data = data.dropna(subset=["message", "label"])
data["message"] = data["message"].astype(str)
data["label"] = data["label"].str.strip().str.lower()

# Convert labels to numbers
data["label"] = data["label"].map({
    "safe": 0,
    "phishing": 1
})
data = data.dropna(subset=["label"])
data["label"] = data["label"].astype(int)

X = data["message"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Model training completed!")
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
print(classification_report(
    y_test,
    predictions,
    target_names=["Safe", "Phishing"],
    zero_division=0
))

# Save model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/phishshift_model.pkl")

print("Model saved successfully!")
