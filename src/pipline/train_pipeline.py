import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "..", "..", "archivedata", "emails.csv")
artifact_dir = os.path.join(BASE_DIR, "..", "..", "artifacts")

os.makedirs(artifact_dir, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(data_path)

print("Columns:", df.columns)

# ---------------- FEATURES ----------------
# text column
X = df["message"]

# label from filename (spam / ham)
y = df["file"].apply(lambda x: 1 if "spam" in str(x).lower() else 0)

# ---------------- VECTORIZATION ----------------
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# ---------------- MODEL ----------------
model = MultinomialNB()
model.fit(X_vec, y)

# ---------------- SAVE ----------------
model_path = os.path.join(artifact_dir, "model.pkl")
vectorizer_path = os.path.join(artifact_dir, "preprocessing.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully!")