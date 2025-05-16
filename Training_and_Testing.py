# ✅ Fixed Training Script - Movie Genre Predictor

import pandas as pd
import ast
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ✅ Step 1: Load cleaned dataset
df = pd.read_csv("movies_final.csv")

# ✅ Step 2: Ensure 'genres' column is parsed as list

def extract_genre_names(genre_str):
    try:
        return ast.literal_eval(genre_str)
    except:
        return []

df['genres'] = df['genres'].apply(extract_genre_names)
df = df[df['genres'].map(len) > 0]

# ✅ Step 3: Remove rows with empty or missing clean_overview
df = df[df['clean_overview'].notnull()]
df = df[df['clean_overview'].str.strip() != ""]

# ✅ Step 4: Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_overview'])

# ✅ Step 5: Encode genre labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

print("✅ Genre classes:", mlb.classes_)

# ✅ Step 6: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 7: Train model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print("✅ Model training complete.")

# ✅ Step 8: Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# ✅ Step 9: Save model and components
joblib.dump(model, 'genre_predictor_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(mlb, 'mlb.pkl')
print("\n✅ Model and components saved!")
