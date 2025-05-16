# 🎬 Movie Genre Prediction - Final Notebook

# ✅ Step 1: Imports
import pandas as pd
import ast
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

# ✅ Step 2: Load and preprocess data
df = pd.read_csv("movies_final.csv")

# Convert genres column from string to list of genre names
def extract_genre_names(genre_str):
    try:
        return [g['name'] for g in ast.literal_eval(genre_str)]
    except:
        return []

df['genres'] = df['genres'].apply(extract_genre_names)

# Remove rows with no genres
df = df[df['genres'].map(len) > 0]

# ✅ Step 3: Vectorize text and encode labels
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['clean_overview'].fillna(''))

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

# ✅ Step 4: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 5: Train the model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# ✅ Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# ✅ Step 7: Visualize performance
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.iloc[:-3][['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12,6))
plt.title("Genre-wise Classification Performance")
plt.xlabel("Genre")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ✅ Step 8: Test prediction on new input
def predict_genres(text):
    X_new = vectorizer.transform([text])
    y_pred = model.predict(X_new)
    return mlb.inverse_transform(y_pred)

sample = "A lion saves his jungle from humans and leads his animal kingdom."
print("\nSample Description:", sample)
print("Predicted Genres:", predict_genres(sample)[0])

# ✅ Step 9: Save model and components
joblib.dump(model, 'genre_predictor_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(mlb, 'mlb.pkl')
print("\n✅ Model and components saved!")
