
import pandas as pd
import ast
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("movies_final.csv")

def extract_genre_names(genre_str):
    try:
        return ast.literal_eval(genre_str)
    except:
        return []

df['genres'] = df['genres'].apply(extract_genre_names)
df = df[df['genres'].map(len) > 0]

df = df[df['clean_overview'].notnull()]
df = df[df['clean_overview'].str.strip() != ""]

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_overview'])

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

print(" Genre classes:", mlb.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Training model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print("✅ Model training complete.")

y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

joblib.dump(model, 'genre_predictor_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(mlb, 'mlb.pkl')
print("\n✅ Model and components saved!")
