import streamlit as st
import joblib
import string

# Load saved model components
model = joblib.load('genre_predictor_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
mlb = joblib.load('mlb.pkl')

# Simple text cleaning function
def clean_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    stopwords = {"the", "a", "and", "in", "of", "to", "on", "for", "with", "is", "at", "by"}
    tokens = [word for word in text.split() if word not in stopwords]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="🎬 Movie Genre Predictor")
st.title("🎬 Movie Genre Predictor")
desc = st.text_area("Enter a movie description")

if st.button("Predict Genre"):
    if desc.strip() == "":
        st.warning("Please enter a description.")
    else:
        cleaned = clean_text(desc)
        X_new = vectorizer.transform([cleaned])
        y_pred = model.predict(X_new)[0]  # Single prediction row

        # Match predictions to genre labels
        predicted_genres = [genre for genre, val in zip(mlb.classes_, y_pred) if val == 1]

        if predicted_genres:
            st.success("Predicted Genres: " + ", ".join(predicted_genres))
        else:
            st.info("No confident prediction for this input.")
