import pandas as pd
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#data loading
fake_df=pd.read_csv("C:\\Users\\lenovo\\Downloads\\news\\Fake.csv")
true_df =pd.read_csv("C:\\Users\\lenovo\\Downloads\\news\\True.csv")

# Add label: 1 = fake, 0 = real
fake_df['label'] = 1
true_df['label'] = 0

# Combine and shuffle
df = pd.concat([fake_df, true_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

#tfidf vectrization
#splitting the data into train and test sets
x=df['text']
y=df['label']
vectorizer= TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2), min_df=5)
x_vectorizer=vectorizer.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_vectorizer, y, test_size=0.2, random_state=42)

#model training
model=LogisticRegression()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
print("Accuracy:", accuracy_score(y_test,y_pred))

#model saving
joblib.dump(model,"model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

#streamlit the news
model=joblib.load("model.pkl")
vectorizer=joblib.load("vectorizer.pkl")

st.title("📰 fake news detection")
inputtext=st.text_area("Enter the news headline")
if st.button("check"):
    input_vector=vectorizer.transform([inputtext])
    prediction=model.predict(input_vector)[0]
    prob=model.predict_proba(input_vector)[0][prediction]

    st.write(f"🧠 Prediction: {'Fake' if prediction else 'Real'}")
    st.write(f"🔍 Confidence: {prob * 100:.2f}%")

#uploading the model
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type="csv")
if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)

        if 'text' not in batch_df.columns:
            st.error("❌ Error: The uploaded CSV must contain a 'text' column.")
        else:
            text = batch_df['text']
            vectors = vectorizer.transform(text)
            batch_prediction = model.predict(vectors)
            batch_df['prediction'] = ['Fake' if pred == 0 else 'Real' for pred in batch_prediction]

            st.write("✅ Batch predictions completed. Preview below:")
            st.dataframe(batch_df.head())

            csv = batch_df.to_csv(index=False)
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

            st.success("🎉 File processed and predictions are ready.")
    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the file: {e}")
