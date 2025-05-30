import streamlit as st
import pickle

# Load models and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

st.set_page_config(page_title="Phishing Email Detector", layout="centered")

st.title("📧 Phishing Email Detector")
st.write("Paste an email below to analyze whether it may be a phishing attempt using machine learning models.")

email_input = st.text_area("✉️ Enter Email Text", height=200)

if st.button("Analyze Email"):
    if not email_input.strip():
        st.warning("Please enter some email text.")
    else:
        # Vectorize input
        email_vec = vectorizer.transform([email_input])

        # Predictions
        nb_pred = nb_model.predict(email_vec)[0]
        svm_pred = svm_model.predict(email_vec)[0]

        # Confidence using model score (for illustration)
        nb_conf = nb_model.score(vectorizer.transform([email_input]), [nb_pred])
        svm_conf = svm_model.score(vectorizer.transform([email_input]), [svm_pred])

        # Decide final verdict
        final_pred = svm_pred if svm_conf > nb_conf else nb_pred
        final_conf = max(svm_conf, nb_conf)

        # Display result
        if final_pred == 1:
            st.error("⚠️ Phishing Email Detected!")
            st.write("This email has features commonly found in phishing attempts.")
        else:
            st.success("✅ Safe Email")
            st.write("This email appears to be legitimate. Stay cautious anyway.")

        st.subheader("🔍 Model Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Multinomial Naive Bayes**")
            st.write("Prediction:", "Phishing" if nb_pred == 1 else "Safe")
            st.write("Confidence:", f"{nb_conf * 100:.1f}%")

        with col2:
            st.markdown("**Support Vector Machine**")
            st.write("Prediction:", "Phishing" if svm_pred == 1 else "Safe")
            st.write("Confidence:", f"{svm_conf * 100:.1f}%")
