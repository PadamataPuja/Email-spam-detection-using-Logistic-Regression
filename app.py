import streamlit as st
import joblib
import numpy as np
import os

st.title("Spam Detection App")
st.write("Enter a message to check if it is spam or not.")

# Check if model and vectorizer exist
if os.path.exists("logistic_regression_model.pkl") and os.path.exists("TfidfVectorizer.pkl"):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    # User input
    user_input = st.text_area("Enter your message:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)

            if prediction[0] == 1:
                st.error("This message is SPAM!")
            else:
                st.success("This message is NOT SPAM.")
else:
    st.error("Required model files not found. Please upload 'model.pkl' and 'vectorizer.pkl'.")
