# -*- coding: utf-8 -*-


import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("TfidfVectorizer.pkl")

# Streamlit app
st.title("Spam Detection App")
st.write("Enter a message to check if it is spam or not.")

# User input
user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform input text using the vectorizer
        input_vector = vectorizer.transform([user_input])

        # Predict using the model
        prediction = model.predict(input_vector)

        # Display the result
        if prediction[0] == 1:
            st.error("This message is SPAM!")
        else:
            st.success("This message is NOT SPAM.")
