import streamlit as st
import joblib

# Load the saved pipeline model
model = joblib.load('positive-negative.pkl')

# Streamlit app setup
st.set_page_config(page_title="Restaurant Review Sentiment", page_icon="🍽️")

st.title("🍽️ Restaurant Review Sentiment Analysis")
st.write("Enter a restaurant review below and predict if it's Positive or Negative!")

# Text input from user
user_input = st.text_area("Write your review here:")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        prediction = model.predict([user_input])

        if prediction[0] == 1:
            st.success("✅ Positive Review!")
        else:
            st.error("❌ Negative Review!")

# Footer
st.markdown("---")
st.markdown("Developed by Naidu 🔥 | CS 5710 Project")
