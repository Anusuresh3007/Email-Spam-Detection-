import streamlit as st
import requests

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧"
)

st.title("📧 Email Spam Detection")

st.write("Enter your email/message below and check whether it is spam.")

email = st.text_area(
    "Enter your message",
    height=150,
    placeholder="Type your email message here..."
)

if st.button("Check Spam"):

    if not email.strip():
        st.warning("Please enter an email message.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json={"email": email}
            )

            if response.status_code == 200:

                result = response.json()["result"]

                if result == "SPAM EMAIL":
                    st.error("🚨 SPAM EMAIL")
                else:
                    st.success("✅ NORMAL EMAIL")

            else:
                st.error("Backend error. Please try again.")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Backend is not running. "
                "Please start the Flask backend first."
            )