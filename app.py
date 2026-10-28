
import streamlit as st
import joblib
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="PhishShift Message Risk Scanner",
    page_icon="🛡️",
    layout="centered"
)

# Load the trained model
model_path = Path("models/phishshift_model.pkl")

st.title("🛡️ PhishShift")
st.subheader("Message Risk Scanner")

st.write(
    "Check a message for possible phishing risks "
    "using a Machine Learning model."
)

if not model_path.exists():
    st.error("Model not found. Please run: python train_model.py")
    st.stop()

model = joblib.load(model_path)

# Message input
message = st.text_area(
    "Enter your message",
    placeholder="Paste your SMS or email message here..."
)

# Scan button
if st.button("Scan Message"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]

        st.divider()

        if prediction == 1:
            st.error("⚠️ Potential phishing message detected!")
            st.write(
                "Avoid clicking suspicious links or sharing "
                "passwords, OTPs, or bank details."
            )
        else:
            st.success("✅ Classified as a safe message.")
            st.write(
                "This model did not flag the message as phishing. "
                "That does not guarantee it is safe."
            )

st.caption(
    "Educational prototype. Verify suspicious messages "
    "through official channels."
)
