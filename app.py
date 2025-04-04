import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model and scaler
model = tf.keras.models.load_model("hybrid_model.h5")
scaler = joblib.load("scaler.pkl")

# Set up Streamlit page
st.set_page_config(page_title="Parkinson's Disease Prediction", layout="wide")

# Custom Styling for Background Image
st.markdown(
    """
    <style>
    .main { background-color: #f0f2f6; }
    .title { text-align: center; font-size: 32px; color: #4CAF50; }
    .subtitle { text-align: center; font-size: 20px; color: #555; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Layout - Two Columns
col1, col2 = st.columns([2, 3])

with col1:
    # Show background image on left side
    st.image("parkinsons_background.jpg", use_container_width=True)

with col2:
    # Title and description
    st.markdown("<h1 style='color: blue;'>Parkinson’s Disease Prediction</h1>", unsafe_allow_html=True)
    st.write("## 🧠 About the Project")
    st.write("""
    - 🏥 **Purpose:** Early detection of Parkinson’s Disease using voice data.
    - 🤖 **Model Used:** Hybrid ANN + CNN model.
    - 📊 **Dataset:** Public dataset with voice recordings from **Kaggle**.
    - 📉 **How it Works:** The model analyzes **voice frequency variations** to detect potential **Parkinson’s Disease**.
    - 🎯 **Goal:** Provide **early-stage detection** to help medical professionals with timely diagnosis.
    """)
st.write("## 🧠 Understanding Parkinson’s Disease")

st.write("""
Parkinson’s disease is a **neurodegenerative disorder** that primarily affects movement. 
It is caused by the progressive loss of dopamine-producing neurons in the brain.

### 🔍 **Common Symptoms**
- **Tremors** (Involuntary shaking, often in hands)
- **Bradykinesia** (Slowed movement)
- **Muscle Rigidity** (Stiffness in limbs and joints)
- **Postural Instability** (Impaired balance and coordination)
- **Speech Changes** (Soft or slurred speech)
- **Facial Expression Loss** (Reduced ability to show emotions)

### 🧪 **Causes & Risk Factors**
- **Age:** Most commonly affects people over **60 years old**, but early-onset cases exist.
- **Genetics:** Certain gene mutations may contribute to the disease.
- **Environmental Triggers:** Exposure to pesticides, toxins, and head injuries may increase risk.
- **Dopamine Deficiency:** Loss of dopamine-producing neurons in the brain.

### 🎯 **Why Early Detection Matters?**
- There is **no cure** for Parkinson’s, but early detection helps manage symptoms with **medications, therapies, and lifestyle changes**.
- Machine learning models, like this one, **analyze voice data** to detect early signs of the disease, helping with **faster diagnosis and better treatment planning**.
""")

# File Upload Section
st.write("### 📂 Upload CSV File for Prediction")
uploaded_file = st.file_uploader("Upload a CSV file containing patient voice data", type=["csv"])

if uploaded_file:
    # Load the uploaded data
    df = pd.read_csv(uploaded_file)

    # Show preview
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    # Standardize the features using the saved scaler
    X_scaled = scaler.transform(df.values)

    # Reshape for CNN
    X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Make predictions
    predictions = model.predict([X_scaled, X_cnn])
    df["Parkinsons_Prediction"] = (predictions > 0.5).astype(int)

    # Show results
    st.subheader("🧾 Prediction Results")
    st.dataframe(df)

    # Count of predictions
    st.subheader("📈 Prediction Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Parkinsons_Prediction"], hue=df["Parkinsons_Prediction"], palette="coolwarm", ax=ax, legend=False)
    ax.set_xticklabels(["No Parkinson's", "Parkinson's"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Download button
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="parkinsons_predictions.csv",
        mime="text/csv",
    )
