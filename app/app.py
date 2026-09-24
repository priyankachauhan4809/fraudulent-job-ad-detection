import sys
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from scipy.sparse import hstack, csr_matrix
import re

# -----------------------------
# Allow Python to find src files
# -----------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

sys.path.append(SRC_PATH)

# -----------------------------
# Load saved model components
# -----------------------------

MODEL_PATH = os.path.join(PROJECT_ROOT, "models")

model = joblib.load(
    os.path.join(MODEL_PATH, "hybrid_logistic_regression.pkl")
)

tfidf = joblib.load(
    os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl")
)

encoder = joblib.load(
    os.path.join(MODEL_PATH, "categorical_encoder.pkl")
)

metadata_columns = joblib.load(
    os.path.join(MODEL_PATH, "metadata_columns.pkl")
)
categorical_columns = joblib.load(
    os.path.join(MODEL_PATH, "categorical_columns.pkl")
)
# -----------------------------
# Text cleaning
# -----------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Streamlit page
# -----------------------------

st.set_page_config(
    page_title="Job Fraud Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fraudulent Job Advertisement Detector")

st.write(
    "Enter the details of a job advertisement below. "
    "The trained machine learning model will estimate the probability "
    "that the advertisement is fraudulent."
)

st.divider()

# -----------------------------
# Job advertisement inputs
# -----------------------------

title = st.text_input(
    "Job Title",
    placeholder="Example: Software Engineer"
)

description = st.text_area(
    "Job Description",
    height=180,
    placeholder="Paste the job description here..."
)

company_profile = st.text_area(
    "Company Profile",
    height=120,
    placeholder="Enter company information if available..."
)

requirements = st.text_area(
    "Job Requirements",
    height=120,
    placeholder="Enter required skills, qualifications, experience..."
)

benefits = st.text_area(
    "Job Benefits",
    height=100,
    placeholder="Enter salary benefits, insurance, leave, etc..."
)

st.subheader("Job Metadata")

col1, col2 = st.columns(2)

with col1:
    telecommuting = st.selectbox(
        "Telecommuting",
        ["No", "Yes"]
    )

    has_company_logo = st.selectbox(
        "Company Logo",
        ["Yes", "No"]
    )

    has_questions = st.selectbox(
        "Screening Questions",
        ["Yes", "No"]
    )

    employment_type = st.text_input(
        "Employment Type",
        placeholder="Example: Full-time"
    )

with col2:
    required_experience = st.text_input(
        "Required Experience",
        placeholder="Example: Entry level"
    )

    required_education = st.text_input(
        "Required Education",
        placeholder="Example: Bachelor's Degree"
    )

    industry = st.text_input(
        "Industry",
        placeholder="Example: Computer Software"
    )

    function = st.text_input(
        "Job Function",
        placeholder="Example: Engineering"
    )

st.divider()

# -----------------------------
# Prediction button
# -----------------------------

if st.button("🔍 Detect Fraud", use_container_width=True):

    if not title or not description:
        st.warning("Please enter at least the Job Title and Job Description.")

    else:

        # -----------------------------
        # Combine text
        # -----------------------------

        combined_text = (
            title + " " +
            company_profile + " " +
            description + " " +
            requirements + " " +
            benefits
        )

        cleaned_text = clean_text(combined_text)

        # -----------------------------
        # TF-IDF features
        # -----------------------------

        text_features = tfidf.transform([cleaned_text])

        # -----------------------------
        # Categorical features
        # -----------------------------

        categorical_data = pd.DataFrame([{
            "employment_type": employment_type or "Missing",
            "required_experience": required_experience or "Missing",
            "required_education": required_education or "Missing",
            "industry": industry or "Missing",
            "function": function or "Missing"
        }])

        categorical_features = encoder.transform(
            categorical_data
        )

        # -----------------------------
        # Metadata features
        # -----------------------------

        metadata = pd.DataFrame([{
            "telecommuting": int(telecommuting == "Yes"),
            "has_company_logo": int(has_company_logo == "Yes"),
            "has_questions": int(has_questions == "Yes"),

            "company_profile_missing": int(not company_profile),
            "requirements_missing": int(not requirements),
            "benefits_missing": int(not benefits),

            "salary_missing": 1,
            "employment_type_missing": int(not employment_type),
            "education_missing": int(not required_education),
            "experience_missing": int(not required_experience),
            "industry_missing": int(not industry),
            "function_missing": int(not function),

            "location_missing": 1
        }])

        metadata = metadata[metadata_columns]

        metadata_features = csr_matrix(
            metadata.values
        )

        # -----------------------------
        # Combine features
        # -----------------------------

        final_features = hstack([
            text_features,
            categorical_features,
            metadata_features
        ])

        # -----------------------------
        # Prediction
        # -----------------------------

        fraud_score = model.predict_proba(
            final_features
        )[0][1]
        # -----------------------------
# Model-based explanation
# -----------------------------

feature_names = np.array(
    list(tfidf.get_feature_names_out())
    + list(encoder.get_feature_names_out(categorical_columns))
    + metadata_columns
)

feature_values = final_features.toarray()[0]
model_coefficients = model.coef_[0]

contributions = feature_values * model_coefficients

top_indices = np.argsort(contributions)[-5:][::-1]

top_features = []

for index in top_indices:
    if contributions[index] > 0:
        top_features.append(
            (feature_names[index], contributions[index])
        )

        threshold = 0.85

        if fraud_score >= threshold:
            prediction = "FRAUDULENT"
        else:
            prediction = "LEGITIMATE"

        # -----------------------------
        # Display result
        # -----------------------------

        st.subheader("Prediction Result")

        st.metric(
            "Fraud Score",
            f"{fraud_score * 100:.2f}%"
        )
        # -----------------------------
        # Risk level
        # -----------------------------

        if fraud_score < 0.30:
            risk_level = "LOW RISK"
        elif fraud_score < 0.70:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "HIGH RISK"

        st.write(f"**Risk Level:** {risk_level}")

        if prediction == "FRAUDULENT":
            st.error(
                "⚠️ Prediction: FRAUDULENT"
            )
           
            st.write(
                "The model estimates a high probability "
                "that this job advertisement is fraudulent."
            )

        else:
            st.success(
                "✅ Prediction: LEGITIMATE"
            )

            st.write(
                "The model estimates a low probability "
                "that this job advertisement is fraudulent."
            )

        st.caption(
            "Classification threshold: 0.85"
        )
        # -----------------------------
# Advertisement indicators
# -----------------------------

st.subheader("🔎 Advertisement Indicators")

indicators = []

if not company_profile:
    indicators.append("Company profile is missing")

if not requirements:
    indicators.append("Job requirements are missing")

if not benefits:
    indicators.append("Job benefits are missing")

if telecommuting == "Yes":
    indicators.append("Telecommuting is enabled")

if has_company_logo == "No":
    indicators.append("Company logo is missing")

if has_questions == "No":
    indicators.append("No screening questions provided")

if not employment_type:
    indicators.append("Employment type is missing")

if not required_experience:
    indicators.append("Required experience is missing")

if not required_education:
    indicators.append("Required education is missing")

if not industry:
    indicators.append("Industry is missing")

if not function:
    indicators.append("Job function is missing")

if indicators:
    for indicator in indicators:
        st.write("•", indicator)
else:
    st.write("No obvious missing-information indicators detected.")
# -----------------------------
# Model evidence
# -----------------------------

st.subheader("🧠 Model Evidence")

if top_features:
    st.write(
        "These features contributed most strongly toward the "
        "fraudulent classification:"
    )

    for feature, contribution in top_features:
        st.write(
            f"• **{feature}** "
            f"(contribution: {contribution:.4f})"
        )
else:
    st.write(
        "No strong fraud-associated features were detected "
        "for this advertisement."
    )