import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import re


# -----------------------------
# Load saved model components
# -----------------------------

model = joblib.load("../models/hybrid_logistic_regression.pkl")
tfidf = joblib.load("../models/tfidf_vectorizer.pkl")
encoder = joblib.load("../models/categorical_encoder.pkl")
metadata_columns = joblib.load("../models/metadata_columns.pkl")
categorical_columns = joblib.load("../models/categorical_columns.pkl")


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
# Get job advertisement input
# -----------------------------

print("\n" + "=" * 50)
print("FRAUDULENT JOB ADVERTISEMENT DETECTOR")
print("=" * 50)

title = input("\nEnter job title: ")

description = input(
    "\nEnter job description:\n"
)

company_profile = input(
    "\nEnter company profile (press Enter if unavailable): "
)

requirements = input(
    "\nEnter job requirements (press Enter if unavailable): "
)

benefits = input(
    "\nEnter job benefits (press Enter if unavailable): "
)


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
# TF-IDF transformation
# -----------------------------

text_features = tfidf.transform([cleaned_text])


# -----------------------------
# Collect metadata
# -----------------------------

print("\n--- Job Metadata ---")

telecommuting = input(
    "Telecommuting? (t/f): "
).strip().lower()

has_company_logo = input(
    "Has company logo? (t/f): "
).strip().lower()

has_questions = input(
    "Has screening questions? (t/f): "
).strip().lower()

employment_type = input(
    "Employment type: "
).strip()

required_experience = input(
    "Required experience: "
).strip()

required_education = input(
    "Required education: "
).strip()

industry = input(
    "Industry: "
).strip()

function = input(
    "Job function: "
).strip()


# -----------------------------
# Create categorical features
# -----------------------------

categorical_data = pd.DataFrame([{
    "employment_type": employment_type or "Missing",
    "required_experience": required_experience or "Missing",
    "required_education": required_education or "Missing",
    "industry": industry or "Missing",
    "function": function or "Missing"
}])

categorical_features = encoder.transform(categorical_data)


# -----------------------------
# Create metadata features
# -----------------------------

metadata = pd.DataFrame([{
    "telecommuting": int(telecommuting == "t"),
    "has_company_logo": int(has_company_logo == "t"),
    "has_questions": int(has_questions == "t"),

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

metadata_features = csr_matrix(metadata.values)


# -----------------------------
# Combine all features
# -----------------------------

final_features = hstack([
    text_features,
    categorical_features,
    metadata_features
])


# -----------------------------
# Prediction
# -----------------------------

fraud_score = model.predict_proba(final_features)[0][1]

threshold = 0.85

if fraud_score >= threshold:
    prediction = "FRAUDULENT"
else:
    prediction = "LEGITIMATE"


# -----------------------------
# Display result
# -----------------------------

print("\n" + "=" * 50)
print("PREDICTION RESULT")
print("=" * 50)

print(f"\nFraud score: {fraud_score * 100:.2f}%")
print(f"Prediction: {prediction}")
print(f"Classification threshold: {threshold}")

print("=" * 50)