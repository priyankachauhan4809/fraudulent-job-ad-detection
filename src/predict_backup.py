import joblib
import pandas as pd
import os
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
# Prediction function
# -----------------------------

def predict_job(job_data):

    # Combine text fields
    text_fields = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits"
    ]

    combined_text = " ".join(
        str(job_data.get(field, ""))
        for field in text_fields
    )

    cleaned_text = clean_text(combined_text)

    # TF-IDF
    text_features = tfidf.transform([cleaned_text])

    # Categorical features
    categorical_data = pd.DataFrame(
        [{
            column: job_data.get(column, "Missing")
            for column in categorical_columns
        }]
    )

    categorical_data = categorical_data.fillna("Missing")
    categorical_features = encoder.transform(categorical_data)

        # Metadata features
    metadata = pd.DataFrame([{
        "telecommuting": int(job_data.get("telecommuting") == "t"),
        "has_company_logo": int(job_data.get("has_company_logo") == "t"),
        "has_questions": int(job_data.get("has_questions") == "t"),

        "company_profile_missing": int(
            pd.isna(job_data.get("company_profile"))
        ),

        "requirements_missing": int(
            pd.isna(job_data.get("requirements"))
        ),

        "benefits_missing": int(
            pd.isna(job_data.get("benefits"))
        ),

        "salary_missing": int(
            pd.isna(job_data.get("salary_range"))
        ),

        "employment_type_missing": int(
            pd.isna(job_data.get("employment_type"))
        ),

        "education_missing": int(
            pd.isna(job_data.get("required_education"))
        ),

        "experience_missing": int(
            pd.isna(job_data.get("required_experience"))
        ),

        "industry_missing": int(
            pd.isna(job_data.get("industry"))
        ),

        "function_missing": int(
            pd.isna(job_data.get("function"))
        ),

        "location_missing": int(
            pd.isna(job_data.get("location"))
        )
    }])
     

    metadata = metadata[metadata_columns]

    # Convert metadata to sparse matrix
    metadata_features = csr_matrix(metadata.values)

    # Combine all features
    hybrid_features = hstack([
        text_features,
        categorical_features,
        metadata_features
    ])

    # Get fraud score
    fraud_score = model.predict_proba(
        hybrid_features
    )[0, 1]

    # Selected threshold
    threshold = 0.85

    prediction = (
        "FRAUDULENT"
        if fraud_score >= threshold
        else "LEGITIMATE"
    )

    return prediction, fraud_score


# -----------------------------
# Test example
# -----------------------------

if __name__ == "__main__":

    # Load the original dataset
    dataset = pd.read_csv("../data/raw/Dataset.csv")

    # Select the exact advertisement used in our notebook
    sample_idx = 98
    row = dataset.iloc[sample_idx]

    # Build input from the original dataset row
    test_job = {
        "title": row["title"],
        "company_profile": row["company_profile"],
        "description": row["description"],
        "requirements": row["requirements"],
        "benefits": row["benefits"],

        "telecommuting": row["telecommuting"],
        "has_company_logo": row["has_company_logo"],
        "has_questions": row["has_questions"],

        "employment_type": row["employment_type"],
        "required_experience": row["required_experience"],
        "required_education": row["required_education"],
        "industry": row["industry"],
        "function": row["function"],

        "location": row["location"],
        "salary_range": row["salary_range"]
    }

    prediction, score = predict_job(test_job)

    print("Advertisement index:", sample_idx)
    print("Actual label:", row["fraudulent"])
    print("Prediction:", prediction)
    print("Fraud score:", round(score * 100, 2), "%")