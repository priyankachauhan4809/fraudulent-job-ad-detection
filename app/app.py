import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import hstack, csr_matrix


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models")


# ============================================================
# 2. LOAD SAVED XGBOOST MODEL AND PREPROCESSING COMPONENTS
# ============================================================

@st.cache_resource
def load_model_components():

    model = joblib.load(
        os.path.join(
            MODEL_PATH,
            "hybrid_xgboost_threshold_070.pkl"
        )
    )

    tfidf = joblib.load(
        os.path.join(
            MODEL_PATH,
            "xgboost_tfidf_vectorizer.pkl"
        )
    )

    encoder = joblib.load(
        os.path.join(
            MODEL_PATH,
            "xgboost_categorical_encoder.pkl"
        )
    )

    metadata_columns = joblib.load(
        os.path.join(
            MODEL_PATH,
            "xgboost_metadata_columns.pkl"
        )
    )

    categorical_columns = joblib.load(
        os.path.join(
            MODEL_PATH,
            "xgboost_categorical_columns.pkl"
        )
    )

    threshold = float(
        joblib.load(
            os.path.join(
                MODEL_PATH,
                "xgboost_threshold.pkl"
            )
        )
    )

    return (
        model,
        tfidf,
        encoder,
        metadata_columns,
        categorical_columns,
        threshold
    )


try:
    (
        model,
        tfidf,
        encoder,
        metadata_columns,
        categorical_columns,
        threshold
    ) = load_model_components()

except Exception as error:
    st.error(
        "Unable to load the trained model or its preprocessing files."
    )
    st.error(f"Technical details: {error}")
    st.stop()


# ============================================================
# 3. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# 4. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Job Fraud Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fraudulent Job Advertisement Detector")

st.write(
    "Enter the details of a job advertisement below. "
    "The trained XGBoost machine learning model will "
    "estimate its likelihood of being fraudulent."
)

st.info(
    "This tool provides an automated screening result. "
    "It cannot independently verify whether a company or "
    "job advertisement is genuine."
)

st.divider()


# ============================================================
# 5. JOB ADVERTISEMENT INPUTS
# ============================================================

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
    placeholder="Enter required skills, qualifications and experience..."
)

benefits = st.text_area(
    "Job Benefits",
    height=100,
    placeholder="Enter salary benefits, insurance, leave, etc..."
)
salary_range = st.text_input(
    "Salary Range",
    placeholder="Example: ₹4–6 LPA"
)
job_location = st.text_input(
    "Job Location",
    placeholder="Example: Bengaluru, Karnataka"
)

# ============================================================
# 6. JOB METADATA INPUTS
# ============================================================

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


# ============================================================
# 7. PREDICTION
# ============================================================

if st.button(
    "🔍 Detect Fraud",
    use_container_width=True
):

    if not title.strip() or not description.strip():

        st.warning(
            "Please enter at least the Job Title "
            "and Job Description."
        )

    else:

        try:

            # ------------------------------------------------
            # 7.1 COMBINE AND CLEAN TEXT
            # ------------------------------------------------

            combined_text = (
                title + " " +
                company_profile + " " +
                description + " " +
                requirements + " " +
                benefits
            )

            cleaned_text = clean_text(combined_text)


            # ------------------------------------------------
            # 7.2 TF-IDF FEATURES
            # ------------------------------------------------

            text_features = tfidf.transform(
                [cleaned_text]
            )


            # ------------------------------------------------
            # 7.3 CATEGORICAL FEATURES
            # ------------------------------------------------

            categorical_values = {
                "employment_type": employment_type or "Missing",
                "required_experience": required_experience or "Missing",
                "required_education": required_education or "Missing",
                "industry": industry or "Missing",
                "function": function or "Missing"
            }

            categorical_data = pd.DataFrame(
                [{
                    column: categorical_values[column]
                    for column in categorical_columns
                }],
                columns=categorical_columns
            )

            categorical_features = encoder.transform(
                categorical_data
            )


            # ------------------------------------------------
            # 7.4 METADATA FEATURES
            # ------------------------------------------------

            metadata_values = {
                "telecommuting": int(
                    telecommuting == "Yes"
                ),

                "has_company_logo": int(
                    has_company_logo == "Yes"
                ),

                "has_questions": int(
                    has_questions == "Yes"
                ),

                "company_profile_missing": int(
                    not company_profile.strip()
                ),

                "requirements_missing": int(
                    not requirements.strip()
                ),

                "benefits_missing": int(
                    not benefits.strip()
                ),

                "salary_missing": int(not salary_range.strip()),

                "employment_type_missing": int(
                    not employment_type.strip()
                ),

                "education_missing": int(
                    not required_education.strip()
                ),

                "experience_missing": int(
                    not required_experience.strip()
                ),

                "industry_missing": int(
                    not industry.strip()
                ),

                "function_missing": int(
                    not function.strip()
                ),

                "location_missing": int(not job_location.strip())
            }

            missing_metadata_columns = [
                column
                for column in metadata_columns
                if column not in metadata_values
            ]

            if missing_metadata_columns:
                raise ValueError(
                    "The following metadata columns are not "
                    "provided by the app: "
                    + ", ".join(missing_metadata_columns)
                )

            metadata = pd.DataFrame(
                [{
                    column: metadata_values[column]
                    for column in metadata_columns
                }],
                columns=metadata_columns
            )

            metadata_features = csr_matrix(
                metadata.to_numpy(dtype=float)
            )


            # ------------------------------------------------
            # 7.5 COMBINE FEATURES
            # ------------------------------------------------

            final_features = hstack(
                [
                    text_features,
                    categorical_features,
                    metadata_features
                ],
                format="csr"
            )


            # ------------------------------------------------
            # 7.6 VERIFY FEATURE DIMENSIONS
            # ------------------------------------------------

            expected_features = model.n_features_in_

            actual_features = final_features.shape[1]

            if actual_features != expected_features:

                raise ValueError(
                    f"Feature mismatch: the model expects "
                    f"{expected_features} features, but the app "
                    f"created {actual_features}. The preprocessing "
                    f"and training feature definitions must match."
                )


            # ------------------------------------------------
            # 7.7 CALCULATE FRAUD SCORE
            # ------------------------------------------------

            fraud_score = float(
                model.predict_proba(
                    final_features
                )[0][1]
            )


            # ------------------------------------------------
            # 7.8 CLASSIFY USING THE SELECTED THRESHOLD
            # ------------------------------------------------

            if fraud_score >= threshold:

                prediction = "FRAUDULENT"

            else:

                prediction = "LEGITIMATE"


            # ------------------------------------------------
            # 7.9 DISPLAY PREDICTION
            # ------------------------------------------------

            st.subheader("Prediction Result")

            st.metric(
                "Model Fraud Score",
                f"{fraud_score * 100:.2f}%"
            )

            st.caption(
                f"Classification threshold: {threshold:.2f}"
            )

            st.caption(
                "The score is a model estimate and is not "
                "necessarily a calibrated real-world probability."
            )


            # ------------------------------------------------
            # 7.10 RISK CATEGORY
            # ------------------------------------------------

            if fraud_score < 0.30:

                risk_level = "LOW MODEL SCORE"

            elif fraud_score < 0.70:

                risk_level = "MEDIUM MODEL SCORE"

            else:

                risk_level = "HIGH MODEL SCORE"

            st.write(
                f"**Risk category:** {risk_level}"
            )


            # ------------------------------------------------
            # 7.11 DISPLAY CLASSIFICATION
            # ------------------------------------------------

            if prediction == "FRAUDULENT":

                st.error(
                    "⚠️ Prediction: FRAUDULENT"
                )

                st.write(
                    "The model classified this advertisement "
                    "as potentially fraudulent. Verify the "
                    "employer and job details independently "
                    "before sharing personal information "
                    "or making payments."
                )

            else:

                st.success(
                    "✅ Prediction: LEGITIMATE"
                )

                st.write(
                    "The model classified this advertisement "
                    "as legitimate. This does not guarantee "
                    "that the employer or advertisement is genuine."
                )


            # ------------------------------------------------
            # 7.12 ADVERTISEMENT INDICATORS
            # ------------------------------------------------

            st.subheader(
                "🔎 Advertisement Indicators"
            )

            indicators = []

            if not company_profile.strip():

                indicators.append(
                    "Company profile is missing"
                )

            if not requirements.strip():

                indicators.append(
                    "Job requirements are missing"
                )

            if not benefits.strip():

                indicators.append(
                    "Job benefits are missing"
                )
            if not salary_range.strip():
                indicators.append("Salary range is missing")

            if not job_location.strip():
                indicators.append("Job location is missing")
                
            if telecommuting == "Yes":

                indicators.append(
                    "Telecommuting is enabled"
                )

            if has_company_logo == "No":

                indicators.append(
                    "Company logo is missing"
                )

            if has_questions == "No":

                indicators.append(
                    "No screening questions provided"
                )

            if not employment_type.strip():

                indicators.append(
                    "Employment type is missing"
                )

            if not required_experience.strip():

                indicators.append(
                    "Required experience is missing"
                )

            if not required_education.strip():

                indicators.append(
                    "Required education is missing"
                )

            if not industry.strip():

                indicators.append(
                    "Industry is missing"
                )

            if not function.strip():

                indicators.append(
                    "Job function is missing"
                )

            if indicators:

                for indicator in indicators:

                    st.write(
                        f"• {indicator}"
                    )

            else:

                st.write(
                    "No missing-information indicators "
                    "were detected among the fields checked."
                )


            # ------------------------------------------------
            # 7.13 GLOBAL MODEL FEATURE IMPORTANCE
            # ------------------------------------------------

            st.subheader(
                "🧠 Important Model Features"
            )

            feature_names = np.array(
                list(
                    tfidf.get_feature_names_out()
                )
                +
                list(
                    encoder.get_feature_names_out(
                        categorical_columns
                    )
                )
                +
                list(metadata_columns)
            )

            importances = model.feature_importances_

            if len(feature_names) != len(importances):

                st.warning(
                    "Feature names do not match the model's "
                    "feature importance array. The explanation "
                    "has been omitted to avoid misleading results."
                )

            else:

                top_indices = np.argsort(
                    importances
                )[-5:][::-1]

                displayed_features = [
                    index
                    for index in top_indices
                    if importances[index] > 0
                ]

                if displayed_features:

                    st.write(
                        "The following features have high "
                        "overall importance in the trained model. "
                        "They are not necessarily the reasons "
                        "for this individual prediction."
                    )

                    for index in displayed_features:

                        st.write(
                            f"• **{feature_names[index]}** — "
                            f"importance: "
                            f"{importances[index]:.4f}"
                        )

                else:

                    st.write(
                        "No non-zero feature importance values "
                        "were available to display."
                    )


        except Exception as error:

            st.error(
                "The prediction could not be completed."
            )

            st.exception(error)