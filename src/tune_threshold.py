import os
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

from preprocessing import clean_text
from features import create_metadata_features, create_combined_text


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR, "data", "raw", "Dataset.csv"
)

print("=" * 60)
print("XGBOOST THRESHOLD TUNING")
print("=" * 60)


# --------------------------------------------------
# 2. LOAD AND PREPARE DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["combined_text"] = create_combined_text(df)
df["clean_text"] = df["combined_text"].apply(clean_text)
df["target"] = df["fraudulent"].map({"f": 0, "t": 1})

if df["target"].isna().any():
    raise ValueError("Unexpected values in the target column.")

metadata = create_metadata_features(df)

categorical_columns = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

categorical_data = df[categorical_columns].fillna("Missing")

# First reserve 20% as the final test set.
(
    X_development,
    X_test,
    y_development,
    y_test,
    metadata_development,
    metadata_test,
    categorical_development,
    categorical_test,
) = train_test_split(
    df["clean_text"],
    df["target"],
    metadata,
    categorical_data,
    test_size=0.20,
    random_state=42,
    stratify=df["target"],
)

# Split the remaining 80% into training and validation.
(
    X_train,
    X_validation,
    y_train,
    y_validation,
    metadata_train,
    metadata_validation,
    categorical_train,
    categorical_validation,
) = train_test_split(
    X_development,
    y_development,
    metadata_development,
    categorical_development,
    test_size=0.20,
    random_state=42,
    stratify=y_development,
)

print("\nTraining samples:", len(y_train))
print("Validation samples:", len(y_validation))
print("Final test samples:", len(y_test))


# --------------------------------------------------
# 3. FIT FEATURE TRANSFORMATIONS ON TRAINING DATA
# --------------------------------------------------

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_validation_tfidf = tfidf.transform(X_validation)
X_test_tfidf = tfidf.transform(X_test)

print("Creating categorical features...")

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True,
)

categorical_train_encoded = encoder.fit_transform(
    categorical_train
)

categorical_validation_encoded = encoder.transform(
    categorical_validation
)

categorical_test_encoded = encoder.transform(
    categorical_test
)


# --------------------------------------------------
# 4. COMBINE HYBRID FEATURES
# --------------------------------------------------

X_train_hybrid = hstack([
    X_train_tfidf,
    categorical_train_encoded,
    csr_matrix(metadata_train.values),
]).tocsr()

X_validation_hybrid = hstack([
    X_validation_tfidf,
    categorical_validation_encoded,
    csr_matrix(metadata_validation.values),
]).tocsr()

X_test_hybrid = hstack([
    X_test_tfidf,
    categorical_test_encoded,
    csr_matrix(metadata_test.values),
]).tocsr()


# --------------------------------------------------
# 5. TRAIN XGBOOST
# --------------------------------------------------

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

model = XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=negative_count / positive_count,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

print("\nTraining XGBoost...")
model.fit(X_train_hybrid, y_train)

# Obtain probabilities for validation and final test sets.
validation_probabilities = model.predict_proba(
    X_validation_hybrid
)[:, 1]

test_probabilities = model.predict_proba(
    X_test_hybrid
)[:, 1]


# --------------------------------------------------
# 6. CHOOSE THRESHOLD USING VALIDATION DATA ONLY
# --------------------------------------------------

print("\nEvaluating thresholds on validation data...")

thresholds = np.arange(0.10, 0.91, 0.05)
validation_results = []

for threshold in thresholds:
    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    validation_results.append({
        "Threshold": round(float(threshold), 2),
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    })

validation_df = pd.DataFrame(validation_results)

print("\nValidation results:")
print(validation_df.round(4).to_string(index=False))

# Choose the threshold with the highest validation F1-score.
best_row = validation_df.loc[
    validation_df["F1-score"].idxmax()
]

best_threshold = float(best_row["Threshold"])

print("\nSelected threshold:", best_threshold)
print("Validation F1-score:", round(best_row["F1-score"], 4))


# --------------------------------------------------
# 7. FINAL EVALUATION ON UNTOUCHED TEST SET
# --------------------------------------------------

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Selected threshold: {best_threshold:.2f}")
print(f"Accuracy: {accuracy_score(y_test, test_predictions):.4f}")
print(
    "Fraud precision:",
    f"{precision_score(y_test, test_predictions, zero_division=0):.4f}",
)
print(
    "Fraud recall:",
    f"{recall_score(y_test, test_predictions, zero_division=0):.4f}",
)
print(
    "Fraud F1-score:",
    f"{f1_score(y_test, test_predictions, zero_division=0):.4f}",
)
print(
    "ROC-AUC:",
    f"{roc_auc_score(y_test, test_probabilities):.4f}",
)
print(
    "PR-AUC:",
    f"{average_precision_score(y_test, test_probabilities):.4f}",
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_predictions))


# --------------------------------------------------
# 8. SAVE VALIDATION RESULTS
# --------------------------------------------------

results_path = os.path.join(
    PROJECT_DIR,
    "models",
    "xgboost_threshold_results.csv",
)

validation_df.to_csv(results_path, index=False)

print("\nValidation threshold results saved to:")
print(results_path)

print("\nThreshold tuning completed!")


# Save the threshold-tuned model and its matching preprocessing files
import joblib

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "hybrid_xgboost_threshold_070.pkl")
)

joblib.dump(
    tfidf,
    os.path.join(MODEL_DIR, "xgboost_tfidf_vectorizer.pkl")
)

joblib.dump(
    encoder,
    os.path.join(MODEL_DIR, "xgboost_categorical_encoder.pkl")
)

joblib.dump(
    list(metadata.columns),
    os.path.join(MODEL_DIR, "xgboost_metadata_columns.pkl")
)

joblib.dump(
    categorical_columns,
    os.path.join(MODEL_DIR, "xgboost_categorical_columns.pkl")
)

joblib.dump(
    best_threshold,
    os.path.join(MODEL_DIR, "xgboost_threshold.pkl")
)

print("\nThreshold-tuned XGBoost model and matching files saved successfully.")

