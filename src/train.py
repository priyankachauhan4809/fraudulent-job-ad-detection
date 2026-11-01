import os
import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
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

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("FRAUDULENT JOB ADVERTISEMENT DETECTION")
print("Hybrid NLP + Machine Learning Model Comparison")
print("=" * 60)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(df["fraudulent"].value_counts())

df["combined_text"] = create_combined_text(df)
df["clean_text"] = df["combined_text"].apply(clean_text)

# Legitimate = 0, Fraudulent = 1
df["target"] = df["fraudulent"].map({"f": 0, "t": 1})

if df["target"].isna().any():
    raise ValueError(
        "Unexpected values found in the fraudulent column."
    )

# --------------------------------------------------
# 3. PREPARE FEATURES
# --------------------------------------------------

metadata = create_metadata_features(df)

categorical_columns = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

categorical_data = df[categorical_columns].fillna("Missing")

# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

(
    X_text_train,
    X_text_test,
    y_train,
    y_test,
    metadata_train,
    metadata_test,
    categorical_train,
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

print("\nTraining samples:", len(y_train))
print("Testing samples:", len(y_test))

# --------------------------------------------------
# 5. TEXT FEATURES: TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

X_train_tfidf = tfidf.fit_transform(X_text_train)
X_test_tfidf = tfidf.transform(X_text_test)

# --------------------------------------------------
# 6. CATEGORICAL FEATURES
# --------------------------------------------------

print("Creating categorical features...")

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True,
)

categorical_train_encoded = encoder.fit_transform(
    categorical_train
)

categorical_test_encoded = encoder.transform(
    categorical_test
)

# --------------------------------------------------
# 7. METADATA FEATURES
# --------------------------------------------------

metadata_train_sparse = csr_matrix(metadata_train.values)
metadata_test_sparse = csr_matrix(metadata_test.values)

# --------------------------------------------------
# 8. COMBINE HYBRID FEATURES
# --------------------------------------------------

X_train_hybrid = hstack([
    X_train_tfidf,
    categorical_train_encoded,
    metadata_train_sparse,
]).tocsr()

X_test_hybrid = hstack([
    X_test_tfidf,
    categorical_test_encoded,
    metadata_test_sparse,
]).tocsr()

print("\nHybrid training shape:", X_train_hybrid.shape)
print("Hybrid testing shape:", X_test_hybrid.shape)

# --------------------------------------------------
# 9. DEFINE MODELS
# --------------------------------------------------

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = negative_count / positive_count

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=30,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    ),

    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=1,
            random_state=42,
        ),
        n_estimators=100,
        learning_rate=0.5,
        random_state=42,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    ),
}

# AdaBoost needs sample weights to account for class imbalance.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

results = []
trained_models = {}

# --------------------------------------------------
# 10. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    if model_name == "AdaBoost":
        model.fit(
            X_train_hybrid,
            y_train,
            sample_weight=sample_weights,
        )
    else:
        model.fit(X_train_hybrid, y_train)

    # Fraud probability
    y_prob = model.predict_proba(X_test_hybrid)[:, 1]

    # Same initial threshold for every model
    y_pred = (y_prob >= 0.50).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Fraud Precision:   {precision:.4f}")
    print(f"Fraud Recall:      {recall:.4f}")
    print(f"Fraud F1-score:    {f1:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"PR-AUC:            {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Fraud Precision": precision,
        "Fraud Recall": recall,
        "Fraud F1-score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    })

    trained_models[model_name] = model

# --------------------------------------------------
# 11. SAVE TRAINED MODELS
# --------------------------------------------------

model_filenames = {
    "Logistic Regression": "hybrid_logistic_regression.pkl",
    "Random Forest": "hybrid_random_forest.pkl",
    "AdaBoost": "hybrid_adaboost.pkl",
    "XGBoost": "hybrid_xgboost.pkl",
}

print("\nSaving trained models...")

for model_name, model in trained_models.items():
    model_path = os.path.join(
        MODEL_DIR,
        model_filenames[model_name],
    )

    joblib.dump(model, model_path)
    print(f"{model_name} saved to: {model_path}")

# Save preprocessing objects for the hybrid features.
joblib.dump(
    tfidf,
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
)

joblib.dump(
    encoder,
    os.path.join(MODEL_DIR, "categorical_encoder.pkl"),
)

joblib.dump(
    list(metadata.columns),
    os.path.join(MODEL_DIR, "metadata_columns.pkl"),
)

joblib.dump(
    categorical_columns,
    os.path.join(MODEL_DIR, "categorical_columns.pkl"),
)

# --------------------------------------------------
# 12. SAVE COMPARISON RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

# Sort by fraud F1-score for convenient comparison.
results_df = results_df.sort_values(
    by="Fraud F1-score",
    ascending=False,
)

results_path = os.path.join(
    MODEL_DIR,
    "model_comparison_results.csv",
)

results_df.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nResults saved to:", results_path)
print("\nTraining and comparison completed!")