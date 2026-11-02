import os
import numpy as np
import pandas as pd
import joblib

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold

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

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

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
print("\nChecking duplicate labels in the complete dataset...")

duplicate_groups = (
    df.groupby("clean_text")["target"]
    .nunique()
)

conflicting_groups = duplicate_groups[
    duplicate_groups > 1
]

print(
    "Total duplicate text groups:",
    df["clean_text"].duplicated().sum()
)

print(
    "Duplicate groups with conflicting labels:",
    len(conflicting_groups)
)

print(
    "Rows belonging to conflicting groups:",
    df["clean_text"].isin(conflicting_groups.index).sum()
)
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

# --------------------------------------------------
# 3. SPLIT DATA WITHOUT DUPLICATE TEXT OVERLAP
# --------------------------------------------------

print("\nCreating duplicate-aware data splits...")

groups = df["clean_text"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

development_indices, test_indices = next(
    splitter.split(
        np.zeros(len(df)),
        df["target"],
        groups
    )
)

development_df = df.iloc[development_indices]

development_metadata = metadata.iloc[development_indices]

development_categorical = categorical_data.iloc[development_indices]

development_text = development_df["clean_text"]
development_target = development_df["target"]
development_groups = groups.iloc[development_indices]


second_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_relative_indices, validation_relative_indices = next(
    second_splitter.split(
        np.zeros(len(development_df)),
        development_target,
        development_groups
    )
)

X_train = development_text.iloc[train_relative_indices]
X_validation = development_text.iloc[validation_relative_indices]
X_test = df["clean_text"].iloc[test_indices]

y_train = development_target.iloc[train_relative_indices]
y_validation = development_target.iloc[validation_relative_indices]
y_test = df["target"].iloc[test_indices]

metadata_train = development_metadata.iloc[train_relative_indices]
metadata_validation = development_metadata.iloc[validation_relative_indices]
metadata_test = metadata.iloc[test_indices]

categorical_train = development_categorical.iloc[train_relative_indices]
categorical_validation = development_categorical.iloc[validation_relative_indices]
categorical_test = categorical_data.iloc[test_indices]

print("\nTraining samples:", len(y_train))
print("Validation samples:", len(y_validation))
print("Final test samples:", len(y_test))

print("\nFraud distribution:")

print("Training:", y_train.value_counts().to_dict())
print("Validation:", y_validation.value_counts().to_dict())
print("Testing:", y_test.value_counts().to_dict())

print("\nDuplicate overlap check:")

print(
    "Train-test:",
    len(set(X_train).intersection(set(X_test)))
)

print(
    "Train-validation:",
    len(set(X_train).intersection(set(X_validation)))
)

print(
    "Validation-test:",
    len(set(X_validation).intersection(set(X_test)))
)

# --------------------------------------------------
# 4. CHECK DUPLICATES AND LABEL CONSISTENCY
# --------------------------------------------------

print("\nChecking duplicate advertisements and labels...")

def check_overlap(text_a, labels_a, text_b, labels_b, name):

    map_a = {}

    for text, label in zip(text_a, labels_a):
        map_a.setdefault(text, set()).add(label)

    map_b = {}

    for text, label in zip(text_b, labels_b):
        map_b.setdefault(text, set()).add(label)

    common_texts = set(map_a).intersection(map_b)
    conflicts = 0

    for text in common_texts:
        if map_a[text] != map_b[text]:
            conflicts += 1

    print(f"\n{name}")
    print("Unique matching texts:", len(common_texts))
    print("Conflicting label groups:", conflicts)


check_overlap(
    X_train, y_train,
    X_test, y_test,
    "TRAIN-TEST"
)

check_overlap(
    X_train, y_train,
    X_validation, y_validation,
    "TRAIN-VALIDATION"
)

check_overlap(
    X_validation, y_validation,
    X_test, y_test,
    "VALIDATION-TEST"
)

print("\nDuplicate investigation completed.")


# --------------------------------------------------
# 5. FIT TF-IDF ON TRAINING DATA
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


# --------------------------------------------------
# 6. CREATE CATEGORICAL FEATURES
# --------------------------------------------------

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
# 7. COMBINE HYBRID FEATURES
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
# 8. TRAIN XGBOOST
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


validation_probabilities = model.predict_proba(
    X_validation_hybrid
)[:, 1]


test_probabilities = model.predict_proba(
    X_test_hybrid
)[:, 1]


# --------------------------------------------------
# 9. CHOOSE THRESHOLD USING VALIDATION DATA
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


best_row = validation_df.loc[
    validation_df["F1-score"].idxmax()
]

best_threshold = float(best_row["Threshold"])

print("\nSelected threshold:", best_threshold)

print(
    "Validation F1-score:",
    round(best_row["F1-score"], 4)
)


# --------------------------------------------------
# 10. FINAL EVALUATION ON TEST DATA
# --------------------------------------------------

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Selected threshold: {best_threshold:.2f}")

print(
    f"Accuracy: {accuracy_score(y_test, test_predictions):.4f}"
)

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
# 11. SAVE VALIDATION RESULTS
# --------------------------------------------------

results_path = os.path.join(
    MODEL_DIR,
    "xgboost_threshold_results.csv",
)

validation_df.to_csv(results_path, index=False)

print("\nValidation results saved to:")

print(results_path)


# --------------------------------------------------
# 12. SAVE MODEL AND PREPROCESSING FILES
# --------------------------------------------------

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "hybrid_xgboost_threshold_tuned.pkl"
    )
)

joblib.dump(
    tfidf,
    os.path.join(
        MODEL_DIR,
        "xgboost_tfidf_vectorizer.pkl"
    )
)

joblib.dump(
    encoder,
    os.path.join(
        MODEL_DIR,
        "xgboost_categorical_encoder.pkl"
    )
)

joblib.dump(
    list(metadata.columns),
    os.path.join(
        MODEL_DIR,
        "xgboost_metadata_columns.pkl"
    )
)

joblib.dump(
    categorical_columns,
    os.path.join(
        MODEL_DIR,
        "xgboost_categorical_columns.pkl"
    )
)

joblib.dump(
    best_threshold,
    os.path.join(
        MODEL_DIR,
        "xgboost_threshold.pkl"
    )
)


print("\nThreshold-tuned XGBoost model and preprocessing files saved successfully.")

print("\nTHRESHOLD TUNING COMPLETED!")