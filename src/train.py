import os
import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from preprocessing import clean_text
from features import create_metadata_features, create_combined_text


# -----------------------------
# 1. Load dataset
# -----------------------------

DATA_PATH = "../data/raw/Dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# -----------------------------
# 2. Prepare text
# -----------------------------

df["combined_text"] = create_combined_text(df)
df["clean_text"] = df["combined_text"].apply(clean_text)

df["target"] = df["fraudulent"].map({
    "f": 0,
    "t": 1
})


# -----------------------------
# 3. Prepare metadata
# -----------------------------

metadata = create_metadata_features(df)

categorical_columns = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

categorical_data = df[categorical_columns].fillna("Missing")


# -----------------------------
# 4. Train-test split
# -----------------------------

X_text_train, X_text_test, y_train, y_test, \
metadata_train, metadata_test, \
categorical_train, categorical_test = train_test_split(
    df["clean_text"],
    df["target"],
    metadata,
    categorical_data,
    test_size=0.20,
    random_state=42,
    stratify=df["target"]
)


# -----------------------------
# 5. TF-IDF
# -----------------------------

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_text_train)
X_test_tfidf = tfidf.transform(X_text_test)


# -----------------------------
# 6. Encode categorical features
# -----------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

categorical_train_encoded = encoder.fit_transform(
    categorical_train
)

categorical_test_encoded = encoder.transform(
    categorical_test
)


# -----------------------------
# 7. Combine features
# -----------------------------

metadata_train_sparse = csr_matrix(
    metadata_train.values
)

metadata_test_sparse = csr_matrix(
    metadata_test.values
)

X_train_hybrid = hstack([
    X_train_tfidf,
    categorical_train_encoded,
    metadata_train_sparse
])

X_test_hybrid = hstack([
    X_test_tfidf,
    categorical_test_encoded,
    metadata_test_sparse
])


print("Hybrid training shape:", X_train_hybrid.shape)
print("Hybrid testing shape:", X_test_hybrid.shape)


# -----------------------------
# 8. Train Logistic Regression
# -----------------------------

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_hybrid,
    y_train
)


# -----------------------------
# 9. Save model and preprocessors
# -----------------------------

MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "hybrid_logistic_regression.pkl")
)

joblib.dump(
    tfidf,
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)

joblib.dump(
    encoder,
    os.path.join(MODEL_DIR, "categorical_encoder.pkl")
)

joblib.dump(
    list(metadata.columns),
    os.path.join(MODEL_DIR, "metadata_columns.pkl")
)

joblib.dump(
    categorical_columns,
    os.path.join(MODEL_DIR, "categorical_columns.pkl")
)


print("\nTraining completed successfully!")
print("Model and preprocessing files saved in:", MODEL_DIR)