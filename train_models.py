import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/nyambene_final_dashboard_dataset.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("NYAMBENE MATERNAL DELIVERY MODEL RETRAINING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

FEATURES = [
    "AGE",
    "ANC",
    "Previous_Births",
    "Previous_Losses",
    "Gravidae",
    "Diagnosis",
    "Gestation",
]

TARGET = "Delivery_Type"


# Check required columns
required_columns = FEATURES + [TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# PREPARE TARGET
# ============================================================

model_df = df[required_columns].copy()

# Remove records where the delivery target is missing
model_df = model_df.dropna(subset=[TARGET])

# Normalize target text
model_df[TARGET] = (
    model_df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("\nOriginal target values:")
print(model_df[TARGET].value_counts())


# Keep only the two classes used by the application
model_df = model_df[
    model_df[TARGET].isin([
        "normal",
        "caesarean"
    ])
].copy()


# Convert target to the numeric classes expected by app.py
#
# 0 = Normal
# 1 = Caesarean
#
model_df["TARGET_NUMERIC"] = model_df[TARGET].map({
    "normal": 0,
    "caesarean": 1
})


print("\nTarget after preparation:")
print(
    model_df["TARGET_NUMERIC"]
    .value_counts()
    .sort_index()
)


# ============================================================
# FEATURES / TARGET
# ============================================================

X = model_df[FEATURES].copy()
y = model_df["TARGET_NUMERIC"].copy()


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

numeric_features = [
    "AGE",
    "ANC",
    "Previous_Births",
    "Previous_Losses",
    "Gravidae",
    "Gestation",
]

categorical_features = [
    "Diagnosis",
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# ============================================================
# MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        ),
    ]
)

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)

print(
    "\nLogistic Regression Accuracy:",
    round(accuracy_score(y_test, logistic_pred), 4)
)

print(
    classification_report(
        y_test,
        logistic_pred,
        target_names=["Normal", "Caesarean"],
        zero_division=0
    )
)


# ============================================================
# MODEL 2 — RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        ),
    ]
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print(
    "\nRandom Forest Accuracy:",
    round(accuracy_score(y_test, rf_pred), 4)
)

print(
    classification_report(
        y_test,
        rf_pred,
        target_names=["Normal", "Caesarean"],
        zero_division=0
    )
)


# ============================================================
# MODEL 3 — HISTOGRAM GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING HISTGRADIENTBOOSTING")
print("=" * 70)

hgb_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                random_state=42
            )
        ),
    ]
)

hgb_model.fit(X_train, y_train)

hgb_pred = hgb_model.predict(X_test)

print(
    "\nHistGradientBoosting Accuracy:",
    round(accuracy_score(y_test, hgb_pred), 4)
)

print(
    classification_report(
        y_test,
        hgb_pred,
        target_names=["Normal", "Caesarean"],
        zero_division=0
    )
)


# ============================================================
# SAVE MODELS
# ============================================================

logistic_path = os.path.join(
    MODEL_DIR,
    "delivery_logistic_model.pkl"
)

rf_path = os.path.join(
    MODEL_DIR,
    "delivery_random_forest_model.pkl"
)

hgb_path = os.path.join(
    MODEL_DIR,
    "delivery_hgb_model.pkl"
)


joblib.dump(
    logistic_model,
    logistic_path
)

joblib.dump(
    rf_model,
    rf_path
)

joblib.dump(
    hgb_model,
    hgb_path
)


# ============================================================
# VERIFY SAVED MODELS
# ============================================================

print("\n" + "=" * 70)
print("MODELS SAVED SUCCESSFULLY")
print("=" * 70)

print("\n", logistic_path)
print("\n", rf_path)
print("\n", hgb_path)


# ============================================================
# TEST RELOADING
# ============================================================

print("\n" + "=" * 70)
print("TESTING MODEL RELOADING")
print("=" * 70)

loaded_logistic = joblib.load(logistic_path)
loaded_rf = joblib.load(rf_path)
loaded_hgb = joblib.load(hgb_path)

print("\nAll three models loaded successfully.")


# Test one real record
sample = X_test.iloc[[0]]

print("\nTesting prediction on one test record:")

print(
    "\nLogistic:",
    loaded_logistic.predict(sample)[0],
    loaded_logistic.predict_proba(sample)[0]
)

print(
    "Random Forest:",
    loaded_rf.predict(sample)[0],
    loaded_rf.predict_proba(sample)[0]
)

print(
    "HistGradientBoosting:",
    loaded_hgb.predict(sample)[0],
    loaded_hgb.predict_proba(sample)[0]
)


print("\n" + "=" * 70)
print("RETRAINING COMPLETE")
print("=" * 70)
print("\nYou can now run:")
print("streamlit run app.py")
