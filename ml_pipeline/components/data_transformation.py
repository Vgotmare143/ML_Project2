import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["reading_score", "writing_score"]
CATEGORICAL = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]
TARGET = "math_score"


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def transform(train_df, test_df, preprocessor_path):
    """Fit the preprocessor on train only, apply to both, and persist it."""
    preprocessor = build_preprocessor()
    features = NUMERIC + CATEGORICAL
    X_train = preprocessor.fit_transform(train_df[features])
    X_test = preprocessor.transform(test_df[features])
    joblib.dump(preprocessor, preprocessor_path)
    return X_train, X_test, train_df[TARGET].values, test_df[TARGET].values
