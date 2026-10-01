from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "stud.csv"

# The Kaggle CSV uses spaces and slashes in column names; normalise them.
RENAME = {
    "race/ethnicity": "race_ethnicity",
    "parental level of education": "parental_level_of_education",
    "test preparation course": "test_preparation_course",
    "math score": "math_score",
    "reading score": "reading_score",
    "writing score": "writing_score",
}


def ingest(test_size: float = 0.2, seed: int = 42):
    """Load the raw CSV and return (train_df, test_df)."""
    df = pd.read_csv(DATA_PATH).rename(columns=RENAME)
    return train_test_split(df, test_size=test_size, random_state=seed)
