"""Run from the project root:  python -m ml_pipeline.pipelines.train_pipeline"""
from pathlib import Path

from ml_pipeline.components.data_ingestion import ingest
from ml_pipeline.components.data_transformation import transform
from ml_pipeline.components.model_trainer import train_best_model

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    train_df, test_df = ingest()
    X_train, X_test, y_train, y_test = transform(
        train_df, test_df, ARTIFACTS / "preprocessor.pkl"
    )
    train_best_model(X_train, X_test, y_train, y_test, ARTIFACTS / "model.pkl")


if __name__ == "__main__":
    main()
