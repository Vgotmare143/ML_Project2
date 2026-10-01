import joblib
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

CANDIDATES = {
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, verbosity=0),
    "CatBoost": CatBoostRegressor(iterations=300, learning_rate=0.05, depth=4, verbose=False, allow_writing_files=False),
}


def train_best_model(X_train, X_test, y_train, y_test, model_path):
    """Train every candidate, keep the one with the highest test R², save it."""
    results = {}
    for name, model in CANDIDATES.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "r2": r2_score(y_test, pred),
            "mae": mean_absolute_error(y_test, pred),
        }
        print(f"{name:<18} R2={results[name]['r2']:.3f}  MAE={results[name]['mae']:.2f}")

    best = max(results, key=lambda n: results[n]["r2"])
    joblib.dump(CANDIDATES[best], model_path)
    print(f"\nBest model: {best} -> saved to {model_path}")
    return best, results
