import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from matplotlib import pyplot as plt


def evaluate_model(y_true, y_pred) -> dict:
    """
    Return evaluation metrics.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred)
    }


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features to enhance model learning.
    """
    df = df.copy()
    df["Area_per_Bedroom"] = df["Area"] / (df["Bedrooms"] + 0.1)
    df["Bathroom_to_Bedroom"] = df["Bathrooms"] / (df["Bedrooms"] + 0.1)
    return df


def run_models(df: pd.DataFrame, target: str):
    df = feature_engineering(df)

    features = ["Price/m²", "Area", "Bedrooms", "Bathrooms", "Area_per_Bedroom", "Bathroom_to_Bedroom"]
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    metrics_rf = evaluate_model(y_test, y_pred_rf)
    metrics_rf["Model"] = "Random Forest"
    fig_rf = plot_importance(rf, X, title="Random Forest Feature Importance")
    results.append((metrics_rf, fig_rf))

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    metrics_xgb = evaluate_model(y_test, y_pred_xgb)
    metrics_xgb["Model"] = "XGBoost"
    fig_xgb = plot_importance(xgb, X, title="XGBoost Feature Importance")
    results.append((metrics_xgb, fig_xgb))

    # CatBoost
    cat = CatBoostRegressor(verbose=0, random_state=42)
    cat.fit(X_train, y_train)
    y_pred_cat = cat.predict(X_test)
    metrics_cat = evaluate_model(y_test, y_pred_cat)
    metrics_cat["Model"] = "CatBoost"
    fig_cat = plot_importance(cat, X, title="CatBoost Feature Importance")
    results.append((metrics_cat, fig_cat))

    return results


def plot_importance(model, X, title="Feature Importances"):
    importances = model.feature_importances_
    features = X.columns
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(features[indices], rotation=45)
    fig.tight_layout()
    return fig
