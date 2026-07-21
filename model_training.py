import os
import numpy as np
import pickle
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    load_data,
    feature_engineering,
    build_preprocessing_pipeline
)


def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def train_models():

    # create models folder
    os.makedirs("models", exist_ok=True)

    # load data
    df = load_data("data/AmesHousing.csv")
    df = feature_engineering(df)

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # save template row for prediction
    joblib.dump(X_train.iloc[[0]], "models/X_template.pkl")

    preprocessor = build_preprocessing_pipeline(X)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(objective="reg:squarederror")
    }

    params = {
        "Decision Tree": {
            "model__max_depth": [5, 10]
        },
        "Random Forest": {
            "model__n_estimators": [100],
            "model__max_depth": [10]
        },
        "XGBoost": {
            "model__n_estimators": [100],
            "model__learning_rate": [0.1],
            "model__max_depth": [3]
        }
    }

    results = {}
    best_model = None
    best_score = float("inf")

    for name, model in models.items():

        print(f"Training {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        if name in params:
            grid = GridSearchCV(
                pipeline,
                params[name],
                cv=3,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            trained = grid.best_estimator_
        else:
            trained = pipeline.fit(X_train, y_train)

        preds = trained.predict(X_test)
        rmse, mae, r2 = evaluate_model(y_test, preds)

        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

        if rmse < best_score:
            best_score = rmse
            best_model = trained

    print("Saving model...")

    joblib.dump(results, "models/model_results.pkl")

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("Training complete. Model saved.")


if __name__ == "__main__":
    train_models()