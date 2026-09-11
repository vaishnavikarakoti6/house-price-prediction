import pandas as pd
import numpy as np
from data_preprocessing import feature_engineering, build_preprocessing_pipeline


def test_feature_engineering_drops_pid_and_logs_price():
    df = pd.DataFrame({
        "PID": [1, 2, 3],
        "SalePrice": [100000, 200000, 300000],
        "GrLivArea": [1500, 1800, 2000]
    })

    result = feature_engineering(df)

    assert "PID" not in result.columns
    assert np.isclose(result["SalePrice"].iloc[0], np.log1p(100000))


def test_build_preprocessing_pipeline_returns_transformer():
    X = pd.DataFrame({
        "GrLivArea": [1500, 1800, 2000],
        "Neighborhood": ["NAmes", "CollgCr", "OldTown"]
    })

    preprocessor = build_preprocessing_pipeline(X)

    # Should have both a numeric and categorical branch
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names
