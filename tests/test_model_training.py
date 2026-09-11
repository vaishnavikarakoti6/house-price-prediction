import numpy as np
from model_training import evaluate_model


def test_evaluate_model_returns_three_metrics():
    y_true = np.array([100000, 150000, 200000])
    y_pred = np.array([105000, 145000, 210000])

    rmse, mae, r2 = evaluate_model(y_true, y_pred)

    assert rmse > 0
    assert mae > 0
    assert isinstance(r2, float)
