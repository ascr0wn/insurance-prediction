import numpy as np
import pytest
from src.regression import MyLinearRegression

def test_initialization():
    """Test that the model initializes with correct default values."""
    model = MyLinearRegression(learning_rate=0.05, iterations=500)
    assert model.lr == 0.05
    assert model.iterations == 500
    assert model.weights is None
    assert model.bias is None
    assert model.cost_history == []

def test_fit_simple_data():
    """Test fitting on a simple dataset (y = 2x + 1)."""
    # Create simple data
    X = np.array([[1], [2], [3], [4], [5]])
    # y = 2 * x + 1
    y = np.array([3, 5, 7, 9, 11])

    model = MyLinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X, y)

    assert model.weights is not None
    assert model.bias is not None

    # Check if weights are close to expected values (within a tolerance)
    # With enough iterations, weight should be close to 2 and bias to 1
    # Note: Gradient descent might not be exact without many iterations/tuning

    assert np.allclose(model.weights, [2.0], atol=0.1)
    assert np.isclose(model.bias, 1.0, atol=0.1)

def test_predict():
    """Test prediction logic."""
    model = MyLinearRegression()
    model.weights = np.array([2.0])
    model.bias = 1.0

    X_new = np.array([[6]])
    prediction = model.predict(X_new)

    # Expected: 2*6 + 1 = 13
    assert prediction[0] == 13.0

def test_predict_without_fit_raises_error():
    """Test that predict raises an error if called before fit."""
    model = MyLinearRegression()
    X_new = np.array([[1]])
    with pytest.raises(ValueError):
        model.predict(X_new)

def test_cost_history_recorded():
    """Test that cost history is populated during training."""
    X = np.array([[1], [2]])
    y = np.array([2, 4])

    model = MyLinearRegression(iterations=10)
    model.fit(X, y)

    assert len(model.cost_history) == 10
    # Cost should generally decrease
    assert model.cost_history[-1] < model.cost_history[0]
