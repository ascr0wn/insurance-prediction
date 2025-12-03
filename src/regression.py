import numpy as np
from numpy.typing import NDArray

class MyLinearRegression:
    """
    A custom Linear Regression model implementation using Gradient Descent.

    Attributes:
        lr (float): The learning rate for gradient descent.
        iterations (int): The number of iterations to run gradient descent.
        weights (NDArray | None): The weights (coefficients) of the model.
        bias (float | None): The bias (intercept) of the model.
        cost_history (list[float]): A record of the cost (MSE) at each iteration.
    """

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        """
        Initializes the MyLinearRegression model.

        Args:
            learning_rate (float): The step size for parameter updates. Defaults to 0.01.
            iterations (int): The number of passes over the training data. Defaults to 1000.
        """
        self.lr = learning_rate
        self.iterations = iterations
        self.weights: NDArray | None = None
        self.bias: float | None = None
        self.cost_history: list[float] = []

    def fit(self, X: NDArray, y: NDArray) -> None:
        """
        Trains the model using Gradient Descent.

        Args:
            X (NDArray): The feature matrix of shape (n_samples, n_features).
            y (NDArray): The target values of shape (n_samples,).
        """
        # 1. Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Start weights at 0
        self.bias = 0.0                      # Start bias at 0
        self.cost_history = []               # Reset history

        # 2. Gradient Descent Loop
        for i in range(self.iterations):
            # A. Make Predictions (Forward Pass)
            # Math: y_pred = X * w + b
            y_predicted = np.dot(X, self.weights) + self.bias

            # B. Calculate Gradients (The Calculus part)
            # dw = (1/m) * X.T * (y_pred - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # db = (1/m) * sum(y_pred - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # C. Update Parameters (The "Descent")
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # D. Track Progress (Calculate MSE)
            cost = np.mean((y_predicted - y)**2)
            self.cost_history.append(cost)

            # Optional: Print progress every 1000 iterations
            if i % 1000 == 0:
                print(f"Iteration {i}: Cost {cost}")

    def predict(self, X: NDArray) -> NDArray:
        """
        Predicts target values for the given input features.

        Args:
            X (NDArray): The feature matrix of shape (n_samples, n_features).

        Returns:
            NDArray: The predicted values.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Please call fit() first.")

        return np.dot(X, self.weights) + self.bias
