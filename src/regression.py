import numpy as np

class MyLinearRegression:
    """
    A custom Linear Regression model using Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the model using Gradient Descent.
        X: features (m_samples, n_features)
        y: target values (m_samples,)
        """
        # 1. Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) # Start weights at 0
        self.bias = 0                       # Start bias at 0
        self.cost_history = []              # Reset history

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

    def predict(self, X):
        """
        Predict future values.
        """
        return np.dot(X, self.weights) + self.bias