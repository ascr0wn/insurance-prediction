import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Import your custom modules
# (You might need to adjust path depending on where you run this from)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.regression import MyLinearRegression
from src.preprocessing import InsurancePreprocessor


def main():
    # 1. Load Data
    print("Loading data...")
    # Using relative path assuming you run from project root
    data_path = os.path.join('data', 'raw', 'insurance.csv')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    insurance_dataset = pd.read_csv(data_path)

    # 2. Split Data
    X = insurance_dataset.drop(columns=['charges'])
    y = insurance_dataset['charges'].values  # Convert to numpy array immediately

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Preprocessing
    print("Preprocessing data...")
    preprocessor = InsurancePreprocessor()

    # Fit on Train, Transform on Test
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    # 4. Train Model
    print("Training Custom Linear Regression...")
    model = MyLinearRegression(learning_rate=0.1, iterations=2000)
    model.fit(X_train_processed, y_train)

    # 5. Evaluate
    print("Evaluating...")
    predictions = model.predict(X_test_processed)
    score = r2_score(y_test, predictions)

    print("-" * 30)
    print(f"Final R2 Score: {score:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()
