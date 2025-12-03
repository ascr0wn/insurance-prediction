import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class InsurancePreprocessor:
    """
    A class to handle data preprocessing and feature engineering for the insurance dataset.
    """

    def __init__(self):
        """
        Initializes the InsurancePreprocessor with the necessary column definitions and pipeline.
        """
        # We define the columns here so the class knows what to look for
        self.numeric_features = ['age', 'bmi', 'children', 'bmi_smoker_interaction']
        self.categorical_features = ['sex', 'smoker', 'region']

        # Define the transformer pipeline
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_features)
            ])

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs feature engineering, specifically adding the interaction term between BMI and Smoker.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The dataframe with the new interaction feature.
        """
        df = df.copy()
        # Handle the smoker mapping safely
        # Assuming 'smoker' column exists and contains 'yes'/'no'
        if 'smoker' in df.columns:
            is_smoker = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
            df['bmi_smoker_interaction'] = df['bmi'] * is_smoker
        else:
             # Fallback if smoker column is already encoded or missing (though expected for this dataset)
             # If missing, we might need to raise error or handle it, but for now we assume input structure is consistent.
             # If user inputs data via app, we ensure 'smoker' is present.
             pass
        return df

    def fit_transform(self, X_train: pd.DataFrame) -> NDArray:
        """
        Fits the preprocessing pipeline to the training data and transforms it.

        Args:
            X_train (pd.DataFrame): The training features.

        Returns:
            NDArray: The processed feature matrix suitable for model training.
        """
        # 1. Add custom features
        X_eng = self.feature_engineering(X_train)
        # 2. Fit and transform using the pipeline
        return self.pipeline.fit_transform(X_eng)

    def transform(self, X_test: pd.DataFrame) -> NDArray:
        """
        Transforms new data using the already fitted pipeline.

        Args:
            X_test (pd.DataFrame): The test or new features.

        Returns:
            NDArray: The processed feature matrix.
        """
        # Used for test data or new predictions (DO NOT REFIT SCALERS)
        X_eng = self.feature_engineering(X_test)
        return self.pipeline.transform(X_eng)
