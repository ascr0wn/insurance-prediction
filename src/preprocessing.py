import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class InsurancePreprocessor:
    def __init__(self):
        # We define the columns here so the class knows what to look for
        self.numeric_features = ['age', 'bmi', 'children', 'bmi_smoker_interaction']
        self.categorical_features = ['sex', 'smoker', 'region']

        # Define the transformer pipeline
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_features)
            ])

    def feature_engineering(self, df):
        """Replicates the manual interaction term creation."""
        df = df.copy()
        # Handle the smoker mapping safely
        is_smoker = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
        df['bmi_smoker_interaction'] = df['bmi'] * is_smoker
        return df

    def fit_transform(self, X_train):
        # 1. Add custom features
        X_eng = self.feature_engineering(X_train)
        # 2. Fit and transform using the pipeline
        return self.pipeline.fit_transform(X_eng)

    def transform(self, X_test):
        # Used for test data or new predictions (DO NOT REFIT SCALERS)
        X_eng = self.feature_engineering(X_test)
        return self.pipeline.transform(X_eng)