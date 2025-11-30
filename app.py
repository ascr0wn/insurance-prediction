import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# 1. Setup path so we can import from src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.regression import MyLinearRegression
from src.preprocessing import InsurancePreprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Insurance Cost Predictor", layout="wide")

st.title("🏥 Medical Insurance Cost Prediction")
st.markdown("This dashboard allows you to tune hyperparameters for the **Gradient Descent** algorithm in real-time.")

# --- SIDEBAR: HYPERPARAMETERS ---
st.sidebar.header("Model Hyperparameters")

# These widgets return the values you select
learning_rate = st.sidebar.number_input(
    "Learning Rate (alpha)",
    min_value=0.0001,
    max_value=1.0,
    value=0.1,
    step=0.01,
    format="%.4f"
)

iterations = st.sidebar.slider(
    "Iterations",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)


# --- LOAD AND PREPROCESS DATA ---
@st.cache_data  # This makes the app fast by not reloading data every time you click a button
def load_data():
    data_path = os.path.join('data', 'raw', 'insurance.csv')
    return pd.read_csv(data_path)


df = load_data()

# Show raw data if checkbox is clicked
if st.checkbox("Show Raw Data Snippet"):
    st.write(df.head())

# Prepare Data (Same logic as main.py)
X = df.drop(columns=['charges'])
y = df['charges'].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = InsurancePreprocessor()
X_train_processed = preprocessor.fit_transform(X_train_raw)
X_test_processed = preprocessor.transform(X_test_raw)

# --- TRAIN MODEL BUTTON ---
if st.button("Train Model"):
    with st.spinner('Training...'):
        # Initialize model with Sidebar values
        model = MyLinearRegression(learning_rate=learning_rate, iterations=iterations)

        # Train
        model.fit(X_train_processed, y_train)

        # Predict
        predictions = model.predict(X_test_processed)
        r2 = r2_score(y_test, predictions)

    # --- DISPLAY METRICS ---
    col1, col2 = st.columns(2)
    col1.metric("R2 Score", f"{r2:.4f}")
    col1.success(f"Training Complete! Final Cost: {model.cost_history[-1]:.2f}")

    # --- DISPLAY GRAPHS ---
    st.subheader("Visualizations")

    # Create two columns for graphs
    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        st.write("#### Cost History (Convergence)")
        fig1, ax1 = plt.subplots()
        ax1.plot(model.cost_history)
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Cost (MSE)")
        ax1.set_title("Gradient Descent Progress")
        st.pyplot(fig1)

    with graph_col2:
        st.write("#### Predictions vs Actual")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=y_test, y=predictions, ax=ax2, alpha=0.6)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Perfect fit line
        ax2.set_xlabel("Actual Charges")
        ax2.set_ylabel("Predicted Charges")
        st.pyplot(fig2)