import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# Setup path to import from src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.regression import MyLinearRegression
from src.preprocessing import InsurancePreprocessor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL STYLES ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #306998;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4B8BBE;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    data_path = os.path.join('data', 'raw', 'insurance.csv')
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}")
        return None
    return pd.read_csv(data_path)

df = load_data()

# --- PREPROCESSING & SPLITTING ---
if df is not None:
    X = df.drop(columns=['charges'])
    y = df['charges'].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Preprocessor (Fit on Train)
    preprocessor = InsurancePreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
else:
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis (EDA)", "Model Training", "Predict Cost", "Benchmark"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** Medical Insurance Cost Prediction\n\n"
    "**Goal:** Predict insurance bills using custom Gradient Descent logic."
)

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "Home":
    st.markdown('<div class="main-header">🏥 Medical Insurance Cost Prediction</div>', unsafe_allow_html=True)

    st.markdown("### 📌 Project Overview")
    st.write("""
    This project aims to predict individual medical costs billed by health insurance.
    It features a **Custom Linear Regression** implementation built from scratch using NumPy to demonstrate
    the core mathematical concepts of Machine Learning, specifically **Gradient Descent**.
    """)

    st.divider()

    st.markdown("### 🧮 The Mathematics")
    st.write("The core model is based on the following principles:")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. Hypothesis Function:**")
        st.latex(r"h_\theta(x) = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n")

        st.markdown("**2. Cost Function (MSE):**")
        st.latex(r"J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2")

    with col2:
        st.markdown("**3. Gradient Descent Update Rule:**")
        st.latex(r"\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)")
        st.latex(r"\text{where } \alpha \text{ is the learning rate.}")

    st.divider()
    st.markdown("### 📂 Project Structure")
    st.code("""
    ├── src/
    │   ├── regression.py      # Custom Linear Regression Class
    │   ├── preprocessing.py   # Feature Engineering & Pipeline
    ├── notebooks/             # EDA and experimentation
    ├── tests/                 # Unit tests
    └── app.py                 # This Streamlit Dashboard
    """, language="text")

# ==========================================
# PAGE 2: EDA
# ==========================================
elif page == "Data Analysis (EDA)":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    st.write("Understanding the dataset is crucial before modeling. Here are the key insights.")

    if st.checkbox("Show Raw Data"):
        st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1. Charges Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['charges'], kde=True, ax=ax1, color='skyblue')
        ax1.set_title('Distribution of Medical Charges')
        st.pyplot(fig1)
        st.info("The charges are right-skewed, meaning most people pay lower amounts, but there are outliers with very high costs.")

    with col2:
        st.markdown("#### 2. Smoker vs Non-Smoker Costs")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='smoker', y='charges', data=df, ax=ax2, palette='Set2')
        ax2.set_title('Impact of Smoking on Charges')
        st.pyplot(fig2)
        st.info("Smokers have significantly higher medical costs compared to non-smokers.")

    st.markdown("#### 3. The Critical Interaction: BMI x Smoker")
    st.write("""
    This plot reveals why a linear model needs **Feature Engineering**.
    For non-smokers, BMI has little effect on cost. For smokers, higher BMI leads to drastically higher costs.
    """)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', alpha=0.7, ax=ax3)
    ax3.set_title('BMI vs Charges (Colored by Smoker)')
    st.pyplot(fig3)

# ==========================================
# PAGE 3: MODEL TRAINING
# ==========================================
elif page == "Model Training":
    st.markdown('<div class="main-header">⚙️ Custom Model Training</div>', unsafe_allow_html=True)
    st.write("Tune the hyperparameters of the Gradient Descent algorithm and see how it performs.")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Hyperparameters")
        lr = st.number_input("Learning Rate (alpha)", 0.0001, 1.0, 0.1, 0.01, format="%.4f")
        iters = st.slider("Iterations", 100, 5000, 1000, 100)

        train_btn = st.button("🚀 Train Model", type="primary")

    with col2:
        if train_btn:
            with st.spinner('Training Custom Linear Regression...'):
                model = MyLinearRegression(learning_rate=lr, iterations=iters)
                model.fit(X_train_processed, y_train)

                preds = model.predict(X_test_processed)
                r2 = r2_score(y_test, preds)
                mse = mean_squared_error(y_test, preds)

                st.success("Training Complete!")

                # Metrics
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("R2 Score", f"{r2:.4f}")
                m_col2.metric("Final Cost (MSE)", f"{model.cost_history[-1]:.2f}")
                m_col3.metric("Iterations", iters)

                # Plots
                tab1, tab2 = st.tabs(["Convergence Plot", "Actual vs Predicted"])

                with tab1:
                    fig_cost, ax_cost = plt.subplots()
                    ax_cost.plot(model.cost_history)
                    ax_cost.set_xlabel("Iterations")
                    ax_cost.set_ylabel("Cost (MSE)")
                    ax_cost.set_title("Gradient Descent Convergence")
                    st.pyplot(fig_cost)

                with tab2:
                    fig_pred, ax_pred = plt.subplots()
                    sns.scatterplot(x=y_test, y=preds, ax=ax_pred, alpha=0.6)
                    ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax_pred.set_xlabel("Actual Charges")
                    ax_pred.set_ylabel("Predicted Charges")
                    st.pyplot(fig_pred)
        else:
            st.info("👈 Adjust parameters and click 'Train Model' to start.")

# ==========================================
# PAGE 4: PREDICT COST
# ==========================================
elif page == "Predict Cost":
    st.markdown('<div class="main-header">💰 Predict Your Insurance Cost</div>', unsafe_allow_html=True)

    # Needs a trained model to work best, but we'll retrain a default one for this view
    # or we could save/load a model. For simplicity, we retrain on the fly (it's fast).

    st.write("Enter patient details below to get an estimated insurance cost.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 18, 100, 30)
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
            children = st.number_input("Children", 0, 10, 0)

        with col2:
            sex = st.selectbox("Sex", ["male", "female"])
            smoker = st.selectbox("Smoker", ["yes", "no"])
            region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

        submitted = st.form_submit_button("Predict")

        if submitted:
            # 1. Create Dataframe for input
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'bmi': [bmi],
                'children': [children],
                'smoker': [smoker],
                'region': [region]
            })

            # 2. Preprocess
            # Note: We use the already fitted preprocessor from global scope
            input_processed = preprocessor.transform(input_data)

            # 3. Train Model (Hidden) - Ideally load a saved model
            model = MyLinearRegression(learning_rate=0.1, iterations=2000)
            model.fit(X_train_processed, y_train)

            # 4. Predict
            prediction = model.predict(input_processed)[0]

            st.markdown("---")
            st.metric("Estimated Insurance Cost", f"${prediction:,.2f}")

            if smoker == 'yes':
                st.warning("💡 Note: Being a smoker significantly increases your estimated cost.")

# ==========================================
# PAGE 5: BENCHMARK
# ==========================================
elif page == "Benchmark":
    st.markdown('<div class="main-header">🏆 Model Benchmark</div>', unsafe_allow_html=True)

    st.write("Comparing our **Custom Implementation** against the industry-standard **Scikit-Learn** library.")

    if st.button("Run Benchmark"):
        with st.spinner("Running comparison..."):
            # 1. Custom Model
            custom_model = MyLinearRegression(learning_rate=0.1, iterations=2000)
            custom_model.fit(X_train_processed, y_train)
            custom_preds = custom_model.predict(X_test_processed)
            custom_r2 = r2_score(y_test, custom_preds)
            custom_mae = mean_absolute_error(y_test, custom_preds)

            # 2. Scikit-Learn Model
            # Note: Scikit-learn LinearRegression uses Normal Equation (OLS) or SVD, not Gradient Descent usually,
            # so it finds the exact analytical solution.
            sklearn_model = SklearnLinearRegression()
            sklearn_model.fit(X_train_processed, y_train)
            sklearn_preds = sklearn_model.predict(X_test_processed)
            sklearn_r2 = r2_score(y_test, sklearn_preds)
            sklearn_mae = mean_absolute_error(y_test, sklearn_preds)

            # 3. Display Results
            results = pd.DataFrame({
                'Metric': ['R2 Score', 'Mean Absolute Error'],
                'Custom (Gradient Descent)': [custom_r2, custom_mae],
                'Scikit-Learn (OLS)': [sklearn_r2, sklearn_mae]
            })

            st.table(results.style.format({
                'Custom (Gradient Descent)': "{:.4f}",
                'Scikit-Learn (OLS)': "{:.4f}"
            }))

            # Conclusion
            diff = abs(custom_r2 - sklearn_r2)
            if diff < 0.01:
                st.success(f"✅ Our custom implementation matches Scikit-Learn performance! (Difference: {diff:.5f})")
            else:
                st.warning(f"⚠️ There is a slight difference in performance. (Difference: {diff:.5f})")
                st.write("This might be due to the need for more iterations or hyperparameter tuning in Gradient Descent.")
