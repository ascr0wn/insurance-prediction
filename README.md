# 🏥 Medical Insurance Cost Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Complete-green)

A machine learning project that predicts individual medical costs billed by health insurance. This project compares a **Custom Linear Regression implementation (built from scratch using Gradient Descent)** against standard Scikit-Learn libraries, featuring an interactive **Streamlit Dashboard** for visualization and real-time prediction.

## 🎯 Project Goals
1.  **Mathematical Deep Dive:** Implement Linear Regression and Gradient Descent algorithms using raw NumPy (no `sklearn` for the core logic).
2.  **Feature Engineering:** Analyze interaction effects (specifically `BMI * Smoker`) to drastically improve model accuracy.
3.  **Model Comparison:** Benchmark the custom implementation against industry-standard libraries.
4.  **Interactive Application:** Provide a user-friendly interface to explore data, train models, and predict costs.

---

## 🚀 How to Run

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
Launch the interactive dashboard:
```bash
streamlit run app.py
```
This will open the application in your browser at `http://localhost:8501`.

### 3. Run Unit Tests
To verify the custom regression logic:
```bash
pytest tests/
```

---

## 🧠 The "From Scratch" Implementation

To demonstrate a fundamental understanding of Machine Learning, the core regression logic is implemented in `src/regression.py` without high-level libraries.

**The Math Behind the Code:**

1.  **Hypothesis Function:**
    $$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$$

2.  **Cost Function (Mean Squared Error):**
    $$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

3.  **Gradient Descent Update Rule:**
    $$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

*Where $\alpha$ is the learning rate.*

---

## 📂 Project Structure

```text
├── src/
│   ├── regression.py      # Custom Linear Regression Class (Gradient Descent)
│   ├── preprocessing.py   # Feature Engineering & Pipeline
├── tests/                 # Unit tests for the custom regression model
├── data/                  # Dataset directory
├── notebooks/             # Exploratory Data Analysis (EDA) notebooks
├── app.py                 # Streamlit Dashboard application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 📊 Key Insights & EDA

During the Exploratory Data Analysis, we discovered a critical interaction:

* **Non-Smokers:** Increasing BMI has a negligible effect on medical costs.
* **Smokers:** Increasing BMI results in a massive, non-linear spike in costs.

**Feature Engineering:**
To capture this, I created a manual interaction feature:
```python
df['bmi_smoker_interaction'] = df['bmi'] * df['smoker_code']
```
This is handled automatically in `src/preprocessing.py`.
