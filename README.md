# Customer Churn Prediction & Analytics System

An end-to-end **Machine Learning and Business Analytics project** that analyzes customer behavior and predicts customer churn using the **IBM Telco Customer Churn dataset**.

The project covers data cleaning, exploratory data analysis, feature engineering, machine learning, model evaluation, feature analysis, and business-oriented insights.

---

## 📌 Project Overview

Customer churn is an important business problem for subscription-based companies. Understanding which customer groups are more associated with churn can help businesses analyze customer behavior and develop retention strategies.

This project uses customer demographic, service, contract, payment, tenure, and billing information to:

* Clean and preprocess customer data
* Analyze churn patterns through EDA
* Engineer machine-learning-ready features
* Train multiple classification models
* Compare model performance
* Evaluate predictions using multiple metrics
* Analyze important model features
* Generate business-oriented insights
* Save the trained model for future integration

---

## 🎯 Objectives

1. Clean and preprocess the customer dataset.
2. Perform exploratory data analysis.
3. Analyze relationships between customer characteristics and churn.
4. Convert categorical variables into machine-learning features.
5. Train multiple classification algorithms.
6. Compare model performance using appropriate metrics.
7. Analyze important features associated with model predictions.
8. Generate business-oriented insights.
9. Save the trained model and feature names for reproducibility.

---

## 🗂️ Dataset

This project uses the **IBM Telco Customer Churn dataset**.

### Dataset Information

| Property           | Value                                      |
| ------------------ | ------------------------------------------ |
| Total Records      | 7,043                                      |
| Original Columns   | 21                                         |
| Target Variable    | `Churn`                                    |
| Numerical Features | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Data Type          | Mixed numerical and categorical            |

### Target Variable

The `Churn` column contains two classes:

* `Yes` → Customer churned
* `No` → Customer did not churn

The dataset contains approximately:

* **26.54% churned customers**
* **73.46% non-churned customers**

---

## 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Statistical Analysis

* SciPy

### Model Serialization

* Joblib

### Development

* Jupyter Notebook
* VS Code

---

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train-Test Split
     ↓
Machine Learning
     ↓
Model Evaluation
     ↓
Feature Analysis
     ↓
Business Analysis
     ↓
Saved Model
```

---

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── telco_churn_cleaned.csv
│   └── churn_business_analysis.csv
│
├── model/
│   ├── logistic_regression_model.pkl
│   └── feature_names.pkl
│
├── 01_churn_eda.ipynb
├── 02_churn_feature_engineering.ipynb
├── 03_churn_modeling.ipynb
├── 04_churn_business_analysis.ipynb
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

# 📓 Notebook Details

## 01 — Exploratory Data Analysis

**File:** `01_churn_eda.ipynb`

This notebook focuses on understanding, cleaning, and exploring the raw dataset.

### Main Tasks

* Load the raw dataset
* Inspect dataset dimensions
* Check data types
* Check missing values
* Check duplicate records
* Analyze unique values
* Convert `TotalCharges` to numeric
* Handle missing `TotalCharges`
* Analyze churn distribution
* Analyze numerical variables
* Analyze categorical variables
* Create visualizations
* Save the cleaned dataset

### Key Observations

The analysis found several associations with churn:

* Month-to-month customers had a higher observed churn rate than customers on longer contracts.
* Customers with higher monthly charges showed higher observed churn rates.
* Customers with shorter tenure showed higher observed churn rates.
* Churn rates varied across payment methods and internet service types.

These are **observed associations in the dataset and do not establish causation**.

---

# ⚙️ 02 — Feature Engineering

**File:** `02_churn_feature_engineering.ipynb`

This notebook prepares the dataset for machine learning.

### Steps

1. Load the dataset
2. Verify data quality
3. Remove `customerID`
4. Encode the target variable
5. Identify categorical features
6. Apply one-hot encoding
7. Separate features and target
8. Perform stratified train-test split
9. Apply feature scaling
10. Perform final validation

### Train-Test Split

```text
Training Records: 5,634
Testing Records:  1,409
```

A stratified split was used to preserve the target-class distribution between training and testing data.

---

# 🤖 03 — Machine Learning Modeling

**File:** `03_churn_modeling.ipynb`

Three classification models were trained and evaluated.

### Models

#### Logistic Regression

Used as a baseline linear classification model.

#### Decision Tree

Used to model non-linear relationships between customer features and churn.

#### Random Forest

An ensemble classification model based on multiple decision trees.

---

# 📊 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

## Model Comparison

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------: | ------: |
| Logistic Regression |   80.48% |   0.843 |
| Decision Tree       |   79.42% |   0.827 |
| Random Forest       |   78.78% |   0.825 |

On this evaluation split, Logistic Regression produced the highest accuracy and ROC-AUC among the three tested models.

### Logistic Regression Classification Report

| Class    | Precision | Recall | F1-Score |
| -------- | --------: | -----: | -------: |
| No Churn |      0.85 |   0.89 |     0.87 |
| Churn    |      0.65 |   0.56 |     0.60 |

### Overall Performance

* **Accuracy:** 80.48%
* **ROC-AUC:** 0.843

The results show that predicting the churn class is more challenging than predicting non-churn customers.

---

# 🔲 Confusion Matrix

Logistic Regression produced the following confusion matrix:

```text
[[924 111]
 [164 210]]
```

| Actual / Predicted | No Churn | Churn |
| ------------------ | -------: | ----: |
| No Churn           |      924 |   111 |
| Churn              |      164 |   210 |

The confusion matrix provides more detail about correct and incorrect predictions than accuracy alone.

---

# 🔍 Feature Analysis

Logistic Regression coefficients were analyzed to understand which features had relatively stronger associations with the model's predictions.

### Top Features by Absolute Coefficient

| Feature                        | Coefficient |
| ------------------------------ | ----------: |
| Contract_Two year              |      -1.325 |
| InternetService_Fiber optic    |       0.743 |
| Contract_One year              |      -0.685 |
| OnlineSecurity_Yes             |      -0.437 |
| PhoneService_Yes               |      -0.431 |
| TechSupport_Yes                |      -0.389 |
| PaymentMethod_Electronic check |       0.388 |
| PaperlessBilling_Yes           |       0.376 |
| MultipleLines_Yes              |       0.275 |
| Dependents_Yes                 |      -0.220 |

Positive and negative coefficients represent the direction of association within the fitted Logistic Regression model.

> **Note:** Model coefficients represent associations learned by the model. They do not establish that a feature directly causes customer churn.

---

# 💼 04 — Business Analysis

**File:** `04_churn_business_analysis.ipynb`

The final notebook converts the analytical results into business-oriented observations.

### Areas Analyzed

* Overall churn rate
* Contract type
* Internet service
* Payment method
* Customer tenure
* Monthly charges
* Customer characteristics

### Business Questions

The analysis explores questions such as:

* Which customer groups have higher observed churn rates?
* How does contract type relate to churn?
* How does tenure relate to churn?
* How do monthly charges differ between churned and non-churned customers?
* How do payment methods differ in observed churn?
* How do service categories relate to observed churn?

---

# 📈 Key Business Findings

### Contract

Observed churn varies substantially by contract type. Month-to-month customers had a higher observed churn rate than customers with one-year or two-year contracts.

### Tenure

Customers who churned had lower average tenure than customers who did not churn.

```text
Average Tenure

No Churn  → 37.57 months
Churn     → 17.98 months
```

### Monthly Charges

Churned customers had higher average monthly charges in this dataset.

```text
Average Monthly Charges

No Churn  → 61.27
Churn     → 74.44
```

### Internet Service

Observed churn rates differed across internet service categories, with Fiber optic customers showing a higher observed churn rate than DSL and customers without internet service.

### Payment Method

Electronic check customers showed a substantially higher observed churn rate than the other payment-method categories.

> These findings describe patterns in the dataset and should not be interpreted as causal conclusions.

---

# 💾 Saved Machine Learning Model

The trained Logistic Regression model was saved using Joblib:

```text
model/logistic_regression_model.pkl
```

The feature names used during modeling were also saved:

```text
model/feature_names.pkl
```

These files support reproducibility and can be used as components of a future prediction application.

The current repository does **not** include a production-ready prediction API or deployment pipeline.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Adarsh8158/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the Notebooks

Open the notebooks in this order:

```text
01_churn_eda.ipynb
02_churn_feature_engineering.ipynb
03_churn_modeling.ipynb
04_churn_business_analysis.ipynb
```

---

# 📦 Requirements

The project dependencies are listed in `requirements.txt`.

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
joblib
```

---

# 🔬 Methodology

## Data Cleaning

* Checked missing values
* Checked duplicate records
* Converted `TotalCharges` to numeric
* Handled missing values
* Verified data types

## Feature Engineering

* Removed customer identifier
* Converted target variable into binary format
* Applied one-hot encoding
* Performed stratified train-test split
* Applied feature scaling in the modeling workflow

## Modeling

Three classification algorithms were evaluated:

```text
Logistic Regression
Decision Tree
Random Forest
```

## Evaluation

Model performance was evaluated using multiple metrics rather than relying only on accuracy.

---

# ⚠️ Limitations

* The dataset is a publicly available benchmark dataset and may not represent every real-world telecom customer population.
* Model performance depends on the selected train-test split and preprocessing approach.
* The project does not establish causal relationships between customer characteristics and churn.
* The current project does not include real-time prediction or production deployment.
* Additional hyperparameter tuning and cross-validation could further improve the modeling workflow.
* The saved model is not packaged with a complete production preprocessing pipeline.

---

# 🔮 Future Improvements

Possible future extensions include:

* Hyperparameter tuning
* Cross-validation
* XGBoost or other boosting models
* Probability-based churn scoring
* Explainable AI using SHAP
* Streamlit prediction application
* Customer-level risk segmentation
* Automated preprocessing and model pipeline
* Model monitoring
* Cloud deployment
* Interactive business dashboard

---

# 📌 Project Highlights

* End-to-end Machine Learning workflow
* 7,043 customer records
* Data cleaning and preprocessing
* Exploratory Data Analysis
* Feature engineering
* Three classification algorithms
* Multiple model evaluation metrics
* Confusion matrix analysis
* Logistic Regression coefficient analysis
* Business-oriented analysis
* Saved trained model
* Reproducible notebook workflow

---

# 👨‍💻 Author

**Adarsh Yadav**

BSc Data Science Student

---

# 📄 License

This project is licensed under the **MIT License**.
