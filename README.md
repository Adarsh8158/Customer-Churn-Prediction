# Customer Churn Prediction & Analytics System

An end-to-end **Machine Learning and Business Analytics project** that analyzes customer behavior and predicts customer churn using the IBM Telco Customer Churn dataset.

The project covers the complete workflow from **data cleaning and exploratory analysis to feature engineering, machine learning, model evaluation, feature analysis, and business insights**.

---

## 📌 Project Overview

Customer churn is a major business problem for subscription-based companies. Identifying customers who are likely to leave can help businesses understand customer behavior and design better retention strategies.

This project uses customer demographic, service, contract, payment, tenure, and billing information to:

* Analyze customer churn patterns
* Identify important churn-related features
* Prepare data for machine learning
* Train multiple classification models
* Compare model performance
* Evaluate predictions using multiple metrics
* Generate business-oriented insights
* Save the trained model for future use

---

## 🎯 Objectives

The main objectives of this project are:

1. Clean and preprocess the customer dataset.
2. Perform exploratory data analysis to understand churn patterns.
3. Identify relationships between customer characteristics and churn.
4. Transform categorical variables into machine-learning-ready features.
5. Train multiple classification algorithms.
6. Compare model performance using appropriate evaluation metrics.
7. Analyze important features associated with model predictions.
8. Generate business-oriented insights from the analysis.
9. Save the trained machine learning model for future predictions.

---

## 🗂️ Dataset

The project uses the **IBM Telco Customer Churn dataset**.

### Dataset Information

| Property             |                                Value |
| -------------------- | -----------------------------------: |
| Total Records        |                                7,043 |
| Original Columns     |                                   21 |
| Target Variable      |                                Churn |
| Numerical Features   | Tenure, MonthlyCharges, TotalCharges |
| Categorical Features |      Customer and service attributes |

### Target Variable

The `Churn` column contains two classes:

* `Yes` → Customer churned
* `No` → Customer did not churn

The dataset contains an imbalanced target distribution, with approximately **26.54% churned customers** and **73.46% non-churned customers**.

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

### Development Environment

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

# 📂 Project Structure

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
└── README.md
```

---

# 📓 Notebook Description

## 01 — Exploratory Data Analysis

**File:** `01_churn_eda.ipynb`

This notebook focuses on understanding and cleaning the raw dataset.

### Main Tasks

* Load the raw dataset
* Inspect dataset dimensions
* Check data types
* Identify missing values
* Check duplicate records
* Analyze unique values
* Convert `TotalCharges` to numeric
* Handle missing `TotalCharges`
* Analyze churn distribution
* Analyze numerical variables
* Analyze categorical variables
* Create visualizations

### Important Findings

Some notable patterns observed in the dataset include:

* Month-to-month customers have a higher observed churn rate than customers on longer contracts.
* Customers with higher monthly charges show higher observed churn rates.
* Customers with shorter tenure show higher observed churn rates.
* Churn rates vary considerably across payment methods and internet service types.

These findings describe **associations in the dataset and should not be interpreted as causal relationships**.

---

# ⚙️ 02 — Feature Engineering

**File:** `02_churn_feature_engineering.ipynb`

This notebook prepares the data for machine learning.

### Steps

1. Load the cleaned dataset
2. Verify data quality
3. Remove `customerID`
4. Encode the target variable
5. Identify categorical and numerical features
6. Apply one-hot encoding
7. Separate features and target
8. Perform train-test split
9. Apply feature scaling
10. Perform final validation

### Train-Test Split

```text
Training Data: 5,634 records
Testing Data:   1,409 records
```

A stratified train-test split was used to preserve the target-class distribution.

---

# 🤖 03 — Machine Learning Modeling

**File:** `03_churn_modeling.ipynb`

Three classification models were trained and evaluated.

### Models

#### 1. Logistic Regression

Used as a baseline linear classification model.

#### 2. Decision Tree

Used to capture non-linear relationships between customer features and churn.

#### 3. Random Forest

An ensemble-based classification model using multiple decision trees.

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
| Logistic Regression |    80.5% |   0.843 |
| Decision Tree       |    79.4% |   0.827 |
| Random Forest       |    78.8% |   0.825 |

The Logistic Regression model achieved the highest accuracy and ROC-AUC among the three tested models on this evaluation split.

---

## Logistic Regression Classification Performance

The Logistic Regression model produced the following results:

| Class    | Precision | Recall | F1-Score |
| -------- | --------: | -----: | -------: |
| No Churn |      0.85 |   0.89 |     0.87 |
| Churn    |      0.65 |   0.56 |     0.60 |

### Overall

* Accuracy: **80.48%**
* ROC-AUC: **0.843**

The results also show that identifying the churn class is more challenging than identifying non-churn customers.

---

# 🔲 Confusion Matrix

For the Logistic Regression model:

```text
[[924 111]
 [164 210]]
```

Interpretation:

|            | Predicted No | Predicted Yes |
| ---------- | -----------: | ------------: |
| Actual No  |          924 |           111 |
| Actual Yes |          164 |           210 |

This provides a more detailed view of correct and incorrect predictions than accuracy alone.

---

# 🔍 Feature Analysis

Logistic Regression coefficients were analyzed to identify features with relatively large associations with the model's predictions.

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

Positive and negative coefficients indicate the direction of association within the fitted Logistic Regression model.

**Important:** Model coefficients indicate statistical associations within the model; they do not establish that a feature directly causes churn.

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
* Which payment methods show different churn patterns?
* Which service categories are associated with different churn rates?

---

# 📈 Key Business Findings

### Contract

Observed churn varies substantially by contract type. Month-to-month customers have a much higher churn rate than customers with one-year or two-year contracts.

### Tenure

Customers who churned had a lower average tenure than customers who did not churn.

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

Observed churn rates differ across internet service categories, with Fiber optic customers showing a higher churn rate than DSL and customers without internet service.

### Payment Method

Electronic check customers showed a substantially higher observed churn rate than the other payment-method categories.

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

These files can be used to reproduce the model's expected input structure in a future prediction application.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Customer-Churn-Prediction
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the Notebooks

Open the notebooks in the following order:

```text
01_churn_eda.ipynb
02_churn_feature_engineering.ipynb
03_churn_modeling.ipynb
04_churn_business_analysis.ipynb
```

---

# 📦 Requirements

The project dependencies are listed in `requirements.txt`.

Main libraries include:

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

### Data Cleaning

* Checked missing values
* Checked duplicate records
* Converted `TotalCharges` to numeric
* Handled missing values
* Verified data types

### Feature Engineering

* Removed customer identifier
* Converted target variable into binary format
* Applied one-hot encoding
* Split data into training and testing sets
* Applied feature scaling where appropriate

### Modeling

Three classification algorithms were tested:

```text
Logistic Regression
Decision Tree
Random Forest
```

### Evaluation

Model performance was evaluated using multiple metrics instead of relying only on accuracy.

---

# ⚠️ Limitations

* The dataset is a publicly available benchmark dataset and may not represent every real-world telecom customer population.
* Model performance depends on the selected train-test split and preprocessing approach.
* The project does not establish causal relationships between customer characteristics and churn.
* The current project does not include real-time prediction or production deployment.
* Additional model tuning and validation could further improve the modeling pipeline.

---

# 🔮 Future Improvements

Possible future extensions include:

* Hyperparameter tuning
* Cross-validation
* XGBoost or other advanced boosting models
* Probability-based churn scoring
* Explainable AI using SHAP
* Streamlit prediction application
* Customer-level risk segmentation
* Automated model pipeline
* Model monitoring
* Deployment through a cloud platform
* Interactive business dashboard

---

# 📌 Project Highlights

* End-to-end ML workflow
* 7,043 customer records
* Data cleaning and preprocessing
* Exploratory data analysis
* Feature engineering
* Three classification algorithms
* Multiple model evaluation metrics
* Confusion matrix analysis
* Feature coefficient analysis
* Business-oriented analysis
* Saved trained model
* Reproducible notebook workflow

---

## 👨‍💻 Author

**Adarsh Yadav**

BSc Data Science Student

---

## 📄 License

This project is licensed under the MIT License.

