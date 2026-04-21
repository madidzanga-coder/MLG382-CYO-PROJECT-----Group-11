# 🧠 Stroke Prediction & Patient Segmentation Project

**CRISP-DM Based Data science Workflow**

Machine Learning

---

## 📌 Project Overview

This project applies machine learning techniques to a healthcare dataset to **predict stroke risk** and **segment patients into clusters**. The work follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework to ensure a structured, reproducible, and business-aligned approach.

Stroke is the second leading cause of death globally. This project builds a two-layer risk assessment system:
 
1. **Stroke Risk Classifier** – A Logistic Regression model predicts the probability of stroke from patient clinical and demographic features. Selected for its superior recall (0.82) on the imbalanced dataset.
2. **Health Risk Clusterer** – A K-Means model (k=4) segments patients into interpretable risk tiers (LOW / MODERATE / MODERATE-HIGH / HIGH) with tailored health recommendations.
   
The models are integrated into an interactive Dash web application where users enter a patient profile and receive instant predictions.


An end-to-end machine learning application that predicts individual stroke risk and segments patients into health risk tiers using classification and clustering models, deployed as an interactive Dash web application.
 
**Live App:** [Dash app](https://mlg382-cyo-project-group-11.onrender.com)

**Dataset:** [Kaggle – Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

---

## 🧭 CRISP-DM Framework

### 1. 🏥 Business Understanding

Stroke is a leading cause of death and disability worldwide. Early prediction can significantly improve intervention and patient outcomes.

**Objectives:**

* Build a classification model to predict stroke occurrence.
* Identify key risk factors influencing stroke.
* Segment patients into meaningful clusters for targeted healthcare strategies.

**Success Criteria:**

* High predictive performance (Accuracy, Precision, Recall, F1-score).
* Interpretable insights for healthcare decision-making.
* Deployable model for real-world use (via web app).

---

### 2. 📊 Data Understanding

**Dataset:** `healthcare-dataset-stroke-data.csv`

**Key Features:**

* Demographics: age, gender
* Health indicators: hypertension, heart disease, BMI
* Lifestyle: smoking status
* Target: `stroke` (0 = No, 1 = Yes)

**Activities:**

* Exploratory Data Analysis (EDA)
* Distribution analysis of features
* Correlation analysis
* Identification of missing values and outliers

📁 Notebook: `notebooks/EDA.ipynb`

---

### 3. 🧹 Data Preparation

**Steps Performed:**

* Handling missing values (e.g., BMI imputation)
* Encoding categorical variables
* Feature scaling (StandardScaler)
* Train-test split
* Feature selection

**Artifacts Generated:**

* Processed datasets (`data/processed/`)
* Scaler (`artifacts/scaler.pkl`)
* Feature columns (`artifacts/feature_columns.pkl`)

📁 Notebook: `notebooks/data_preprocessing.ipynb`

---

### 4. 🤖 Modeling

#### 🔹 Classification Models

Used to predict stroke occurrence:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest
* XGBoost

**Saved Models:**

* `model_lr.pkl`
* `model_knn.pkl`
* `model_xgb.pkl`

📁 Notebook: `notebooks/classification_modelling.ipynb`

#### 🔹 Clustering Model

Used for patient segmentation:

* K-Means Clustering

**Saved Model:**

* `model_km.pkl`

📁 Notebook: `notebooks/cluster_modelling.ipynb`

---
## More detail
### Classification (Stroke Prediction)
 
Three models were trained and evaluated on a held-out 20% test set. Due to severe class imbalance (~4.9% positive cases), **recall** was the primary selection metric.
 
| Model | Test Accuracy | Recall | F1 | Train-Test Gap | Selected |
|---|---|---|---|---|---|
| **Logistic Regression** | 72.6% | **0.816** | 0.228 | 0.057 | ✅ Yes |
| K-Nearest Neighbours | 87.7% | 0.132 | 0.096 | 0.123 | ❌ No |
| XGBoost | 94.0% | 0.158 | 0.207 | 0.060 | ❌ No |
 
Logistic Regression was selected for deployment because it correctly identifies 82% of actual stroke cases — the metric that matters most in a medical screening context.
 
### Clustering (Health Risk Tiers)
 
K-Means (k=4) was trained using the elbow method and silhouette analysis. Clusters are mapped to:
 
| Cluster | Risk Level | Priority |
|---|---|---|
| 0 | MODERATE-HIGH 🟠 | 2 |
| 1 | MODERATE 🟡 | 3 |
| 2 | LOW 🟢 | 4 |
| 3 | HIGH 🔴 | 1 |
 
### Top Predictive Features
 
1. **Age** – Strongest predictor (LR coef: 1.89, XGB importance: 0.240)
2. **Hypertension** – Direct comorbidity
3. **Heart Disease** – Known stroke risk factor
4. **Average Glucose Level** – Hyperglycemia indicator
5. **BMI** – Obesity-related risk

---

### 5. 📈 Evaluation

**Metrics Used:**

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

**Key Considerations:**

* Class imbalance in stroke prediction
* Trade-off between recall (catching strokes) vs precision
* Model generalization on unseen data

Results indicate that ensemble methods (e.g., XGBoost) provide strong predictive performance.

---

### 6. 🚀 Deployment

A simple web application is provided to demonstrate model predictions.

📁 Script: `src/web_app.py`

**Features:**

- **Patient Input Panel** – 7 input controls: gender, age (slider), average glucose, BMI, health conditions (checklist), work type, and smoking status.
- **Stroke Risk Analysis** – Predicted risk category (LOW / MODERATE / HIGH) with probability percentage and identified risk factors.
- **Health Risk Assessment** – Cluster-based risk tier with colour-coded severity and personalised health recommendations.

---

## 🗂️ Project Structure

```
MLG382-CYO-PROJECT/
│
├── data/
│   ├── healthcare-dataset-stroke-data.csv
│   └── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── data_preprocessing.ipynb
│   ├── classification_modelling.ipynb
│   └── cluster_modelling.ipynb
│
├── artifacts/
│   ├── model_lr.pkl
│   ├── model_knn.pkl
│   ├── model_xgb.pkl
│   ├── model_km.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── src/
│   └── web_app.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

1. Clone the repository:

```bash
git clone <repo-url>
cd MLG382-CYO-PROJECT
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the web app:

```bash
python src/web_app.py
```

---

## 💡 Key Insights

* Age, hypertension, and heart disease are strong predictors of stroke.
* Data preprocessing significantly improves model performance.
* Clustering reveals distinct patient groups for targeted intervention.


---

## 📄 License

This project is for academic purposes.

---

## 🧾 Executive Summary

This project demonstrates a complete machine learning pipeline using the CRISP-DM framework to predict stroke risk and segment patients.
Through careful data preparation and model selection, we developed reliable predictive models and actionable insights. 
The inclusion of a web application highlights the project's practical applicability in healthcare environments, supporting early diagnosis and improved patient outcomes.
