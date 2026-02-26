# 🏠 Pune Property Price Predictor & Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Status](https://img.shields.io/badge/Project-Production%20Ready-brightgreen)

🔗 **Live Application:**
[https://puneproperty.streamlit.app/](https://puneproperty.streamlit.app/)

---

# 📌 Project Overview

**Pune Property Intelligence** is an end-to-end Machine Learning system designed to predict residential property prices in Pune. Beyond simple prediction, it provides budget-based recommendations through an interactive analytics dashboard, transforming raw data into a decision-support tool

> #### Project demonstrates:

* Real-world messy data cleaning
* Domain-aware feature engineering
* Multiple model experimentation
* Hyperparameter tuning with cross-validation
* Production-ready ML pipelines
* MAE-based price range estimation
* Deployment using Streamlit

---

# 🗂️ Project Workflow

```
Raw Dataset
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
EDA & Market Insights
     ↓
Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Final Model Selection
     ↓
Production Pipeline Export
     ↓
Streamlit Deployment
```

---

# 📊 Dataset Overview

The system is trained on a comprehensive dataset of **37,500+ residential property listings** in Pune, featuring **24 distinct attributes** including:

* **Property Specs:** Total Area (sqft), BHK/Room count, Bathrooms, and Balconies.
* **Amenities:** Furnishing status, Car Parking, Power Backup, and additional utility rooms.
* **Market Data:** Locality, Price per Sq. Ft., and Total Property Price.



---


# 🧹 Feature Engineering Highlights



> ### 🧩 Engineering & Pipeline Architecture

The core of this project is a robust, deployment-ready preprocessing architecture designed to ensure data integrity and prevent leakage.

* **Feature Selection:** Curated high-impact features (Locality, Area, Bathrooms, Balconies, Furnishing) while stripping redundant and leakage-prone variables.
* **Imputation Strategy:** Utilized `SimpleImputer` with **median** strategies for numerical data and **most-frequent** for categorical features to handle missingness.
* **Preprocessing & Scaling:** * **Categorical:** Applied `OrdinalEncoder` for structured feature representation.
* **Numerical:** Integrated `MinMaxScaler` for feature normalization.


* **Modular Pipelines:** Leveraged Scikit-learn’s `ColumnTransformer` to create isolated, reproducible workflows for numerical and categorical data.
* **Targeted Modeling:** Developed bifurcated production pipelines specifically optimized for **Furnished** vs. **Unfurnished** property segments.

> **Model Optimization:**
*  **Benchmarking:** Evaluated multiple regressors using **Mean Absolute Error (MAE)** and **$R^2$ Score**.
* **Fine-tuning:** Executed `RandomizedSearchCV` for rigorous hyperparameter optimization.


* **Deployment Serialization:** Exported finalized end-to-end pipelines via `joblib` to eliminate training-serving skew.

---

# 🤖 Machine Learning Modeling

### 🤖 Machine Learning & Model Architecture

This section outlines the strategic approach to model selection, target optimization, and performance benchmarking for the Pune real estate market.

### 🎯 Target Variable Optimization

To address the inherent right-skewness of property pricing data, the models were trained on the **Log-Transformed Price**:

```Target=log(Price)```

**Rationale:** Log transformation stabilizes variance (homoscedasticity) and minimizes the impact of high-value outliers, leading to more robust gradient convergence.

---

### 📊 Baseline Model Benchmarking (Pre-Tuning)

We evaluated four distinct regressor architectures to establish a performance baseline. **Tree-based ensembles** significantly outperformed linear benchmarks.

> Model Performance Comparison

### 📈 Unfurnished Property Dataset

| Model             | MAE (Log) | R² Score |
| ----------------- | --------- | -------- |
| Random Forest     | 0.2539    | 0.8211   |
| XGBoost           | 0.2678    | 0.8245   |
| Gradient Boosting | 0.2850    | 0.8056   |
| Linear Regression | 0.3660    | 0.6819   |

**Observation:**
Tree-based ensemble models significantly outperformed Linear Regression, indicating strong non-linear relationships between features and property prices.


### 📈 Furnished Property Dataset

| Model             | MAE (Log) | R² Score |
| ----------------- | --------- | -------- |
| Random Forest     | 0.2549    | 0.8229   |
| XGBoost           | 0.2648    | 0.8267   |
| Gradient Boosting | 0.2846    | 0.8062   |
| Linear Regression | 0.3633    | 0.6852   |

**Observation:**
Again, ensemble models performed substantially better. XGBoost achieved the highest R² score, while Random Forest maintained slightly lower MAE, making both strong candidates for final deployment.



---

# 🔎 Model Selection Logic

From initial results:

* Linear Regression performed weakest
* Gradient Boosting was moderate
* Random Forest and XGBoost performed strongest

So only:

* ✅ Random Forest
* ✅ XGBoost

were selected for **hyperparameter tuning** using RandomizedSearchCV (5-Fold CV).

This reduced unnecessary complexity and focused on high-performing models.

---

# ⚙️ Model Optimization & Hyperparameter Tuning

### 🛠️ Tuning Strategy

**`RandomizedSearchCV`** was utilized to efficiently navigate the high-dimensional parameter space, ensuring a balance between computational efficiency and model accuracy.

* **Validation Framework:** 5-Fold Cross-Validation (CV) to ensure consistent performance across unseen data subsets.
* **Search Intensity:** 20 iterations per model segment.
* **Optimization Metric:** **Negative Mean Absolute Error (MAE)**, selected to directly minimize prediction error in real-world currency terms.

### 🧬 Parameters Optimized

The following hyperparameters were tuned to control model complexity and prevent overfitting:

| Feature | Description |
| --- | --- |
| **`n_estimators`** | Determining the optimal number of decision trees in the ensemble. |
| **`max_depth`** | Controlling the maximum depth of trees to manage variance. |
| **`min_samples_split`** | Setting the minimum number of samples required to split an internal node. |
| **`learning_rate`** | (XGBoost specific) Scaling the contribution of each tree to prevent overshooting. |
| **`subsample`** | Fraction of samples used for fitting individual base learners. |
| **`colsample_bytree`** | Subsample ratio of columns when constructing each tree. |

---



# 🏆 Final Model Performance (After Tuning)
  Post-tuning, **`Random Forest`** consistently demonstrated the most stable cross-validation scores and the lowest generalized MAE. 

 This architecture was selected for the final **Production Pipeline**, providing the most reliable balance between bias and variance for the Pune housing market.

---

### 🏆 Final Model Evaluation: Tuned Random Forest

Following extensive hyperparameter optimization, the **Tuned Random Forest** was selected as the final production model. It achieved superior generalization across both property segments, explaining approximately **83% of the variance** in Pune real estate pricing.

---

### 📊 Performance Metrics of RandomForest (Production-Ready)

The following metrics represent the model's performance on the unseen test set, demonstrating high precision for real-world application:

| Segment | Metric | Value | Interpretation |
| --- | --- | --- | --- |
| **Unfurnished** | **MAE** | **0.2579** | High predictive accuracy with minimal log-variance. |
|  | **R² Score** | **0.8318** | Captures **83.18%** of price fluctuations. |
| **Furnished** | **MAE** | **0.2606** | Stable performance despite increased feature complexity. |
|  | **R² Score** | **0.8253** | Captures **82.53%** of price fluctuations. |



The marginal performance gap between the Furnished and Unfurnished models confirms a highly stable feature importance hierarchy

---


# 🏗️ Production ML Pipelines

To ensure seamless integration between the development environment and the live application, the final models were serialized into comprehensive **Scikit-learn Pipelines**.

#### 📦 Exported Artifacts

| Pipeline File | Target Segment | Integrated Components |
| --- | --- | --- |
| `rf_unfurnished_final_production_model.pkl` | Unfurnished | Preprocessing + Encoding + Scaling + Tuned RF |
| `rf_furnished_full_production_pipeline.pkl` | Furnished | Preprocessing + Encoding + Scaling + Tuned RF |

#### 🚀 Key Advantages

* **Elimination of Training-Serving Skew:** Guarantees that live user input undergoes the exact same transformations as the training data.
* **Encapsulated Preprocessing:** Automates scaling and encoding within a single object call, reducing code complexity in `app.py`.
* **Deployment Readiness:** Optimized for high-speed inference on Streamlit Cloud using `joblib` for efficient serialization.



---

# 📉 MAE-Based Price Range Estimation


To provide realistic market expectations rather than overconfident point estimates, the system generates a **confidence interval** using the model's Mean Absolute Error (MAE).

#### **The Transformation Logic**

1. **Inference:** Predict property value in **Log-Space**.
2. **Error Bounds:** Apply **±MAE** offsets to the log-prediction.
3. **Reversion:** Exponentiate values back to **INR (Real Price)**.

**Final Output:**

* **Predicted Price:** The statistical mean estimate.
* **Lower/Upper Bound:** A realistic market range based on historical model variance.

---


# 🏠 Recommendation Engine

Beyond simple price estimation, the system functions as a **decision-support tool** by cross-referencing predictions with the live property database.

### **Filtering Logic**

Once a price is predicted, the engine scans the dataset using two primary constraints:

* **Price Proximity:** Matches properties within the **MAE-calculated price range**.
* **Area Tolerance:** Filters for units within **1500 sq. ft.** of the user's input.

### **Market Analytics Output**

The dashboard aggregates the filtered results to provide a comprehensive market snapshot:


* ***Total matching flats***
* ***Average price***
* ***Cheapest & most expensive***
* ***Top localities***
* ***Top 10 matching listings***

This turns prediction into a decision-support tool.

---

# 📊 Dashboard Features

Built using Streamlit + Plotly:

* ***KPI Cards***
* ***Area vs Price Trend***
* ***Top 15 Locality Distribution***
* ***Donut Chart (Top 7 Localities)***
* ***Filtered Property Table***

---

# 🛠️ Tech Stack

* **Language:** Python

* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Plotly, Joblib

* **Deployment:** Streamlit, Streamlit Cloud

---

# 🌐 Deployment

> #### Hosted on Streamlit Cloud.

🔗 **Live App:**
[https://puneproperty.streamlit.app/](https://puneproperty.streamlit.app/)

---


# 🚀 How to Run Locally

1. **Clone the repo:**
```bash
git clone https://github.com/Shravani-1325/Pune_Property_Predictions_and_Recommendation_System-.git

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```

3. **Launch the Dashboard:**
```bash
streamlit run app.py
```
---
## 👩‍💻 Author

***Shravani More***
