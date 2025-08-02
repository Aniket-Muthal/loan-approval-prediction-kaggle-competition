# 🏦 Loan Approval Prediction using Machine Learning

A supervised learning project to predict loan approval decisions based on applicants financial and credit attributes using advanced classification models.

---
## 📖 Overview  
This project builds a robust machine learning pipeline to classify loan applications as approved or rejected. It includes:

- Data cleaning  
- Exploratory data analysis (EDA)  
- Feature engineering  
- Model training and evaluation  
- Final submission generation  

---
## ❓ Problem Statement  
Financial institutions must balance risk and profitability when approving loans. Approving high-risk applicants can lead to defaults, while rejecting low-risk ones may reduce revenue. This project aims to build a predictive model that accurately classifies loan applications to optimize approval decisions.

---
## 📂 Dataset Description

This project utilizes two complementary datasets sourced from Kaggle:

#### 1. Playground Series S4E10 Dataset
- **Files**: `train.csv`, `test.csv`, `sample_submission.csv`
- **Purpose**: Primary dataset for training and evaluation
- **Size**:
  - `train.csv`: **58,645** records
  - `test.csv`: **39,098** records

#### 2. Credit Risk Dataset
- **File**: `credit_risk_dataset.csv`
- **Purpose**: Supplementary training data to enrich feature diversity and improve model generalization
- **Size**: **32,581** records
- **Features**: Similar schema to the primary dataset, allowing seamless concatenation

| Feature Name                   | Type        | Description                                                                 |
|--------------------------------|-------------|-----------------------------------------------------------------------------|
| `id`                           | Identifier  | Unique ID for each record (dropped during modeling)                         |
| `person_age`                   | Numerical   | Age of the applicant                                                       |
| `person_income`               | Numerical   | Annual income of the applicant (in USD)                                    |
| `person_home_ownership`       | Categorical | Type of home ownership: `RENT`, `OWN`, `MORTGAGE`, `OTHER`                 |
| `person_emp_length`           | Numerical   | Employment length in years                                                 |
| `loan_intent`                 | Categorical | Purpose of the loan: `EDUCATION`, `MEDICAL`, `VENTURE`, etc.               |
| `loan_grade`                  | Ordinal     | Credit grade assigned to the loan: `A` (best) to `G` (worst)               |
| `loan_amnt`                   | Numerical   | Amount of loan requested                                                   |
| `loan_int_rate`               | Numerical   | Interest rate on the loan (percentage)                                     |
| `loan_percent_income`         | Numerical   | Ratio of loan amount to income                                             |
| `cb_person_default_on_file`   | Binary      | Previous default history: `Y` (Yes), `N` (No)                              |
| `cb_person_cred_hist_length`  | Numerical   | Length of credit history in years                                          |
| `loan_status`                 | Binary      | Target variable: `1` = Approved, `0` = Rejected                            |

### 🔍 Data Characteristics
- **Total Combined Records (after cleaning)**: **`90,970`**
- **Feature Types**:
  - Numerical: Age, income, employment length, loan amount, interest rate, credit history
  - Categorical: Home ownership, loan intent, loan grade, default history
- **Target Imbalance**:
  - Approved loans: **~17%**
  - Rejected loans: **~83%**

---
## 🛠️ Tools and Technologies

- **Languages**: Python
- **Libraries**:
  - Data Handling: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Modeling: `scikit-learn`, `xgboost`
  - Evaluation: `roc_auc_score`, `classification_report`, `confusion_matrix`
- **Environment**: Jupyter Notebook, Kaggle Kernel

---
## 🔍 Methods

- 🧹 **Data Cleaning**:
    - Removed unrealistic entries (e.g., age > 100, employment before age 14)
    - Imputed missing values for `loan_int_rate` and `person_emp_length`
    - Dropped `id` as it holds no predictive value
    - Mapped ordinal features (loan grade) to numeric scale
    - Ensured consistency across merged datasets

- 📊 **EDA**:
  - Univariate, bivariate, and multivariate analysis
  - Custom visualization functions

- ⚙️ **Preprocessing**:
  - OneHotEncoding for nominal features
  - StandardScaler for numerical features
  - Ordinal mapping for loan grades

- 🤖 **`Modeling`**:
  - Logistic Regression
  - Decision Tree
  - Random Forest (Tuned)
  - Gradient Boosting
  - **XGBoost** (Final Model)

---
##### Encoded Features Summary:
| Original Feature              | Encoding Type     | Notes                                                                 |
|------------------------------|-------------------|-----------------------------------------------------------------------|
| `person_home_ownership`      | One-Hot Encoding  | Converted into binary columns: `home_OWN`, `home_RENT`, etc.         |
| `loan_intent`                | One-Hot Encoding  | Encoded into multiple columns: `intent_EDUCATION`, `intent_MEDICAL`, etc. |
| `loan_grade`                 | Ordinal Encoding  | Mapped: `A` → 0, `B` → 1, ..., `G` → 6                                |
| `cb_person_default_on_file` | Binary Encoding   | Mapped: `Y` → 1, `N` → 0                                              |

##### Missing Values & Imputation Strategy:
| Feature Name                 | Missing % | Imputation Method                | Notes                                      |
|-----------------------------|-----------|----------------------------------|--------------------------------------------|
| `loan_int_rate`             | ~3.5%     | Median Imputation                | Preserves central tendency                 |
| `person_emp_length`         | ~2.1%     | Median Imputation                | Avoids skewing employment distribution     |
| `cb_person_default_on_file` | <1%       | Mode Imputation (`N`)            | Assumes conservative default status        |

## 📈 Key Insights

- 🏠 Renters have the highest loan approval rate (~25.5%)
- 💳 Debt consolidation loans are most likely to be approved (~22.5%)
- 🔍 Lower loan grades (F, G) surprisingly show higher approval rates
- ⚠️ 33% of applicants with past defaults still get approved
- ⚖️ Dataset is highly imbalanced (~17% approvals)

---
## 🚀 How to Run This Project

```bash
# Clone the repository
git clone https://github.com/Aniket-Muthal/loan-approval-prediction-kaggle-competition.git
cd loan-approval-prediction

# Install dependencies
pip install -r requirements.txt
```
Then, run the notebook

---
## ✅ Results

| Metric                     | Value     |
|----------------------------|-----------|
| Best Model                 | XGBoost   |
| Test Accuracy              | 94.8%     |
| ROC-AUC (Test)             | 95.4%     |
| F1 Score (Approved Class)  | 0.83      |

The XGBoost model demonstrates strong generalization and balances precision and recall effectively, especially for the minority class.

---
## 🧾 Conclusion
This project delivers a high-performing and interpretable machine learning solution for predicting loan approvals. Using a combination of public datasets and advanced modeling techniques, we built a pipeline that:

- Accurately predicts loan approval outcomes with **94.8% accuracy**
- Maintains strong class balance, with an **F1 score of 0.83** for approved loans
- Leverages key financial and behavioral indicators to drive predictions

It is designed to support financial institutions in improving approval efficiency, reducing risk, and enhancing customer experience.

---
## 🔮 Future Work
- 📊 Integrate SHAP for model interpretability
- 🌐 Deploy via Streamlit for real-time predictions
- 🧠 Explore ensemble stacking for further performance gains
- 📈 Incorporate additional features like credit utilization or debt-to-income ratio

---
## 👨‍💻 Author & Contact
**Aniket Muthal**

Let’s connect and advance strategic machine learning together 🚀

- 📧 Email: [aniketmuthal4@gmail.com]
- 🔗 LinkedIn: [https://www.linkedin.com/in/aniket-muthal]

---
