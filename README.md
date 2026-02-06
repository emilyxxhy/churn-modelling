
```markdown
# End-to-End Customer Churn Prediction Pipeline

## 👋 About This Project
This repository represents my approach to building a production-ready machine learning pipeline. Rather than simply fitting a model in a Jupyter Notebook, I wanted to simulate a real-world workflow where data engineering, modular code, and business interpretability are just as important as the model's accuracy.

My goal was to answer a critical business question: **"Who is leaving the bank, and how can we stop them?"**

---

## 🔑 Key Findings & Model Insights
By analyzing the Feature Importance extracted from the Random Forest model, we identified the top 3 drivers of customer churn. This allows the business to stop guessing and start targeting the right problems.

**1. Age is the #1 Predictor (Importance: ~28%)**
* **Finding:** Older customers are significantly more likely to churn than younger ones. This feature outweighs salary and credit score combined.
* **Action:** We need to investigate if our digital interface is alienating older users or if our longevity rewards are insufficient.

**2. Product Usage "Sweet Spot" (Importance: ~17%)**
* **Finding:** The number of products a customer uses is a critical stability indicator. Customers with **2 products** are the most stable, while those with 1 product are high-risk.
* **Action:** A tailored "Cross-Sell" campaign to move 1-product users to 2 products could drastically reduce churn.

**3. Account Balance Matters (Importance: ~12%)**
* **Finding:** High account balances don't guarantee loyalty. In fact, balance is a stronger predictor of churn than Estimated Salary (~11%).
* **Action:** High-net-worth individuals are leaving; we need a "VIP Retention" program immediately.

---

## 🏗 Pipeline Architecture
I designed this project to move beyond basic analysis and demonstrate a structured engineering approach.

```text
[ Raw CSV Data ]
       |
       v
[ 1. SQL Ingestion (SQLite) ]  <-- Simulates a Data Warehouse
       |                           (Cleaned data, created Age/Tenure bands)
       v
[ 2. Preprocessing Pipeline ]  <-- Scikit-Learn ColumnTransformer
       |                           (One-Hot Encoding, Scaling)
       v
[ 3. Model Training ]          <-- Modular Python Scripts
       |                           (GridSearch CV, Random Forest Classifier)
       v
[ 4. Evaluation & Reporting ]  <-- Automated Insight Generation
       |----> Metrics (ROC-AUC: ~0.86)
       |----> Feature Importance CSV
       |----> Business Visualizations (PNGs)

```

### 1. Data Engineering with SQL

Instead of doing all data manipulation in Pandas, I utilized **SQLite** to simulate a data warehouse environment.

* **Logic:** I implemented feature engineering directly in SQL using Views (e.g., `v_customers_banded`). This creates reusable logic for binning continuous variables like Age, Tenure, and Balance into categorical "bands".
* **Benefit:** This keeps the raw data immutable and ensures that the definition of a "high-balance customer" is consistent across all reports and models.

### 2. Modular Machine Learning Pipeline

I moved the training logic out of notebooks and into modular scripts (`src/models/train_model.py`) to ensure reproducibility.

* **Preprocessing:** I used Scikit-Learn's `ColumnTransformer` and `Pipeline` to handle scaling and One-Hot Encoding automatically, preventing data leakage.
* **Model Selection:** I implemented a `GridSearchCV` to rigorously compare a baseline Logistic Regression against a Random Forest Classifier. The Random Forest proved superior with an ROC-AUC of **~0.86**.

### 3. Business-Centric Evaluation

Accuracy alone is often insufficient for business stakeholders.

* **Threshold Tuning:** I wrote a script to sweep through classification thresholds, allowing business users to trade off Precision vs. Recall based on their marketing budget.
* **Interpretability:** The pipeline automatically extracts feature importance, identifying that **Age, Number of Products, and Balance** are the primary drivers of churn.

---

## 📂 Repository Structure

* `src/data/make_dataset.py`: The ETL script. It creates the SQLite database, defines SQL views for age/tenure banding, and exports aggregated CSVs for the dashboard.
* `src/models/train_model.py`: The core training script. It handles the train/test split, cross-validation, and serializes the best model to `.joblib`.
* `src/models/evaluate.py`: Generates feature importance reports to explain *why* the model makes specific predictions.
* `reports/`: Contains generated charts and confusion matrices.
* `notebooks/`: Contains the initial Exploratory Data Analysis (EDA) and prototyping.

---

### 📊 Deep Dive: Risk Personas

Beyond top-level metrics, the analysis uncovered specific "risk personas" within the customer base.

#### 1. The "German Market" Anomaly

While France and Spain maintain healthy retention rates (~16-20% churn), **Germany** is a critical outlier.

* **Female customers in Germany** have a churn rate of **37.6%**, nearly double the average.
* **Male customers in Germany** follow closely with a **27.8%** churn rate.
* *Hypothesis:* This suggests a systemic issue with the product offering or competitive landscape specifically in the DACH region.

#### 2. The "One-Product" Trap

* Customers with **only 1 product** are highly volatile. For example, German Female customers with 1 product have a churn rate exceeding **50%**.
* **Cross-sell Protection:** Customers who hold **2 products** are incredibly stable. Even in the high-risk German demographic, moving a user from 1 to 2 products drops their churn risk from ~50% to **~12-18%**.

---

### 🧠 Strategic Recommendations

Based on the model's precision of **~86%** (weighted), we recommend the following targeted interventions:

1. **Operation "Cross-Sell":**
* **Action:** Launch an automated marketing campaign targeting "1-product" users with high balances. Offer fee waivers for opening a second account (e.g., Savings or Credit Card).


2. **The "First-Year" Bundle:**
* **Insight:** New customers (Tenure 0-1 years) in Germany churn at **48-50%**.
* **Action:** Implement a high-touch onboarding program for German clients during their first 12 months.


3. **Cost-Effective Retention:**
* **Action:** Use this model for **high-value interventions**. Since false positives are low, we can afford to offer expensive retention incentives (e.g., cash bonuses) to the predicted churners without wasting budget on loyal customers.



---

## 🚀 Getting Started

### 1. Setup Environment

```bash
git clone [https://github.com/emilyxxhy/churn-modelling.git](https://github.com/emilyxxhy/churn-modelling.git)
cd churn-modelling
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

```

### 2. Run the Pipeline

The entire workflow can be executed via command line:

```bash
# 1. Ingest data and run SQL transformations
python src/data/load_to_sqlite.py
python src/data/make_dataset.py

# 2. Train and evaluate the model
python src/models/train_model.py
python src/models/evaluate.py

# 3. Generate charts and reports
python src/viz/export_charts.py

```

## 🛠 Tools Used

* **Python 3.10+**: Pandas, Scikit-Learn, Matplotlib
* **SQL (SQLite)**: Data transformation and aggregation
* **Project Structure**: Modular scripts separated from notebooks for maintainability

---

*Author: Emily Huynh*

```

```
