
# End-to-End Customer Churn Prediction Pipeline

## 👋 About This Project
This repository represents my approach to building a production-ready machine learning pipeline. Rather than simply fitting a model in a Jupyter Notebook, I wanted to simulate a real-world workflow where data engineering, modular code, and business interpretability are just as important as the model's accuracy.

My goal was to answer a critical business question: **"Who is leaving the bank, and how can we stop them?"**

## 💡 Implementation Strategy
I designed this project to move beyond basic analysis and demonstrate a structured engineering approach. Here is how I implemented the solution:

### 1. Data Engineering with SQL
Instead of doing all data manipulation in Pandas, I utilized **SQLite** to simulate a data warehouse environment.
* **Logic:** I implemented feature engineering directly in SQL using Views (e.g., `v_customers_banded`). This creates reusable logic for binning continuous variables like Age, Tenure, and Balance into categorical "bands".
* **Benefit:** This keeps the raw data immutable and ensures that the definition of a "high-balance customer" is consistent across all reports and models.

### 2. Modular Machine Learning Pipeline
I moved the training logic out of notebooks and into modular scripts (`src/models/train_model.py`) to ensure reproducibility.
* **Preprocessing:** I used Scikit-Learn's `ColumnTransformer` and `Pipeline` to handle scaling and One-Hot Encoding automatically, preventing data leakage.
* **Model Selection:** I implemented a `GridSearchCV` to rigorously compare a baseline Logistic Regression against a Random Forest Classifier. The Random Forest proved superior with an ROC-AUC of ~0.84.

### 3. Business-Centric Evaluation
Accuracy alone is often insufficient for business stakeholders.
* **Threshold Tuning:** I wrote a script to sweep through classification thresholds (0.1 to 0.9), allowing business users to trade off Precision vs. Recall based on their marketing budget.
* **Interpretability:** The pipeline automatically extracts feature importance, identifying that **Age, Number of Products, and Geography** are the primary drivers of churn.

---

## 📂 Repository Structure
* `src/data/make_dataset.py`: The ETL script. It creates the SQLite database, defines SQL views for age/tenure banding, and exports aggregated CSVs for the dashboard.
* `src/models/train_model.py`: The core training script. It handles the train/test split, cross-validation, and serializes the best model to `.joblib`.
* `src/models/evaluate.py`: Generates feature importance reports to explain *why* the model makes specific predictions.
* `notebooks/`: Contains the initial Exploratory Data Analysis (EDA) and prototyping.

---

### 📊 Deep Dive: Analytics & Insights

Beyond top-level metrics, the analysis uncovered specific "risk personas" within the customer base.

#### 1. The "German Market" Anomaly
While France and Spain maintain healthy retention rates (~16-20% churn), **Germany** is a critical outlier.
* **Female customers in Germany** have a churn rate of **37.6%**, nearly double the average.
* **Male customers in Germany** follow closely with a **27.8%** churn rate.
* *Hypothesis:* This suggests a systemic issue with the product offering or competitive landscape specifically in the DACH region, rather than a general service failure.

#### 2. The "One-Product" Trap
The number of products a customer holds is a massive indicator of loyalty.
* Customers with **only 1 product** are highly volatile. For example, German Female customers with 1 product have a churn rate exceeding **50%**.
* **Cross-sell Protection:** Customers who hold **2 products** are incredibly stable. Even in the high-risk German demographic, moving a user from 1 to 2 products drops their churn risk from ~50% to **~12-18%**.

#### 3. Feature Importance (Model Interpretability)
Using Random Forest feature importance, we identified that **Age** (28% importance) is the single strongest predictor of churn, followed by **Number of Products** (16.8%) and **Balance** (12.1%). Interestingly, **Estimated Salary** (11.1%) is less predictive than behavioral metrics, indicating that churn is driven more by engagement than by wealth.

---

### 🧠 Strategic Recommendations

Based on the model's precision of **81%** (meaning when it predicts churn, it is correct 4 out of 5 times), we recommend the following targeted interventions:

1.  **Operation "Cross-Sell":**
    * **Insight:** The drop in churn between 1-product and 2-product holders is drastic.
    * **Action:** Launch an automated marketing campaign targeting "1-product" users with high balances. Offer fee waivers or bonuses for opening a second account (e.g., Savings or Credit Card).

2.  **The "First-Year" Bundle:**
    * **Insight:** New customers (Tenure 0-1 years) in Germany churn at **48-50%**.
    * **Action:** Implement a high-touch onboarding program for German clients during their first 12 months to stabilize the relationship early.

3.  **Cost-Effective Retention:**
    * **Insight:** The model has low Recall (~44%) but high Precision (~81%).
    * **Action:** Use this model for **high-value interventions**. Since false positives are low, we can afford to offer expensive retention incentives (e.g., cash bonuses) to the predicted churners without wasting budget on loyal customers.

---

## 🚀 Getting Started

### 1. Setup Environment
```bash
git clone [https://github.com/emilyxxhy/churn-modelling.git](https://github.com/emilyxxhy/churn-modelling.git)
cd churn-modelling
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### 2\. Run the Pipeline

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

-----

*Author: Emily Huynh*

```
```
