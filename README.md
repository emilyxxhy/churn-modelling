# 🏦 Customer Churn Analysis (SQL · Python · Excel · ML)

This project analyzes customer churn using the **Kaggle Churn Modelling dataset** (~10k customers).  
It demonstrates **end-to-end data analytics skills**: data cleaning, SQL, exploratory data analysis, machine learning, and business reporting.

**Key Deliverables**
- Cleaned dataset (`data_intermediate/churn_clean.csv`)
- SQL summaries & aggregates (`data_exports/*.csv`, `churn_sql_summaries.xlsx`)
- Trained ML model (`models/best_model.joblib`) + metrics
- Excel pivots (`reports/churn_excel_views.xlsx`)
- Business readout (`reports/churn_readout.pdf`)

---

## 📂 Project Structure

churn-modelling/
├─ data_raw/ # raw Kaggle CSV
├─ data_intermediate/ # cleaned & banded datasets
├─ data_exports/ # SQL + model outputs for Excel/BI
├─ models/ # saved models + metrics.json
├─ notebooks/ # Jupyter notebooks for EDA
├─ reports/ # Excel views, PDF readout, charts
├─ src/
│ ├─ data/ # data loaders, SQLite, exports
│ ├─ models/ # training & evaluation scripts
│ └─ viz/ # optional chart exports
├─ requirements.txt # Python dependencies
├─ README.md # this file
└─ .gitignore


---

## ⚙️ Setup

### 1. Clone & create environment
```bash
git clone https://github.com/<your-username>/churn-modelling.git
cd churn-modelling
python3 -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

mkdir -p data_raw
kaggle datasets download -d shrutimechlearn/churn-modelling -p data_raw --unzip

# 0. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download raw data
kaggle datasets download -d shrutimechlearn/churn-modelling -p data_raw --unzip

# 2. Clean → saves data_intermediate/churn_clean.csv
jupyter notebook notebooks/01_quick_audit.ipynb   # or run your cleaner script

# 3. (Optional) Extra EDA → outputs churn_banded.csv
jupyter notebook notebooks/02_eda.ipynb

# 4. Load into SQLite
python src/data/load_to_sqlite.py

# 5. Export SQL aggregates for BI/Excel
python src/data/make_dataset.py

# 6. Train ML model (+ metrics & plots)
python src/models/train_model.py
python src/models/evaluate.py   # optional: feature importance

# 7. Build Excel pivots (manual) → reports/churn_excel_views.xlsx

# 8. (Optional) Tableau/Power BI dashboard → reports/churn_dashboard.pbix / .twb

# 9. Business Readout (Word → PDF)
reports/churn_readout.pdf

📊 Example Insights

Churn rate ≈ 20% overall.

Higher churn in Germany vs. France/Spain.

Customers with 2 products churn the most.

Younger, low-tenure customers at greater risk.

Random Forest model achieved ROC-AUC ≈ 0.84.

📝 Recommendations

Focus retention strategy on 2-product customers.

Target German market with tailored offers.

Engage younger customers through digital-first channels.

Aim to reduce churn by 3–5% in 6 months.

🔑 Skills Highlighted

Python (pandas, scikit-learn, matplotlib, seaborn)

SQL (SQLite, group-bys, aggregations)

Excel (PivotTables, heatmaps)

ML (Random Forest, Logistic Regression, GridSearchCV)

Business storytelling → Excel pivots + PDF readout