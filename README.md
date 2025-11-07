# 🏦 Customer Churn Analysis (SQL · Python · Excel · Machine Learning)

This end-to-end analytics project investigates **why customers leave a retail bank** and builds a **machine learning model to predict churn risk** using the [Kaggle Churn Modelling dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling).  
It combines **data cleaning, SQL analytics, machine learning, and business storytelling** into a single, reproducible workflow.

---

## 🚀 Project Overview
- **Goal:** Understand the key drivers of customer churn and develop an actionable prediction model.  
- **Dataset:** ~10,000 customer records with demographics, account activity, and product usage.  
- **Deliverables:**  
  - Cleaned dataset & SQL summaries  
  - Trained ML model (Random Forest / Logistic Regression)  
  - Excel pivots for visualization  
  - Business readout in PDF format  

---

## 📂 Project Structure
churn-modelling/
├─ data_raw/ # Original Kaggle CSV
├─ data_intermediate/ # Cleaned & banded datasets
├─ data_exports/ # SQL + model outputs for Excel/BI
├─ models/ # Saved models + metrics.json
├─ notebooks/ # Jupyter notebooks for EDA & testing
├─ reports/ # Excel views, PDF readout, charts
├─ src/
│ ├─ data/ # Data loaders, SQLite, exports
│ ├─ models/ # Training & evaluation scripts
│ └─ viz/ # Visualization exports
├─ requirements.txt # Python dependencies
└─ README.md

⚙️ Setup & Execution
1️⃣ Clone and Create Virtual Environment
git clone https://github.com/emilyxxhy/churn-modelling.git
cd churn-modelling
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Download Dataset
mkdir -p data_raw
kaggle datasets download -d shrutimechlearn/churn-modelling -p data_raw --unzip

3️⃣ Data Cleaning & Exploration

Run notebooks for initial audit and EDA:

jupyter notebook notebooks/01_quick_audit.ipynb
jupyter notebook notebooks/02_eda.ipynb

4️⃣ SQL Integration
python src/data/load_to_sqlite.py
python src/data/make_dataset.py

5️⃣ Train & Evaluate ML Model
python src/models/train_model.py
python src/models/evaluate.py

6️⃣ Reporting

Excel pivots → reports/churn_excel_views.xlsx

Business summary → reports/churn_readout.pdf

(Optional) Tableau / Power BI dashboard → reports/churn_dashboard.twbx

📊 Key Insights
Insight	Observation
Overall churn rate	~20%
Country differences	Germany shows the highest churn
Product count effect	Customers with 2 products churn the most
Demographic risk	Younger, low-tenure customers more likely to leave
Model performance	Random Forest ROC-AUC ≈ 0.84
🧠 Business Recommendations

Prioritize retention for 2-product customers.

Focus Germany with localized offers and engagement programs.

Build digital-first campaigns for younger customers.

Target a 3–5% churn reduction over the next 6 months.

🧰 Tools & Technologies
Category	Tools
Languages	Python, SQL
Libraries	pandas, scikit-learn, matplotlib, seaborn, joblib
Storage / Query	SQLite
Visualization	Excel, Tableau / Power BI
Workflow	Jupyter Notebook, command-line Python scripts
🧩 Highlights & Skills Demonstrated

Data Cleaning: handling missing values, encoding, scaling

SQL Analytics: joins, group-bys, aggregations, exports

Feature Engineering: categorical encoding, binning, correlation analysis

Modeling: Logistic Regression, Random Forest, GridSearchCV

Evaluation: Accuracy, ROC-AUC, feature importance

Business Storytelling: translating model results into insights and recommendations

📈 Example Visuals

(Add screenshots from your reports or notebooks here — e.g., correlation heatmap, feature importances, churn by geography, etc.)
Example:


🪄 Future Improvements

Deploy model as an API using FastAPI or Streamlit

Add SHAP explainability for better interpretability

Automate SQL data ingestion and report generation

Integrate live dashboards with Power BI or Tableau Public

👩‍💻 Author

Emily Huynh
📊 Data Analyst | Business Analytics & Machine Learning Enthusiast
🔗 Portfolio
 · LinkedIn
 · GitHub

📝 License

This project is released under the MIT License.
Feel free to fork and adapt for learning or portfolio purposes — with attribution.
