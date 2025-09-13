# Churn Modelling — **Zero-Stress Quickstart**

This tiny kit is the _simplest possible_ version of the project. No fancy tools yet.  
You’ll do 3 things:
1) put the CSV in `data_raw/`,  
2) run **one** Python script,  
3) open the PNG charts and CSV outputs.

---

## 1) Setup (copy/paste)

**macOS / Linux**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If you’ve never used a virtual environment, don’t worry—just copy the two lines.

---

## 2) Get the data (no Kaggle CLI needed)

- Download the dataset in your browser: **Kaggle → “Churn Modelling” (by shrutimechlearn)**.
- Save the file as `Churn_Modelling.csv` into the `data_raw/` folder in this project.

> Tip: If the file is zipped, unzip it and move the CSV only.

---

## 3) Run one command

```bash
python src/quickstart.py
```

What you get:
- Cleaned CSV → `data_intermediate/churn_clean.csv`
- Simple aggregates → `data_exports/geo_gender.csv`, `data_exports/age_tenure.csv`
- Quick charts → PNGs in `reports/`
- A tiny model (Logistic Regression) + metrics printed in the terminal

---

## 4) Open the outputs

- PNG charts in `reports/`
- CSVs in `data_exports/` (open in Excel if you want)
- (Optional later) Load CSVs into Power BI for an interactive dashboard.

---

## Upgrades when ready (optional)
- Swap in SQLite and run the example SQL in `sql/exploration.sql`
- Try `src/load_to_sqlite.py` then `sql/exploration.sql`
- Replace the simple model with RandomForest and add feature importance

---

_Last updated: 2025-09-12_
