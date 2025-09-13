# src/data/make_dataset.py
import sqlite3
import pandas as pd
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[2]   # project root
DB   = ROOT / "churn_data.db"
OUT  = ROOT / "data_exports"
OUT.mkdir(exist_ok=True)

def die(msg: str):
    raise SystemExit(f"[ERROR] {msg}")

# ---------- Preconditions ----------
if not DB.exists():
    die(f"Database not found at {DB}. Run: python src/data/load_to_sqlite.py")

conn = sqlite3.connect(DB)
cur  = conn.cursor()

# Make sure table exists and has the expected columns
required_cols = {
    "CreditScore","Geography","Gender","Age","Tenure","Balance",
    "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"
}

cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='customers';")
if cur.fetchone() is None:
    die("Table 'customers' not found. Did you load the CSV into SQLite?")

cur.execute("PRAGMA table_info(customers);")
have_cols = {row[1] for row in cur.fetchall()}
missing = required_cols - have_cols
if missing:
    die(f"Missing columns in 'customers': {sorted(missing)}")

print("[OK] customers table present with expected columns.")

# ---------- Helper ----------
def export(name: str, sql: str):
    """Run SQL -> save to CSV in data_exports/"""
    print(f"[RUN] {name}")
    df = pd.read_sql_query(sql, conn)
    out_csv = OUT / f"{name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv.relative_to(ROOT)} (rows={len(df)})")

# ---------- Create view with bands ----------
cur.execute("DROP VIEW IF EXISTS v_customers_banded;")
cur.execute("""
CREATE VIEW v_customers_banded AS
SELECT
  *,
  CASE 
    WHEN Age < 25 THEN '<25'
    WHEN Age < 35 THEN '25-34'
    WHEN Age < 45 THEN '35-44'
    WHEN Age < 55 THEN '45-54'
    WHEN Age < 65 THEN '55-64'
    ELSE '65+'
  END AS AgeBand,
  CASE
    WHEN Tenure < 2 THEN '0-1'
    WHEN Tenure < 5 THEN '2-4'
    WHEN Tenure < 8 THEN '5-7'
    ELSE '8-10'
  END AS TenureBand,
  CASE
    WHEN Balance = 0 THEN '0'
    WHEN Balance < 25000 THEN '<25k'
    WHEN Balance < 50000 THEN '25-50k'
    WHEN Balance < 100000 THEN '50-100k'
    WHEN Balance < 150000 THEN '100-150k'
    ELSE '150k+'
  END AS BalanceBand,
  CASE
    WHEN CreditScore < 600 THEN '<600'
    WHEN CreditScore < 650 THEN '600-649'
    WHEN CreditScore < 700 THEN '650-699'
    WHEN CreditScore < 750 THEN '700-749'
    ELSE '750+'
  END AS CreditScoreBand
FROM customers;
""")
conn.commit()
print("[OK] Created view v_customers_banded with Age/Tenure/Balance/CreditScore bands.")

# ---------- Overview ----------
export("overview", """
SELECT
  COUNT(*) AS customers,
  SUM(Exited) AS churned,
  ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM customers;
""")

# ---------- Single-dimension slices ----------
export("by_geography", """
SELECT Geography,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Geography
ORDER BY churn_rate_pct DESC, customers DESC;
""")

export("by_gender", """
SELECT Gender,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Gender
ORDER BY churn_rate_pct DESC;
""")

export("by_num_products", """
SELECT NumOfProducts,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY NumOfProducts
ORDER BY NumOfProducts;
""")

export("by_is_active_member", """
SELECT IsActiveMember,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY IsActiveMember
ORDER BY churn_rate_pct DESC;
""")

export("by_has_credit_card", """
SELECT HasCrCard,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY HasCrCard
ORDER BY churn_rate_pct DESC;
""")

# ---------- Band-based slices ----------
export("by_age_band", """
SELECT AgeBand,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM v_customers_banded
GROUP BY AgeBand
ORDER BY CASE AgeBand
  WHEN '<25' THEN 0 WHEN '25-34' THEN 1 WHEN '35-44' THEN 2
  WHEN '45-54' THEN 3 WHEN '55-64' THEN 4 ELSE 5 END;
""")

export("by_tenure_band", """
SELECT TenureBand,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM v_customers_banded
GROUP BY TenureBand
ORDER BY CASE TenureBand
  WHEN '0-1' THEN 0 WHEN '2-4' THEN 1 WHEN '5-7' THEN 2 ELSE 3 END;
""")

export("by_balance_band", """
SELECT BalanceBand,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM v_customers_banded
GROUP BY BalanceBand
ORDER BY CASE BalanceBand
  WHEN '0' THEN 0 WHEN '<25k' THEN 1 WHEN '25-50k' THEN 2
  WHEN '50-100k' THEN 3 WHEN '100-150k' THEN 4 ELSE 5 END;
""")

export("by_credit_score_band", """
SELECT CreditScoreBand,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM v_customers_banded
GROUP BY CreditScoreBand
ORDER BY CASE CreditScoreBand
  WHEN '<600' THEN 0 WHEN '600-649' THEN 1 WHEN '650-699' THEN 2
  WHEN '700-749' THEN 3 ELSE 4 END;
""")

# ---------- Interactions / matrices ----------
export("geo_by_gender", """
SELECT Geography, Gender,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Geography, Gender
ORDER BY churn_rate_pct DESC, customers DESC;
""")

export("ageband_by_tenureband", """
SELECT AgeBand, TenureBand,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM v_customers_banded
GROUP BY AgeBand, TenureBand
ORDER BY 1, 2;
""")

export("products_by_tenureband", """
SELECT NumOfProducts, TenureBand,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM v_customers_banded
GROUP BY NumOfProducts, TenureBand
ORDER BY NumOfProducts, TenureBand;
""")

export("geo_by_products", """
SELECT Geography, NumOfProducts,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       ROUND(100.0*SUM(Exited)/COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Geography, NumOfProducts
ORDER BY Geography, NumOfProducts;
""")

# ---------- Top segments ----------
export("top_segments_min100", """
WITH seg AS (
  SELECT Geography, Gender, NumOfProducts, TenureBand,
         COUNT(*) AS customers,
         SUM(Exited) AS churned,
         1.0*SUM(Exited)/COUNT(*) AS churn_rate
  FROM v_customers_banded
  GROUP BY Geography, Gender, NumOfProducts, TenureBand
)
SELECT Geography, Gender, NumOfProducts, TenureBand,
       customers, churned,
       ROUND(100.0*churn_rate, 2) AS churn_rate_pct
FROM seg
WHERE customers >= 100
ORDER BY churn_rate DESC, customers DESC
LIMIT 50;
""")

# ---------- Excel workbook with all CSVs ----------
xlsx_path = OUT / "churn_sql_summaries.xlsx"
print(f"[RUN] Writing Excel workbook -> {xlsx_path.relative_to(ROOT)}")

# Use xlsxwriter if available, else fallback to openpyxl
engine = "xlsxwriter"
try:
    import xlsxwriter  # noqa: F401
except Exception:
    engine = "openpyxl"

with pd.ExcelWriter(xlsx_path, engine=engine) as xw:
    for csv in sorted(OUT.glob("*.csv")):
        df = pd.read_csv(csv)
        sheet = csv.stem[:31]  # Excel sheet name limit
        df.to_excel(xw, index=False, sheet_name=sheet)

print("[OK] Excel workbook written.")
conn.close()
print("\n✅ Done. Check data_exports/ for CSVs + churn_sql_summaries.xlsx")
