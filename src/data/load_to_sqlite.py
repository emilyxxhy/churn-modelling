import os, sqlite3, pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[2]
db = root / "churn_data.db"
out = root / "data_exports"
out.mkdir(exist_ok=True)

conn = sqlite3.connect(db)

q1 = """
SELECT Geography, Gender,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       1.0*SUM(Exited)/COUNT(*) AS churn_rate
FROM customers
GROUP BY Geography, Gender
ORDER BY churn_rate DESC;
"""
pd.read_sql_query(q1, conn).to_csv(out / "geo_gender.sql.csv", index=False)

q2 = """
SELECT CASE
         WHEN Tenure < 2 THEN '0–1'
         WHEN Tenure < 5 THEN '2–4'
         WHEN Tenure < 8 THEN '5–7'
         ELSE '8–10'
       END AS TenureBand,
       COUNT(*) AS customers,
       SUM(Exited) AS churned,
       1.0*SUM(Exited)/COUNT(*) AS churn_rate
FROM customers
GROUP BY TenureBand
ORDER BY TenureBand;
"""
pd.read_sql_query(q2, conn).to_csv(out / "tenure_bands.sql.csv", index=False)
conn.close()
print("Saved SQL exports → data_exports/geo_gender.sql.csv, tenure_bands.sql.csv")
