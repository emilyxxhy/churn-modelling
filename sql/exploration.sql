-- Overall churn rate
SELECT ROUND(AVG(Exited)*100, 2) AS churn_pct FROM customers;

-- Churn by geography
SELECT Geography, ROUND(AVG(Exited)*100, 2) AS churn_pct
FROM customers
GROUP BY Geography
ORDER BY churn_pct DESC;

-- Tenure bands
SELECT 
  CASE 
    WHEN Tenure < 2 THEN '0–1'
    WHEN Tenure < 5 THEN '2–4'
    WHEN Tenure < 8 THEN '5–7'
    ELSE '8–10'
  END AS TenureBand,
  ROUND(AVG(Exited)*100,2) AS churn_pct,
  COUNT(*) AS n
FROM customers
GROUP BY 1
ORDER BY 1;
