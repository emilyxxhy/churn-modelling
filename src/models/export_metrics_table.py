# src/models/export_metrics_table.py
import json, pathlib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
METRICS_JSON = ROOT / "models" / "metrics.json"
OUT_CSV  = ROOT / "data_exports" / "model_metrics.csv"
OUT_PNG  = ROOT / "reports" / "model_metrics.png"

# safety: ensure folders exist
OUT_CSV.parent.mkdir(exist_ok=True)
OUT_PNG.parent.mkdir(exist_ok=True)

if not METRICS_JSON.exists():
    raise SystemExit(f"[ERROR] Missing {METRICS_JSON}. Run training first.")

with open(METRICS_JSON) as f:
    m = json.load(f)

# pull the core numbers
model_name = m.get("model", "unknown")
cv_auc     = m.get("cv_roc_auc", None)
holdout    = m.get("holdout", {})
auc        = holdout.get("roc_auc", None)
acc        = holdout.get("accuracy", None)
prec       = holdout.get("precision", None)
rec        = holdout.get("recall", None)
f1         = holdout.get("f1", None)

# also grab weighted average (nice for appendix)
report = holdout.get("classification_report", {})
weighted = report.get("weighted avg", {})
w_prec = weighted.get("precision", prec)
w_rec  = weighted.get("recall", rec)
w_f1   = weighted.get("f1-score", f1)

rows = [
    ["Model", model_name],
    ["CV ROC-AUC", cv_auc],
    ["Holdout ROC-AUC", auc],
    ["Accuracy", acc],
    ["Precision (weighted)", w_prec],
    ["Recall (weighted)", w_rec],
    ["F1 (weighted)", w_f1],
]
df = pd.DataFrame(rows, columns=["Metric", "Value"])

# save CSV
df.to_csv(OUT_CSV, index=False)
print(f"[OK] Wrote {OUT_CSV.relative_to(ROOT)}")

# also render a clean table image for easy pasting into the doc/PDF
plt.figure(figsize=(6.5, 2.6))
plt.axis("off")
tbl = plt.table(
    cellText=[[r[0], f"{r[1]:.3f}" if isinstance(r[1], (int, float)) else r[1]] for r in rows],
    colLabels=["Metric", "Value"],
    loc="center",
    cellLoc="left",
)
tbl.scale(1, 1.35)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
plt.close()
print(f"[OK] Wrote {OUT_PNG.relative_to(ROOT)}")

print("\n✅ Done. Drop the PNG into your Appendix and/or open the CSV in Excel.")
