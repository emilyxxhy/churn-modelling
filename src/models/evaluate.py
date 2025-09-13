# src/models/evaluate.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "best_model.joblib"
OUT   = ROOT / "data_exports"
OUT.mkdir(exist_ok=True)

if not MODEL.exists():
    raise SystemExit(f"[ERROR] {MODEL} not found. Train first.")

pipe = joblib.load(MODEL)

# Recover feature names after preprocessing
num_cols = pipe.named_steps["pre"].transformers_[0][2]  # numeric list
ohe      = pipe.named_steps["pre"].transformers_[1][1]  # the OneHotEncoder
cat_cols = ohe.get_feature_names_out(pipe.named_steps["pre"].transformers_[1][2])

feature_names = np.r_[num_cols, cat_cols]

clf = pipe.named_steps["clf"]

def export_df(df, name):
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"[OK] wrote {path.relative_to(ROOT)}  (rows={len(df)})")

# RandomForest: use feature_importances_
if hasattr(clf, "feature_importances_"):
    imps = pd.DataFrame({
        "feature": feature_names,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)
    export_df(imps.head(50), "feature_importance.csv")

# LogisticRegression: use absolute coefficients as a proxy
elif hasattr(clf, "coef_"):
    coefs = np.abs(clf.coef_).ravel()
    imps = pd.DataFrame({
        "feature": feature_names,
        "importance_abs": coefs
    }).sort_values("importance_abs", ascending=False)
    export_df(imps.head(50), "feature_importance_logreg.csv")

else:
    print("[INFO] This estimator doesn’t expose importances/coefficients.")
