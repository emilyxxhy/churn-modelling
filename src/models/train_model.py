# src/models/train_model.py
import json, os, math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[2]
IN   = ROOT / "data_intermediate" / "churn_clean.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
EXPORTS = ROOT / "data_exports"
for p in [MODELS, REPORTS, EXPORTS]:
    p.mkdir(exist_ok=True)

if not IN.exists():
    raise SystemExit(f"[ERROR] Missing {IN}. Create it first (run quickstart/EDA).")

# ---------- load ----------
Xy = pd.read_csv(IN)

if "Exited" not in Xy.columns:
    raise SystemExit("[ERROR] 'Exited' column not found in dataset.")

y = Xy["Exited"].astype(int)
X = Xy.drop(columns=["Exited"])

# Choose columns that exist (robust to earlier steps)
num_cols_all = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
cat_cols_all = ['Geography','Gender','HasCrCard','IsActiveMember']

num_cols = [c for c in num_cols_all if c in X.columns]
cat_cols = [c for c in cat_cols_all if c in X.columns]
if not num_cols and not cat_cols:
    raise SystemExit("[ERROR] No usable features left.")

# ---------- preprocessing ----------
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

# ---------- models + small grids ----------
models = {
    "logreg": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"clf__C": [0.1, 1.0, 3.0]}
    ),
    "rf": (
        RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1
        ),
        {"clf__max_depth": [None, 8, 12]}
    ),
}

# ---------- CV model selection on full data ----------
best_model = None
best_auc = -1.0
best_name = None

for name, (clf, grid) in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True, verbose=0
    )
    gs.fit(X, y)
    cv_auc = gs.best_score_
    print(f"[CV] {name} best AUC: {cv_auc:.3f}  params: {gs.best_params_}")
    if cv_auc > best_auc:
        best_auc, best_model, best_name = cv_auc, gs.best_estimator_, name

print(f"[SELECT] Best by CV: {best_name} (AUC={best_auc:.3f})")

# ---------- holdout split + evaluation ----------
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
best_model.fit(Xtr, ytr)
proba = best_model.predict_proba(Xte)[:, 1]
pred  = (proba >= 0.50).astype(int)

auc = roc_auc_score(yte, proba)
acc = accuracy_score(yte, pred)
prec = precision_score(yte, pred, zero_division=0)
rec = recall_score(yte, pred, zero_division=0)
f1 = f1_score(yte, pred, zero_division=0)
report = classification_report(yte, pred, digits=3, output_dict=True)

print(f"[HOLDOUT] AUC={auc:.3f}  ACC={acc:.3f}  PREC={prec:.3f}  REC={rec:.3f}  F1={f1:.3f}")

# ---------- threshold sweep (so you can choose business tradeoff) ----------
rows = []
for t in np.linspace(0.1, 0.9, 17):
    p = (proba >= t).astype(int)
    rows.append({
        "threshold": round(float(t), 3),
        "precision": precision_score(yte, p, zero_division=0),
        "recall": recall_score(yte, p, zero_division=0),
        "f1": f1_score(yte, p, zero_division=0),
        "accuracy": accuracy_score(yte, p),
    })
thresh_df = pd.DataFrame(rows)
thresh_df.to_csv(EXPORTS / "threshold_metrics.csv", index=False)

# ---------- plots ----------
# ROC
fpr, tpr, _ = roc_curve(yte, proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
plt.legend()
plt.tight_layout(); plt.savefig(REPORTS / "roc_curve.png", dpi=160); plt.close()

# Precision-Recall
precisions, recalls, _ = precision_recall_curve(yte, proba)
plt.figure()
plt.plot(recalls, precisions)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
plt.tight_layout(); plt.savefig(REPORTS / "pr_curve.png", dpi=160); plt.close()

# Confusion matrix at 0.50
cm = confusion_matrix(yte, pred)
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (thr=0.50)")
plt.colorbar()
for (i,j), val in np.ndenumerate(cm):
    plt.text(j, i, int(val), ha="center", va="center")
plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
plt.tight_layout(); plt.savefig(REPORTS / "confusion_matrix.png", dpi=160); plt.close()

# ---------- save artifacts ----------
joblib.dump(best_model, MODELS / "best_model.joblib")
with open(MODELS / "metrics.json", "w") as f:
    json.dump({
        "model": best_name,
        "cv_roc_auc": float(best_auc),
        "holdout": {
            "roc_auc": float(auc),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "classification_report": report
        },
        "features": {"numeric": num_cols, "categorical": cat_cols}
    }, f, indent=2)

# ---------- scored output for BI (top N targeting) ----------
scored = Xte.copy()
scored["y_true"] = yte.values
scored["churn_proba"] = proba
scored["pred_0.50"] = pred
scored.sort_values("churn_proba", ascending=False).to_csv(
    EXPORTS / "scored_holdout.csv", index=False
)

print(f"[SAVE] model -> {MODELS/'best_model.joblib'}")
print(f"[SAVE] metrics -> {MODELS/'metrics.json'}")
print(f"[SAVE] plots   -> {REPORTS/'roc_curve.png'}, {REPORTS/'pr_curve.png'}, {REPORTS/'confusion_matrix.png'}")
print(f"[SAVE] tables  -> {EXPORTS/'threshold_metrics.csv'}, {EXPORTS/'scored_holdout.csv'}")
print("\n✅ Training complete.")
