import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "data_exports"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

def pct_axis(ax):
    ax.set_ylim(0, max(0.01, ax.get_ylim()[1]))
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}%")

def read_csv(name):
    fp = EXPORTS / name
    if not fp.exists():
        print(f"[SKIP] {name} not found in data_exports/")
        return None
    return pd.read_csv(fp)

def save_fig(name):
    out = REPORTS / name
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote {out.relative_to(ROOT)}")

# ---------- 1) Geography ----------
df_geo = read_csv("by_geography.csv")
if df_geo is not None and "churn_rate_pct" in df_geo.columns:
    # Convert to rate 0..1
    df_geo["churn_rate"] = df_geo["churn_rate_pct"] / 100.0
    df_geo = df_geo.sort_values("churn_rate", ascending=False)

    plt.figure(figsize=(7,4.5))
    plt.bar(df_geo["Geography"], df_geo["churn_rate"])
    plt.title("Churn Rate by Geography")
    plt.ylabel("Churn Rate")
    pct_axis(plt.gca())
    save_fig("churn_by_geography.png")

# ---------- 2) Age bands ----------
df_age = read_csv("by_age_band.csv")
if df_age is not None and "churn_rate_pct" in df_age.columns:
    order = ['<25','25–34','35–44','45–54','55–64','65+']
    df_age["churn_rate"] = df_age["churn_rate_pct"] / 100.0
    df_age["AgeBand"] = pd.Categorical(df_age["AgeBand"], categories=order, ordered=True)
    df_age = df_age.sort_values("AgeBand")

    plt.figure(figsize=(8,4))
    plt.bar(df_age["AgeBand"].astype(str), df_age["churn_rate"])
    plt.title("Churn Rate by Age Band")
    plt.ylabel("Churn Rate")
    pct_axis(plt.gca())
    save_fig("churn_by_age_band.png")

# ---------- 3) Tenure bands ----------
df_ten = read_csv("by_tenure_band.csv")
if df_ten is not None and "churn_rate_pct" in df_ten.columns:
    order = ['0–1','2–4','5–7','8–10']
    df_ten["churn_rate"] = df_ten["churn_rate_pct"] / 100.0
    df_ten["TenureBand"] = pd.Categorical(df_ten["TenureBand"], categories=order, ordered=True)
    df_ten = df_ten.sort_values("TenureBand")

    plt.figure(figsize=(8,4))
    plt.bar(df_ten["TenureBand"].astype(str), df_ten["churn_rate"])
    plt.title("Churn Rate by Tenure Band")
    plt.ylabel("Churn Rate")
    pct_axis(plt.gca())
    save_fig("churn_by_tenure_band.png")

# ---------- 4) Products ----------
df_prod = read_csv("by_num_products.csv")
if df_prod is not None and "churn_rate_pct" in df_prod.columns:
    df_prod["churn_rate"] = df_prod["churn_rate_pct"] / 100.0

    plt.figure(figsize=(7,4))
    plt.bar(df_prod["NumOfProducts"].astype(str), df_prod["churn_rate"])
    plt.title("Churn Rate by Num of Products")
    plt.xlabel("NumOfProducts")
    plt.ylabel("Churn Rate")
    pct_axis(plt.gca())
    save_fig("churn_by_products.png")

# ---------- 5) Active member / credit card (side-by-side small charts) ----------
df_active = read_csv("by_is_active_member.csv")
if df_active is not None and "churn_rate_pct" in df_active.columns:
    df_active["label"] = df_active["IsActiveMember"].map({0:"No",1:"Yes"})
    df_active["churn_rate"] = df_active["churn_rate_pct"]/100.0

    plt.figure(figsize=(5,4))
    plt.bar(df_active["label"], df_active["churn_rate"])
    plt.title("Churn Rate by Active Member")
    plt.ylabel("Churn Rate")
    pct_axis(plt.gca())
    save_fig("churn_by_active_member.png")

df_card = read_csv("by_has_credit_card.csv")
if df_card is not None and "churn_rate_pct" in df_card.columns:
    df_card["label"] = df_card["HasCrCard"].map({0:"No",1:"Yes"})
    df_card["churn_rate"] = df_card["churn_rate_pct"]/100.0

    plt.figure(figsize=(5,4))
    plt.bar(df_card["label"], df_card["churn_rate"])
    plt.title("Churn Rate by Credit Card Ownership")
    plt.ylabel("Churn Rate")
    pct_axis(plt.gca())
    save_fig("churn_by_credit_card.png")

# ---------- 6) Matrix heatmap: AgeBand x TenureBand ----------
df_mat = read_csv("ageband_by_tenureband.csv")
if df_mat is not None and all(c in df_mat.columns for c in ["AgeBand","TenureBand","churn_rate_pct"]):
    order_age = ['<25','25–34','35–44','45–54','55–64','65+']
    order_ten = ['0–1','2–4','5–7','8–10']
    # pivot to matrix of rates (0..1)
    piv = (df_mat
           .assign(churn_rate=lambda d: d["churn_rate_pct"]/100.0)
           .pivot(index="AgeBand", columns="TenureBand", values="churn_rate")
           .reindex(index=order_age, columns=order_ten))

    plt.figure(figsize=(7.2,4.8))
    plt.imshow(piv, aspect="auto")
    plt.title("Churn Rate Heatmap: Age × Tenure")
    plt.xlabel("TenureBand")
    plt.ylabel("AgeBand")
    # axis ticks
    plt.xticks(ticks=np.arange(len(order_ten)), labels=order_ten)
    plt.yticks(ticks=np.arange(len(order_age)), labels=order_age)
    # colorbar with %
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Churn Rate")
    save_fig("heatmap_age_by_tenure.png")

# ---------- 7) Geography × Products (stacked columns of churn rate) ----------
df_gp = read_csv("geo_by_products.csv")
if df_gp is not None and all(c in df_gp.columns for c in ["Geography","NumOfProducts","churn_rate_pct"]):
    df_gp["churn_rate"] = df_gp["churn_rate_pct"]/100.0
    geos = df_gp["Geography"].unique().tolist()
    prods = sorted(df_gp["NumOfProducts"].unique().tolist())

    width = 0.15
    x = np.arange(len(geos))
    plt.figure(figsize=(9,4.8))
    for i, p in enumerate(prods):
        y = [df_gp.loc[(df_gp["Geography"]==g)&(df_gp["NumOfProducts"]==p), "churn_rate"].mean() for g in geos]
        plt.bar(x + i*width, y, width=width, label=str(p))
    plt.title("Churn Rate by Geography × NumOfProducts")
    plt.xlabel("Geography")
    plt.ylabel("Churn Rate")
    plt.xticks(x + (len(prods)-1)*width/2, geos)
    plt.legend(title="NumOfProducts", frameon=False)
    pct_axis(plt.gca())
    save_fig("churn_geo_by_products.png")

print("\n✅ All available charts exported to reports/")
