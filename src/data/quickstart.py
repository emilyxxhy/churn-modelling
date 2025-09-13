import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

RAW = os.path.join('data_raw', 'Churn_Modelling.csv')
if not os.path.exists(RAW):
    raise FileNotFoundError('Put Churn_Modelling.csv into data_raw/ first.')

os.makedirs('data_intermediate', exist_ok=True)
os.makedirs('data_exports', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# 1) Load + minimal clean
df = pd.read_csv(RAW)
drop_cols = [c for c in ['RowNumber','CustomerId','Surname'] if c in df.columns]
df = df.drop(columns=drop_cols)
df.to_csv(os.path.join('data_intermediate','churn_clean.csv'), index=False)

# 2) Tiny EDA (prints + charts)
churn_rate = df['Exited'].mean()
print(f'Overall churn rate: {churn_rate:.2%}')

by_geo = df.groupby('Geography')['Exited'].mean().sort_values(ascending=False)
print('\nChurn by Geography (top to bottom):')
print((by_geo*100).round(2).astype(str) + '%')

# Save a simple bar chart
plt.figure()
by_geo.plot(kind='bar')
plt.title('Churn Rate by Geography')
plt.ylabel('Churn Rate')
plt.xlabel('Geography')
plt.tight_layout()
plt.savefig(os.path.join('reports','churn_by_geography.png'))
plt.close()

# Age bands for quick matrix export
bins = [0,25,35,45,55,65,200]
labels = ['<25','25–34','35–44','45–54','55–64','65+']
df['AgeBand'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Exports for Excel / BI
geo_gender = (df.groupby(['Geography','Gender'])
                .agg(customers=('Exited','size'),
                     churned=('Exited','sum'))
                .assign(churn_rate=lambda x: x['churned']/x['customers'])
                .reset_index())
geo_gender.to_csv(os.path.join('data_exports','geo_gender.csv'), index=False)

age_tenure = (df.groupby(['AgeBand','Tenure'])
                .agg(customers=('Exited','size'),
                     churned=('Exited','sum'))
                .assign(churn_rate=lambda x: x['churned']/x['customers'])
                .reset_index())
age_tenure.to_csv(os.path.join('data_exports','age_tenure.csv'), index=False)

# 3) Ultra-simple model (LogReg)
y = df['Exited'].astype(int)
X = df.drop(columns=['Exited'])

num_cols = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
num_cols = [c for c in num_cols if c in X.columns]
cat_cols = ['Geography','Gender','HasCrCard','IsActiveMember']
cat_cols = [c for c in cat_cols if c in X.columns]

pre = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

pipe = Pipeline([('pre', pre),
                 ('clf', LogisticRegression(max_iter=200, class_weight='balanced'))])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe.fit(Xtr, ytr)
proba = pipe.predict_proba(Xte)[:,1]
pred = (proba >= 0.5).astype(int)

print('\nModel performance (holdout):')
print('ROC-AUC:', round(roc_auc_score(yte, proba), 3))
print(classification_report(yte, pred, digits=3))

joblib.dump(pipe, os.path.join('data_exports','baseline_logreg.joblib'))
print('Saved model → data_exports/baseline_logreg.joblib')

print('\nDone. Open the PNGs in reports/ and the CSVs in data_exports/.')
