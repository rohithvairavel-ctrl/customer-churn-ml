"""Script mirror of 01_eda.ipynb — run from project root: python notebooks/01_eda.py"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data import load_raw, clean_dataframe, TARGET_COL

OUT = PROJECT_ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

df = load_raw()
clean = clean_dataframe(df)
print("shape", df.shape, "churn rate", clean[TARGET_COL].mean())

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sns.histplot(data=clean, x="tenure", hue=TARGET_COL, bins=24, ax=axes[0], palette="Set1")
sns.boxplot(data=clean, x=TARGET_COL, y="MonthlyCharges", ax=axes[1], palette="Set2")
sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[2], palette="Set1")
fig.tight_layout()
fig.savefig(OUT / "eda_overview.png", dpi=120)
print("wrote", OUT / "eda_overview.png")
