"""
Phishing URL Behaviour Classification
Track 1 Capstone - Problem 02
Learn Depth Academy LLP

This script performs the full pipeline:
1. Load + inspect data
2. EDA (class balance, distributions, correlations, duplicates, outliers)
3. Preprocessing
4. Train Logistic Regression, Decision Tree, KNN
5. Evaluate all three, pick the best
6. Save the final model + scaler for the Streamlit app
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

OUT = "/home/claude/phishing_project/outputs"
MODEL_DIR = "/home/claude/phishing_project/model"
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("/home/claude/phishing_project/data/phishing_dataset.csv")
print("Shape:", df.shape)

# Original UCI encoding uses {-1, 0, 1}. Target 'result': 1 = legitimate, -1 = phishing.
# We remap to a more intuitive binary target: 1 = phishing (suspicious), 0 = legitimate.
df["label"] = df["result"].map({1: 0, -1: 1})   # 1 = phishing, 0 = legitimate
df = df.drop(columns=["result"])

report = {}
report["n_rows_raw"] = int(df.shape[0])
report["n_features"] = int(df.shape[1] - 1)

# ---------------------------------------------------------------
# 2. DATA QUALITY CHECKS
# ---------------------------------------------------------------
missing = df.isnull().sum().sum()
duplicates = df.duplicated().sum()
report["missing_values"] = int(missing)
report["duplicate_rows"] = int(duplicates)

# Class balance
class_counts = df["label"].value_counts().to_dict()
report["class_balance"] = {str(k): int(v) for k, v in class_counts.items()}

plt.figure(figsize=(5, 4))
sns.countplot(x="label", data=df, hue="label", palette="Set2", legend=False)
plt.xticks([0, 1], ["Legitimate (0)", "Phishing (1)"])
plt.title("Class Balance")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT}/01_class_balance.png", dpi=120)
plt.close()

# NOTE on duplicates: because every feature is categorical with only 2-3
# possible values (-1, 0, 1), many distinct websites legitimately share the
# same feature vector. This is a known characteristic of this dataset, not
# a data-entry error, so duplicate rows are RETAINED (dropping them would
# discard real, valid observations and bias the class balance).

# ---------------------------------------------------------------
# 3. CORRELATION WITH TARGET (feature relevance)
# ---------------------------------------------------------------
corr = df.corr(numeric_only=True)["label"].drop("label").sort_values(key=abs, ascending=False)
report["top_10_correlated_features"] = corr.head(10).round(3).to_dict()

plt.figure(figsize=(8, 10))
corr.plot(kind="barh", color=["#d9534f" if v > 0 else "#5bc0de" for v in corr])
plt.title("Feature Correlation with Phishing Label")
plt.xlabel("Correlation")
plt.tight_layout()
plt.savefig(f"{OUT}/02_feature_correlation.png", dpi=120)
plt.close()

plt.figure(figsize=(14, 11))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0, cbar=True)
plt.title("Full Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/03_correlation_heatmap.png", dpi=110)
plt.close()

# ---------------------------------------------------------------
# 4. DISTRIBUTIONS of a few key features
# ---------------------------------------------------------------
key_feats = ["sslfinal_state", "url_of_anchor", "web_traffic", "having_sub_domain", "prefix_suffix"]
fig, axes = plt.subplots(1, len(key_feats), figsize=(20, 4))
for ax, feat in zip(axes, key_feats):
    sns.countplot(x=feat, hue="label", data=df, ax=ax, palette="Set2")
    ax.set_title(feat)
    ax.legend(title="label", labels=["Legit", "Phish"])
plt.tight_layout()
plt.savefig(f"{OUT}/04_key_feature_distributions.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 5. DATA LEAKAGE CHECK
# ---------------------------------------------------------------
# Check if any single feature perfectly predicts the label (correlation ~ 1.0)
leakage_suspects = corr[abs(corr) > 0.9]
report["leakage_check"] = leakage_suspects.to_dict() if len(leakage_suspects) else "No feature has |corr| > 0.9 with label -> no obvious leakage."

# ---------------------------------------------------------------
# 6. PREPROCESSING
# ---------------------------------------------------------------
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

report["train_size"] = int(X_train.shape[0])
report["test_size"] = int(X_test.shape[0])

# ---------------------------------------------------------------
# 7. MODEL TRAINING (3 foundational models)
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
}

results = {}
roc_data = {}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    probs = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
    }

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_data[name] = (fpr, tpr, auc)

    # Confusion matrix plot
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Phish"], yticklabels=["Legit", "Phish"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    plt.savefig(f"{OUT}/cm_{safe_name}.png", dpi=120)
    plt.close()

report["model_results"] = results

# ROC curve comparison
plt.figure(figsize=(6, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/05_roc_comparison.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 8. SELECT BEST MODEL (by F1, tie-break by ROC-AUC)
# ---------------------------------------------------------------
best_name = max(results, key=lambda n: (results[n]["f1_score"], results[n]["roc_auc"]))
best_model = models[best_name]
report["best_model"] = best_name

print(json.dumps(report, indent=2, default=str))

# ---------------------------------------------------------------
# 9. SAVE MODEL + SCALER + FEATURE LIST FOR STREAMLIT APP
# ---------------------------------------------------------------
joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_names.pkl")

with open(f"{OUT}/eda_model_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\nBest model:", best_name)
print("Saved model, scaler, feature list to", MODEL_DIR)
