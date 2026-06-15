import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =========================================================
# PATHS
# =========================================================

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "irrigation_prediction.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

data = pd.read_csv(DATA_PATH)
print(f"[INFO] Dataset loaded: {data.shape[0]} rows × {data.shape[1]} cols")
print(f"[INFO] Class distribution:\n{data['Irrigation_Need'].value_counts()}\n")

# =========================================================
# ENCODING
# =========================================================

# أعمدة data leakage — لا تُستخدم كـ features
LEAKAGE_COLS = ["Irrigation_Type"]

encoders = {}
data_enc = data.copy()

for col in data_enc.columns:
    data_enc[col] = data_enc[col].astype(str)
    if col not in (["Irrigation_Need"] + LEAKAGE_COLS):
        le = LabelEncoder()
        data_enc[col] = le.fit_transform(data_enc[col])
        encoders[col] = le

target_encoder = LabelEncoder()
data_enc["Irrigation_Need"] = target_encoder.fit_transform(
    data_enc["Irrigation_Need"]
)

print(f"[INFO] Classes: {list(target_encoder.classes_)}")
print(f"[INFO] Removed leakage columns: {LEAKAGE_COLS}\n")

# =========================================================
# SPLIT
# =========================================================

X = data_enc.drop(["Irrigation_Need"] + LEAKAGE_COLS, axis=1)
y = data_enc["Irrigation_Need"]

FEATURE_COLS = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")
print(f"[INFO] Train class dist before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}\n"
      if False else "")

# =========================================================
# BALANCING (SMOTE)
# =========================================================

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

import numpy as np
unique, counts = np.unique(y_train_res, return_counts=True)
print(f"[INFO] After SMOTE — classes: {dict(zip(target_encoder.classes_, counts))}\n")

# ── حفظ بيانات التدريب الموازنة للـ DriftRetrainer ─────────
joblib.dump(
    (X_train_res, y_train_res),
    os.path.join(MODEL_DIR, "train_data_smote.pkl"),
)
print("[INFO] SMOTE train data saved → models/train_data_smote.pkl")

# =========================================================
# MODELS
# =========================================================

models = {
    "LogisticRegression": LogisticRegression(max_iter=3000),
    "DecisionTree"      : DecisionTreeClassifier(max_depth=12),
    "RandomForest"      : RandomForestClassifier(
        n_estimators=250, max_depth=15, random_state=42
    ),
    "SVM"               : SVC(probability=True),
    "KNN"               : KNeighborsClassifier(),
    "XGBoost"           : XGBClassifier(
        n_estimators=250,
        max_depth=10,
        learning_rate=0.05,
        eval_metric="mlogloss",
        verbosity=0,
    ),
}

# =========================================================
# TRAIN + EVALUATE
# =========================================================

results  = []
best_model = None
best_name  = ""
best_f1    = 0.0

for name, model in models.items():
    print(f"[TRAINING] {name} ...")

    start = time.perf_counter()
    model.fit(X_train_res, y_train_res)
    latency_ms = (time.perf_counter() - start) * 1000

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # تقييم لكل فئة منفصلة (مهم لفئة High التي 3.4% فقط)
    report = classification_report(
        y_test, y_pred,
        target_names=target_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    high_f1  = report.get("High",  {}).get("f1-score", 0.0)
    high_rec = report.get("High",  {}).get("recall",   0.0)

    results.append({
        "Model"      : name,
        "Accuracy"   : acc,
        "Precision"  : prec,
        "Recall"     : rec,
        "F1"         : f1,
        "High_F1"    : high_f1,
        "High_Recall": high_rec,
        "Latency_ms" : latency_ms,
    })

    print(
        f"  Acc={acc:.4f} | F1={f1:.4f} | "
        f"High-F1={high_f1:.4f} | High-Recall={high_rec:.4f} | "
        f"Latency={latency_ms:.0f}ms"
    )

    if f1 > best_f1:
        best_f1    = f1
        best_model = model
        best_name  = name

print(f"\n[BEST] {best_name} — F1={best_f1:.4f}")

# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(MODEL_DIR, "model_comparison_table.csv"),
    index=False,
)
print("✔ Comparison table saved")

# =========================================================
# SAVE BEST MODEL
# =========================================================

# حساب accuracy الصحيحة للنموذج الأفضل
best_row     = results_df.loc[results_df["F1"].idxmax()]
best_acc_val = float(best_row["Accuracy"])   # ← accuracy حقيقية وليس F1
best_f1_val  = float(best_row["F1"])

joblib.dump(best_model,    os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(encoders,      os.path.join(MODEL_DIR, "encoders.pkl"))
joblib.dump(target_encoder,os.path.join(MODEL_DIR, "target_encoder.pkl"))

meta = {
    "best_model"  : best_name,
    "f1_score"    : best_f1_val,
    "accuracy"    : best_acc_val,        # ← مُصلَّح: accuracy حقيقية
    "features"    : FEATURE_COLS,
    "classes"     : list(target_encoder.classes_),
    "leakage_cols": LEAKAGE_COLS,
    "version"     : "4.0-adaptive-hybrid",
    "trained_at"  : datetime.now().isoformat(),
}

joblib.dump(meta, os.path.join(MODEL_DIR, "model_meta.pkl"))
print(f"✔ Best model saved: {best_name}")
print(f"  Accuracy={best_acc_val:.4f} | F1={best_f1_val:.4f}")

# =========================================================
# CHARTS — F1 Comparison
# =========================================================

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["F1"], color="steelblue")
plt.title("Model Comparison (F1 Score — Weighted)")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"))
plt.close()
print("✔ F1 comparison chart saved")

# =========================================================
# CHARTS — High class F1 (الأهم زراعياً)
# =========================================================

plt.figure(figsize=(8, 5))
bars = plt.bar(results_df["Model"], results_df["High_F1"], color="tomato")
for bar, val in zip(bars, results_df["High_F1"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.3f}", ha="center", va="bottom", fontsize=9,
    )
plt.title("Model Comparison — F1 Score for 'High' Irrigation Class")
plt.ylabel("F1 (High class)")
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "model_comparison_high_f1.png"))
plt.close()
print("✔ High-class F1 chart saved")

# =========================================================
# CHARTS — Accuracy, Precision, Recall per model
# =========================================================

for metric in ["Accuracy", "Precision", "Recall"]:
    plt.figure(figsize=(8, 5))
    bars = plt.bar(results_df["Model"], results_df[metric], color="steelblue")
    for bar, val in zip(bars, results_df[metric]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9,
        )
    plt.title(f"Model Comparison ({metric})")
    plt.ylabel(metric)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"model_comparison_{metric.lower()}.png"))
    plt.close()
    print(f"✔ {metric} chart saved")

# =========================================================
# CONFUSION MATRIX — Best model
# =========================================================

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=target_encoder.classes_,
    yticklabels=target_encoder.classes_,
)
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.close()
print("✔ Confusion matrix saved")

# =========================================================
# CONFUSION MATRIX — كل نموذج
# =========================================================

for name, model in models.items():
    y_pred_m = model.predict(X_test)
    cm_m     = confusion_matrix(y_test, y_pred_m)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm_m, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_encoder.classes_,
        yticklabels=target_encoder.classes_,
    )
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"confusion_matrix_{name}.png"))
    plt.close()
print("✔ All confusion matrices saved")

# =========================================================
# CLASSIFICATION REPORT — Best model (طباعة كاملة)
# =========================================================

print(f"\n{'='*50}")
print(f"Classification Report — {best_name}")
print('='*50)
print(classification_report(
    y_test, y_pred_best,
    target_names=target_encoder.classes_,
    zero_division=0,
))
