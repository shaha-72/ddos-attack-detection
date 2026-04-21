# ================= IMPORTS =================
import pandas as pd
import numpy as np
import os
import json
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, f1_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from xgboost import XGBClassifier


# ================= DATA LOADING =================
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"Loaded {len(df):,} rows — {df['Label'].sum():,} attacks ({df['Label'].mean():.2%})")
    return df


# ================= AUDIT =================
def audit_for_leakage(df):
    print("\n=== LEAKAGE AUDIT ===")

    numeric = df.select_dtypes(include=[np.number])
    benign  = numeric[df['Label'] == 0]
    attack  = numeric[df['Label'] == 1]

    diff = (attack.mean() - benign.mean()).abs().sort_values(ascending=False)
    print("\nTop 10 feature differences:")
    print(diff.head(10))

    print("\nLowest variance features:")
    print(numeric.var().sort_values().head(10))

    print(f"\nClass balance: {df['Label'].value_counts().to_dict()}")
    print(f"Attack ratio : {df['Label'].mean():.2%}")


# ================= VISUALIZATION =================
def plot_heatmap(df):
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, cmap='coolwarm', linewidths=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=150)
    plt.close()
    print("Saved heatmap.png")


def plot_confusion(y_test, pred, name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{name.replace(' ', '_')}_cm.png", dpi=150)
    plt.close()


def plot_roc_pr(results, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["probs"])
        ax1.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.4f})")

        prec, rec, _ = precision_recall_curve(y_test, res["probs"])
        ax2.plot(rec, prec, label=f"{name} (AUC={auc(rec, prec):.4f})")

    ax1.plot([0, 1], [0, 1], 'k--', label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("roc_pr_curves.png", dpi=150)
    plt.close()
    print("Saved roc_pr_curves.png")


def plot_feature_importance(model, feature_names, title, filename):
    if not hasattr(model, "feature_importances_"):
        return
    importance = model.feature_importances_
    indices = np.argsort(importance)[-15:]

    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importance[indices], color='steelblue')
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


# ================= DATA SPLIT =================
def split_data(df):
    """Three-way stratified split: 70% train / 10% val / 20% test"""

    if 'Timestamp' in df.columns:
        print("\n=== TIME-BASED CLASS DISTRIBUTION ===")
        df = df.sort_values('Timestamp')
        chunk_size = len(df) // 10
        for i in range(10):
            chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size]
            ratio = chunk['Label'].mean()
            bar = '█' * int(ratio * 20)
            print(f"  Chunk {i+1:02d}: {ratio:5.1%} attacks  {bar}")

    # Step 1: 80% trainval / 20% test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_trainval, idx_test = next(sss1.split(df, df['Label']))
    trainval_df = df.iloc[idx_trainval]
    test_df     = df.iloc[idx_test]

    # Step 2: Split trainval into 87.5% train / 12.5% val → yields 70/10 of total
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    idx_train, idx_val = next(sss2.split(trainval_df, trainval_df['Label']))
    train_df = trainval_df.iloc[idx_train]
    val_df   = trainval_df.iloc[idx_val]

    print(f"\nTrain : {len(train_df):>7,} rows | attack ratio: {train_df['Label'].mean():.2%}")
    print(f"Val   : {len(val_df):>7,} rows | attack ratio: {val_df['Label'].mean():.2%}")
    print(f"Test  : {len(test_df):>7,} rows | attack ratio: {test_df['Label'].mean():.2%}")

    return train_df, val_df, test_df


# ================= PREPROCESSING =================
def preprocess(train_df, val_df, test_df):
    """All fitting (corr removal, scaling) done on train only — no leakage."""

    drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']

    # Constant/zero-variance columns identified during audit
    zero_var_cols = [
        'Bwd PSH Flags', 'Fwd Avg Bulk Rate', 'Bwd Avg Bulk Rate',
        'Bwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk', 'Fwd Avg Bytes/Bulk',
        'Bwd URG Flags', 'Fwd Avg Packets/Bulk', 'CWE Flag Count', 'Fwd URG Flags'
    ]
    drop_cols.extend(zero_var_cols)

    def clean(df):
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return df.select_dtypes(include=[np.number])

    X_train = clean(train_df.drop('Label', axis=1))
    X_val   = clean(val_df.drop('Label', axis=1))
    X_test  = clean(test_df.drop('Label', axis=1))

    y_train = train_df['Label'].reset_index(drop=True)
    y_val   = val_df['Label'].reset_index(drop=True)
    y_test  = test_df['Label'].reset_index(drop=True)

    # Remove highly correlated features — fit on train only
    corr_matrix   = X_train.corr().abs()
    upper         = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.97)]
    print(f"\nDropping {len(high_corr_cols)} highly correlated features")

    X_train = X_train.drop(columns=high_corr_cols)
    X_val   = X_val.drop(columns=[c for c in high_corr_cols if c in X_val.columns])
    X_test  = X_test.drop(columns=[c for c in high_corr_cols if c in X_test.columns])

    feature_names = X_train.columns

    # Scale — fit on train only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"Final feature count: {X_train.shape[1]}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names


# ================= MODEL TRAINING =================
def train_models(X_train, y_train):
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    models = {}

    print("\nTraining Random Forest...")
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ).fit(X_train, y_train)

    print("Training Logistic Regression...")
    models["Logistic Regression"] = LogisticRegression(
        max_iter=2000,
        solver='liblinear',
        class_weight='balanced',
        C=0.1
    ).fit(X_train, y_train)

    print("Training XGBoost...")
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    ).fit(X_train, y_train)

    return models


# ================= THRESHOLD SELECTION (VAL SET ONLY) =================
def find_best_threshold(model, X_val, y_val, model_name):
    """Sweep thresholds on VALIDATION set — test set is never touched here."""
    probs   = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0

    for t in np.arange(0.10, 0.91, 0.02):
        pred = (probs > t).astype(int)
        f1   = f1_score(y_val, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"  {model_name:20s}: val threshold={best_t:.2f}  val F1={best_f1:.4f}")
    return round(best_t, 2)


# ================= EVALUATION (TEST SET) =================
def evaluate(models, X_val, y_val, X_test, y_test, feature_names):
    results = {}
    print("\n=== THRESHOLD SELECTION (validation set) ===")

    for name, model in models.items():
        # 1. Pick threshold on val — no test peeking
        best_t = find_best_threshold(model, X_val, y_val, name)

        # 2. Evaluate on test set using that threshold
        probs = model.predict_proba(X_test)[:, 1]
        pred  = (probs > best_t).astype(int)

        print(f"\n--- {name} (threshold={best_t:.2f}) ---")
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))

        plot_confusion(y_test, pred, name)

        results[name] = {
            "model"    : model,
            "pred"     : pred,
            "probs"    : probs,
            "threshold": best_t,
            "recall"   : recall_score(y_test, pred),
            "f1"       : f1_score(y_test, pred)
        }

    # Feature importance plots for tree-based models
    plot_feature_importance(
        models["Random Forest"], feature_names,
        "Random Forest — Top 15 Features", "rf_importance.png"
    )
    plot_feature_importance(
        models["XGBoost"], feature_names,
        "XGBoost — Top 15 Features", "xgb_importance.png"
    )

    return results


# ================= SANITY CHECK =================
def sanity_check_test_set(y_test, results):
    print("\n=== TEST SET SANITY CHECK ===")
    print(f"Test attack ratio        : {y_test.mean():.2%}")
    print(f"Naive all-benign accuracy: {1 - y_test.mean():.2%}")
    print()
    for name, res in results.items():
        pct = res["pred"].mean()
        match = "✅" if abs(pct - y_test.mean()) < 0.02 else "⚠️ "
        print(f"  {match} {name:20s}: predicted {pct:.2%} attack  (actual {y_test.mean():.2%})")


# ================= CROSS VALIDATION =================
def validate_robustness(X_train, y_train):
    """Stratified K-Fold CV — each fold has balanced class ratios."""
    print("\n=== STRATIFIED CROSS VALIDATION ===")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        random_state=42, eval_metric='logloss', n_jobs=-1
    )

    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        xgb.fit(X_train[tr_idx], y_train.iloc[tr_idx])
        pred = xgb.predict(X_train[val_idx])

        f1  = f1_score(y_train.iloc[val_idx], pred)
        rec = recall_score(y_train.iloc[val_idx], pred)
        att = int(y_train.iloc[val_idx].sum())
        scores.append(f1)

        print(f"  Fold {fold+1}: F1={f1:.4f}  Recall={rec:.4f}  Attacks={att:,}")

    std = np.std(scores)
    print(f"\n  Mean F1 : {np.mean(scores):.4f}")
    print(f"  Std  F1 : {std:.4f}")
    print(f"  Verdict : {'✅ Stable — model generalizes well' if std < 0.03 else '⚠️  Unstable — review folds'}")

    return scores


# ================= SAVE MODEL =================
def save_best(results, scaler, feature_names):
    best_name = max(results, key=lambda x: results[x]["f1"])
    best      = results[best_name]

    os.makedirs("models", exist_ok=True)
    joblib.dump(best["model"],          "models/best_model.pkl")
    joblib.dump(scaler,                 "models/scaler.pkl")
    joblib.dump(list(feature_names),    "models/feature_names.pkl")
    joblib.dump(best["threshold"],      "models/threshold.pkl")
    joblib.dump(best_name,              "models/model_name.pkl")

    card = {
        "model"      : best_name,
        "f1_test"    : round(float(best["f1"]), 6),
        "recall_test": round(float(best["recall"]), 6),
        "threshold"  : round(float(best["threshold"]), 4),
        "features"   : int(len(feature_names)),
        "dataset"    : "CIC-DDoS2019",
        "trained_at" : datetime.datetime.now().isoformat(),
        "warning"    : (
            "Trained on lab-generated traffic. "
            "Validate on real-world captures before production deployment."
        )
    }
    with open("models/model_card.json", "w") as f:
        json.dump(card, f, indent=2)

    print(f"\n🏆 Best model : {best_name}")
    print(f"   F1 (test)  : {best['f1']:.4f}")
    print(f"   Recall     : {best['recall']:.4f}")
    print(f"   Threshold  : {best['threshold']:.2f}")
    print("   Saved      : models/best_model.pkl, scaler.pkl, feature_names.pkl,")
    print("                threshold.pkl, model_card.json")

    return best["pred"]


# ================= INFERENCE =================
def predict_live(flow_features: dict) -> dict:
    """
    Score a single network flow against the saved model.

    Args:
        flow_features: dict of {feature_name: value}

    Returns:
        {"label": "ATTACK"/"BENIGN", "confidence": float, "threshold": float}

    Example:
        result = predict_live({"Flow Duration": 100, "Fwd IAT Max": 50})
        print(result)  # {"label": "BENIGN", "confidence": 0.03, "threshold": 0.82}
    """
    model      = joblib.load("models/best_model.pkl")
    scaler     = joblib.load("models/scaler.pkl")
    threshold  = joblib.load("models/threshold.pkl")
    feat_names = joblib.load("models/feature_names.pkl")

    # Align columns — missing features filled with 0
    row       = pd.DataFrame([flow_features]).reindex(columns=feat_names, fill_value=0)
    row_scaled = scaler.transform(row)

    prob  = float(model.predict_proba(row_scaled)[0, 1])
    label = "ATTACK" if prob > threshold else "BENIGN"

    return {
        "label"     : label,
        "confidence": round(prob, 4),
        "threshold" : threshold
    }


# ================= MAIN =================
def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_data(r"D:\DDOS Attack Detection\data\DDoS2.csv")

    # ── Audit ─────────────────────────────────────────────────────────────────
    audit_for_leakage(df)
    plot_heatmap(df)

    # ── Split ─────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = split_data(df)

    # ── Preprocess ────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = \
        preprocess(train_df, val_df, test_df)

    # ── Train ─────────────────────────────────────────────────────────────────
    models = train_models(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = evaluate(models, X_val, y_val, X_test, y_test, feature_names)
    sanity_check_test_set(y_test, results)
    plot_roc_pr(results, y_test)

    # ── Validate ──────────────────────────────────────────────────────────────
    validate_robustness(X_train, y_train)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_best(results, scaler, feature_names)

    # ── Final Report ──────────────────────────────────────────────────────────
    best_name = max(results, key=lambda x: results[x]["f1"])
    print(f"\n{'='*50}")
    print(f"FINAL REPORT — {best_name} on held-out test set")
    print('='*50)
    print(classification_report(y_test, results[best_name]["pred"]))


if __name__ == "__main__":
    main()