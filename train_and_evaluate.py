#!/usr/bin/env python3
"""
train_and_evaluate.py

Usage:
  python train_and_evaluate.py --data data/health_data.csv

Produces:
 - outputs/results.json         (metrics, predictions, probabilities, arrays for charts)
 - outputs/*.png                (charts and confusion matrix heatmaps)
 - outputs/metrics_readable.txt (human readable summary)
All outputs are local and do not call external APIs.
"""

import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)

def create_label_by_rule(df):
    # Simple rule-based label: 0 = healthy, 1 = unhealthy
    # thresholds can be adjusted
    hr = df.get("heart_rate")
    temp = df.get("temperature")
    spo2 = df.get("spo2")
    lab = []
    for i in range(len(df)):
        h = float(hr.iloc[i]) if hr is not None else 70.0
        t = float(temp.iloc[i]) if temp is not None else 36.6
        s = float(spo2.iloc[i]) if spo2 is not None else 98.0
        risk = 0
        if s < 95 or t > 38.0 or h > 100 or h < 50:
            risk = 1
        lab.append(risk)
    return pd.Series(lab, index=df.index)

def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(path, nrows=1).columns else None)
    expected_cols = ["heart_rate", "temperature", "spo2"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV must include columns: {expected_cols}. Missing: {missing}")
    if "label" not in df.columns and "status" not in df.columns:
        df["label"] = create_label_by_rule(df)
    elif "label" not in df.columns:
        # convert status -> label (assume 'healthy'/'unhealthy' or 0/1)
        df["label"] = df["status"].map(lambda v: 1 if str(v).lower() not in ("healthy","0","0.0","false","false") else 0)
    return df

def train_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    return rf, svm

def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "y_pred": y_pred.tolist(),
        "y_prob": probs.tolist() if probs is not None else None
    }

def cross_val_scores(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return float(np.mean(scores))

def save_json(data, path="outputs/results.json"):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda o: (o.tolist() if hasattr(o,"tolist") else str(o)))

def plot_trend(df, column, outpath):
    """Create a scatter plot of x vs y where x is time and y is level.
    If a timestamp column exists, use it for x. Otherwise fall back to index as pseudo-time.
    Y-axis label is 'Level' per request.
    """
    plt.figure(figsize=(8,3))
    if "timestamp" in df.columns:
        x = df["timestamp"]
        plt.scatter(x, df[column], s=30)
        plt.gcf().autofmt_xdate()
        plt.xlabel("Time")
    else:
        x = np.arange(len(df))
        plt.scatter(x, df[column].values, s=30)
        plt.xlabel("Index (pseudo-time)")
    plt.title(f"{column} Scatter (Time vs Level)")
    plt.ylabel("Level")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_confusion_matrix(cm, labels, outpath, title="Confusion Matrix"):
    plt.figure(figsize=(4,3))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main(args):
    ensure_dirs()
    print("Loading data:", args.data)
    df = load_data(args.data)
    features = ["heart_rate", "temperature", "spo2"]
    X = df[features].values
    y = df["label"].astype(int).values

    # Train/test split
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(X, y, df, test_size=0.3, random_state=42, stratify=y)

    # Train models
    print("Training models...")
    rf, svm = train_models(X_train, y_train)

    # Compute metrics
    print("Evaluating Random Forest...")
    rf_metrics = compute_metrics(rf, X_test, y_test)
    print("Evaluating SVM...")
    svm_metrics = compute_metrics(svm, X_test, y_test)

    # CV scores
    print("Computing CV scores (5-fold)...")
    rf_cv = cross_val_scores(RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=5)
    svm_cv = cross_val_scores(SVC(kernel="rbf", probability=True, random_state=42), X, y, cv=5)

    # Agreement and model-level outputs
    rf_preds = np.array(rf_metrics["y_pred"])
    svm_preds = np.array(svm_metrics["y_pred"])
    agreement = bool(np.array_equal(rf_preds, svm_preds))
    rf_probs = np.array(rf_metrics["y_prob"]) if rf_metrics["y_prob"] is not None else None
    svm_probs = np.array(svm_metrics["y_prob"]) if svm_metrics["y_prob"] is not None else None
    svm_decision = svm.decision_function(X_test).tolist() if hasattr(svm, "decision_function") else None

    # Save plots
    print("Generating charts...")
    plot_trend(df, "heart_rate", "outputs/heart_rate_trend.png")
    plot_trend(df, "temperature", "outputs/temperature_trend.png")
    plot_trend(df, "spo2", "outputs/spo2_trend.png")
    plot_confusion_matrix(rf_metrics["confusion_matrix"], labels=["healthy","unhealthy"], outpath="outputs/rf_confusion.png", title="Random Forest Confusion Matrix")
    plot_confusion_matrix(svm_metrics["confusion_matrix"], labels=["healthy","unhealthy"], outpath="outputs/svm_confusion.png", title="SVM Confusion Matrix")

    # Model performance grouped bar chart
    perf_df = pd.DataFrame({
        "metric": ["accuracy","precision","recall","f1_score"],
        "RandomForest": [rf_metrics["accuracy"], rf_metrics["precision"], rf_metrics["recall"], rf_metrics["f1_score"]],
        "SVM": [svm_metrics["accuracy"], svm_metrics["precision"], svm_metrics["recall"], svm_metrics["f1_score"]],
    })
    plt.figure(figsize=(6,4))
    x = np.arange(len(perf_df))
    width = 0.35
    plt.bar(x - width/2, perf_df["RandomForest"], width, label="RandomForest")
    plt.bar(x + width/2, perf_df["SVM"], width, label="SVM")
    plt.xticks(x, perf_df["metric"])
    plt.ylim(0,1)
    plt.legend()
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.savefig("outputs/model_performance_comparison.png")
    plt.close()

    # Rule-based overall risk category using the last row
    last = df.iloc[-1]
    rule_risk = "Low"
    steps = "No immediate action"
    if (last["spo2"] < 95) or (last["temperature"] > 38.0) or (last["heart_rate"] > 100) or (last["heart_rate"] < 50):
        rule_risk = "High"
        steps = "Recommend urgent clinical assessment"
    elif (last["spo2"] < 97) or (last["temperature"] > 37.5) or (last["heart_rate"] > 90):
        rule_risk = "Moderate"
        steps = "Monitor closely and consider tele-consult"

    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_performance": {
            "random_forest": {
                "accuracy": rf_metrics["accuracy"],
                "precision": rf_metrics["precision"],
                "recall": rf_metrics["recall"],
                "f1_score": rf_metrics["f1_score"],
                "cv_score": rf_cv,
                "confusion_matrix": rf_metrics["confusion_matrix"],
            },
            "svm": {
                "accuracy": svm_metrics["accuracy"],
                "precision": svm_metrics["precision"],
                "recall": svm_metrics["recall"],
                "f1_score": svm_metrics["f1_score"],
                "cv_score": svm_cv,
                "confusion_matrix": svm_metrics["confusion_matrix"],
            }
        },
        "connectivity_checks": {
            "note": "Connectivity checks performed in browser dashboard (navigator.onLine, HTTP fetch to results.json with timeout, localStorage check)."
        },
        "rule_based": {
            "overall_risk_category": rule_risk,
            "recommended_next_step": steps
        },
        "ml_outputs": {
            "rf_predicted_status": int(rf.predict(X[-1].reshape(1,-1))[0]),
            "svm_predicted_status": int(svm.predict(X[-1].reshape(1,-1))[0]),
            "model_agreement": agreement,
            "rf_probability_last": rf.predict_proba(X[-1].reshape(1,-1)).tolist()[0] if hasattr(rf, "predict_proba") else None,
            "svm_probability_last": svm.predict_proba(X[-1].reshape(1,-1)).tolist()[0] if hasattr(svm, "predict_proba") else None,
            "svm_decision_function_last": svm.decision_function(X[-1].reshape(1,-1)).tolist() if hasattr(svm, "decision_function") else None,
            "svm_support_vectors_count": int(len(svm.support_)) if hasattr(svm, "support_") else None
        },
        "charts": {
            "heart_rate_trend": "outputs/heart_rate_trend.png",
            "temperature_trend": "outputs/temperature_trend.png",
            "spo2_trend": "outputs/spo2_trend.png",
            "model_performance_comparison": "outputs/model_performance_comparison.png",
            "rf_confusion_matrix": "outputs/rf_confusion.png",
            "svm_confusion_matrix": "outputs/svm_confusion.png"
        },
        "raw": {
            "test_true": y_test.tolist(),
            "rf_test_pred": rf_metrics["y_pred"],
            "svm_test_pred": svm_metrics["y_pred"],
            "rf_test_prob": rf_metrics["y_prob"],
            "svm_test_prob": svm_metrics["y_prob"],
            "svm_decision": svm_decision
        }
    }

    save_json(results, "outputs/results.json")

    # human-readable summary
    with open("outputs/metrics_readable.txt", "w") as f:
        f.write("Model Performance Summary\n")
        f.write("=========================\n\n")
        f.write("Random Forest:\n")
        for k,v in results["model_performance"]["random_forest"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nSVM:\n")
        for k,v in results["model_performance"]["svm"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nRule-based:\n")
        for k,v in results["rule_based"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nML outputs:\n")
        for k,v in results["ml_outputs"].items():
            f.write(f"  {k}: {v}\n")

    print("Done. Outputs written to outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file with columns timestamp (opt), heart_rate, temperature, spo2, label (opt)")
    args = parser.parse_args()
    main(args)
