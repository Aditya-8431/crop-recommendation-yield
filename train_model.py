"""
Machine Learning Training Pipeline for Crop Recommendation

This script:
1. Loads the crop recommendation dataset.
2. Validates features and target label.
3. Splits data into 80% train and 20% test sets (stratified).
4. Trains a RandomForestClassifier baseline model.
5. Evaluates model performance on the test set.
6. Saves the trained model and evaluation reports.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def run_training_pipeline():
    # Define file paths
    data_path = os.path.join("data", "Crop_recommendation.csv")
    models_dir = "models"
    reports_dir = "reports"
    model_path = os.path.join(models_dir, "crop_recommender.joblib")
    metrics_path = os.path.join(reports_dir, "metrics.txt")
    class_report_path = os.path.join(reports_dir, "classification_report.txt")
    confusion_matrix_path = os.path.join(reports_dir, "confusion_matrix.csv")

    # Ensure output directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load Dataset
    print(f"Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    df = pd.read_csv(data_path)

    # Verify expected columns
    expected_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    target_column = "label"
    expected_columns = expected_features + [target_column]

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Columns mismatch! Expected {expected_columns}, but found {list(df.columns)}"
        )

    # Separate features (X) and target (y)
    X = df[expected_features]
    y = df[target_column]

    print(f"Dataset shape: {df.shape}")
    print(f"Features: {expected_features}")
    print(f"Target: {target_column} ({y.nunique()} unique crops)")

    # 2. Train-Test Split (80% Train, 20% Test, stratified by crop label)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 3. Model Training
    # RandomForestClassifier is an ensemble of decision trees that handles non-linear relationships well.
    print("Training RandomForestClassifier model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Model training completed successfully.")

    # 4. Model Evaluation on Test Set
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    class_report_str = classification_report(y_test, y_pred, digits=4)
    unique_labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)

    # Print Evaluation Results to Console
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS (Held-Out Test Set)")
    print("=" * 50)
    print(f"Accuracy:           {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro Precision:    {precision_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Macro Recall:       {recall_macro:.4f}")
    print(f"Weighted Recall:    {recall_weighted:.4f}")
    print(f"Macro F1-Score:     {f1_macro:.4f}")
    print(f"Weighted F1-Score:  {f1_weighted:.4f}")
    print("\nDetailed Classification Report:\n")
    print(class_report_str)

    # 5. Save Model and Reports
    print("=" * 50)
    print("SAVING ARTIFACTS")
    print("=" * 50)

    # Save model
    joblib.dump(model, model_path)
    print(f"Trained model saved to: {model_path}")

    # Save classification report
    with open(class_report_path, "w", encoding="utf-8") as f:
        f.write("Crop Recommendation - Detailed Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(class_report_str)
    print(f"Classification report saved to: {class_report_path}")

    # Save confusion matrix as CSV
    cm_df.to_csv(confusion_matrix_path)
    print(f"Confusion matrix saved to: {confusion_matrix_path}")

    # Save summary metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Crop Recommendation - Summary Metrics\n")
        f.write("=" * 45 + "\n")
        f.write(f"Model: RandomForestClassifier(n_estimators=100, random_state=42)\n")
        f.write(f"Test Split Ratio: 0.20 (Stratified)\n")
        f.write(f"Test Samples: {len(X_test)}\n")
        f.write("-" * 45 + "\n")
        f.write(f"Accuracy:           {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
        f.write(f"Macro Precision:    {precision_macro:.4f}\n")
        f.write(f"Weighted Precision: {precision_weighted:.4f}\n")
        f.write(f"Macro Recall:       {recall_macro:.4f}\n")
        f.write(f"Weighted Recall:    {recall_weighted:.4f}\n")
        f.write(f"Macro F1-Score:     {f1_macro:.4f}\n")
        f.write(f"Weighted F1-Score:  {f1_weighted:.4f}\n")
    print(f"Metrics summary saved to: {metrics_path}")

    print("\nAll pipeline tasks completed successfully!")


if __name__ == "__main__":
    run_training_pipeline()
