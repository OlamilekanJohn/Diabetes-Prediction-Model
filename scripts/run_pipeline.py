"""
Runs sepquentially: load data, validate, preprocess, train, tune and evaluate model
"""

import os
import sys
import time 
import argparse
import pandas as pd 
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_recall_curve, 
    auc, 
    confusion_matrix
)
import seaborn as sns # seaborn plotting library
import matplotlib.pyplot as plt # matplot library


# EsSENTIAL: Add project root to sys.path to import modules from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# local modules - core pipeline components 
from src.data.load_data import load_data        # load raw data from CSV
from src.data.preprocess import preprocess_data             # data cleaning and preprocessing
from src.features.features import encode_features    # feature engineering and encoding
from src.models.train import train_model        # train model and log to MLflow
from src.models.tune import tune_model      # hyperparameter tuning with Optuna 
from src.models.evaluate import evaluate_model      # evaluate model on test set
from src.utils.validate_data_3 import validate_data      # validate data before processing

def main (args):
    """Main function to run the ML pipeline sequentially."""
    
    # Configure MLflow to use local file-based tracking 

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment) # creates experiment if it doesn't exist

    with mlflow.start_run(): 
        # == Log hyperparameters and configurations ==

        mlflow.log_param("model", "LogisticRegression")  # Model type for comparison
        mlflow.log_param("threshold", args.threshold)    # Classification threshold
        mlflow.log_param("test_size", args.test_size)    # Train/test split ratio

    # == Stage 1: Load raw data and validate it ==
        print ("Loading raw data...")
        df = load_data(args.input) # Load raw data from CSV with error handling
        print (f" Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        
        # === Stage 2: Data preprocessing ==
        print ("Preprocessing data...")
        df = preprocess_data(df) # Preprocess data (cleaning, imputation, etc.)

        # save processed dataset for reproducibility and debugging 
        processed_path = os.path.join(project_root, "data", "processed", "diabetes_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print (f"Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # === Stage 3: Feature engineering and encoding ===
        print (" Building features and enconding...")
        target = args.target

        if target not in df.columns: 
            raise ValueError (f" Target column '{target}' not found in data") 
        
        df_enc = encode_features(df, target_col=target) # Feature engineering and encoding

        # Important: Convert boolean columns to integers for MLflow logging and model compatibility
        for col in df_enc.select_dtypes(include=['bool']).columns:
            df_enc[col] = df_enc[col].astype(int)
        print (f" Feature engineering completed: {df_enc.shape[1]} features") 

        # === Critical: Save feature metadata for consistency and debugging ===
        # emsures serving pipeline used same features as training in same order 
        import json, joblib 
        artifact_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        # Get feature columns (exclude target) 
        feature_cols = list(df_enc.drop(columns=[target]).columns)

        # save locally for development and debugging
        with open(os.path.join(artifact_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f) 
        
        # Log to MLflow for production serving 
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt") 

        # Essential: Save preprocessing artifacts for serving pipeline 
        # These artifacts ensure training and serving use identical transformations 
        preprocessing_artifacts = {
            "feature_columns": feature_cols, # Exact feature order 
            "target": target                # Target column name 
        }
        joblib.dump(preprocessing_artifacts, os.path.join(artifact_dir, "preprocessing.pk1"))
        mlflow.log_artifact(os.path.join(artifact_dir, "preprocessing.pk1"))
        print (f" Saved {len(feature_cols)} feature columns for serving consistency")

        # == Critical: Validate raw data before processing ==
        print ("Validate data quality and integrity...")
        df_enc, is_valid, failed = validate_data(df_enc) 
        mlflow.log_metric("data_quality_pass", int(is_valid)) # Track data quality over time

        if not is_valid:
            # log validation failures for debugging and monitoring
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expections.json")
            raise ValueError (f" Data validation check failed. Issues: {failed}")
        else: 
            print (" Data validation passed. Logged to MLflow.") 
        

        # === stage 4: Train/Test Split ===
        print (" Splitting data...")
        X = df_enc.drop(columns=[target], axis=1) # independent variable 
        y = df_enc[target] # dependent variable

        # Stratified split to maintain class distribution in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size, 
            random_state=42, 
            stratify=y
        )
        print (f" Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        # === Stage 5: Train model with optimized Hyperparameters ===
        print (" Training Logistic Regression model...") 

        model = LogisticRegression(
            C=545.5594781168514, 
            max_iter=1000, 
            penalty='l2', 
            solver='lbfgs', 
            random_state=42
        )

        # === Train model and Track Training Time ===
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time) # Track training performance 
        print (f" Model trained in {train_time:.2f} seconds")

        # === Stage 6: Evaluate model on test set ===
        print (" Evaluating model...")

        # Generate predictions and track inference time 
        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1] # Get probabilities for positive class

        y_pred = (proba >= args.threshold).astype(int) # Apply classification threshold
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time) # Track inference performance
        

        # === Log Evaluation Metrics to MLflow ===

        classification = classification_report(y_test, y_pred)

        mlflow.log_text (classification, artifact_file="classification_report.txt") # Log full classification report as text

        print (f"Model Performance:")
        print (classification)

        # === Stage 7: Log confusion matrix and plot ===
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_text(
            json.dumps (cm.tolist(), indent=2), artifact_file="confusion_matrix.json") # Log confusion matrix as list for visualization in MLflow UI
        print ("Confusion Matrix:")
        print (cm)

        # === Plot confusion Matrix ===

        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        for i in range(len(cm)):
            for j in range(len(cm[0])):
                plt.text(j, i, cm[i, j], ha="center", va="center")

        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()




        # === Stage 8: Log Precision-Recall curve and AUC ===
        precision, recall, pr_thresholds = precision_recall_curve(y_test, proba)
        pr_auc = auc(recall, precision)
        mlflow.log_metric("pr_auc", pr_auc) # Log AUC for precision-recall curve
        print (f"Precision-Recall AUC: {pr_auc:.4f}")   

        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (AUC={pr_auc:.3f})")

        plt.tight_layout()
        plt.savefig("pr_curve.png")
        mlflow.log_artifact("pr_curve.png")
        plt.close()

        # === Stage 9: Save trained model to MLflow ===

        print ("Saving model to MLflow...")

        mlflow.sklearn.log_model(
            model, 
            artifact_path="model"
        ) # Log model to MLflow

        print ("ML pipeline completed successfully. All artifacts and metrics logged to MLflow.")

        # === final Performance Summary ===

        print (f"\n Performance Summary:")
        print (f" Training Time: {train_time:.2f} seconds")
        print (f" Inference Time: {pred_time:.2f} seconds")
        print (f" Samples per second: {len(X_test)/pred_time:.0f}")

        print (f" Precision-Recall AUC: {pr_auc:.4f}")
        print (f" Classification Report:\n{classification}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run ML pipeline for diabetes prediction")
    p.add_argument("--input", type=str, required=True, 
                   help="path to CSV (e.g.,data/raw/diabetes.csv)")
    p.add_argument("--target", type=str, default="CLASS")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Diabetes_Prediction")
    p.add_argument("--mlflow_uri", type=str, default=None, 
                   help="override MLflow tracking URI, else uses project_root/mlruns")
    
    args = p.parse_args()
    main(args)


