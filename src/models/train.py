"""
Train Model. 
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler 

def train_model(encoded_df: pd.DataFrame, target_col: str = 'CLASS'):
    """Train a logistic regression model and log metrics to MLflow.

    Args:
        encoded_df(pd.DataFrame, target_col:str):
    """

    X = encoded_df.drop(columns=['CLASS'], axis=1) # independent variable
    y = encoded_df['CLASS'] # dependent variable
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        C=545.5594781168514, 
        max_iter=1000, 
        penalty='l2', 
        solver='lbfgs', 
        random_state=42
        )
    
    with mlflow.start_run():
        # Train the model 
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        # Log params, metrics, and model to MLflow
        mlflow.log_param("C", 545.5594781168514)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("penalty", 'l2')  
        mlflow.log_param("solver", 'lbfgs') 
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.sklearn.log_model(model, "model")

        # Log dataset so it shows in Mlflow UI for reference
        train_ds = mlflow.data.from_pandas(encoded_df, source='training_data')
        mlflow.log_input(train_ds, context='training') 

        print(f'Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}')