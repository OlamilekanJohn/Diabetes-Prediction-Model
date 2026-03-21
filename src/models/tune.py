"""
Tune Model. 
"""

from fastapi import params

import optuna
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

def tune_model(X,y): 
    """
    Tunes an logistic regression model using Optuna. 

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """ 
    
    def objective(trial):
        params = {
            'C': trial.suggest_loguniform('C', 1e-5, 1e5),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
            'random_state': 42
        }
        model = LogisticRegression(**params) 
        scores = cross_val_score(model, X, y, cv=5, scoring='recall')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best params: ", study.best_params)
    return study.best_params