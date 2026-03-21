
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import optuna

print ("=== Phase 2: Modelling with Logistic Model ===")

df = pd.read_csv("/Users/macbook/Desktop/Python files/First deployed Project/data/processed/diabetes_processed.csv")

# target must be numeric 0/1
if df["CLASS"].dtype == "object":
    df["CLASS"] = df["CLASS"].str.strip().map({"N": 0, "Y":1})

assert df["CLASS"].isna().sum() == 0, "class has NANs"
assert set(df["CLASS"].unique()) <= {0,1}, "class not 0/1"

X = df.drop(columns=["CLASS"])
y = df["CLASS"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2, stratify=y, random_state=42
)

THRESHOLD = 0.4

def objective(trial): 
    params = {
        'C': trial.suggest_loguniform('C', 1e-5, 1e5),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
            'random_state': 42
    }
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    from sklearn.metrics import (classification_report, precision_recall_curve, auc, confusion_matrix)
    return classification_report(y_test, y_pred,pos_label=1) 
    