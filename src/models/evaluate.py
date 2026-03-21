"""
Evaluate Model. 
"""

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import seaborn as sns # seaborn plotting library
import matplotlib.pyplot as plt # matplot library

def evaluate_model(model, X_test_scaled, y_test): 
    """ Evalaute logistic regression model on test data. 
    Args: 
        model: Trained logistic regression model.
        X_test_scaled: Scaled test features.
        y_test: Test labels.
    """
    preds= model.predict(X_test_scaled)

    cls_report = classification_report(y_test, preds)

    print("Classification Report:") 
    print(cls_report)
    
    print("Confusion Matrix:")

    # plot confusion matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()  

    # Plot Precision-Recall curve
    y_scores = model.predict_proba(X_test_scaled)[:, 1]
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision) 
    plt.plot(recall, precision, label=f"(AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", y=1.04)
    plt.legend(loc="lower left")
    plt.show()