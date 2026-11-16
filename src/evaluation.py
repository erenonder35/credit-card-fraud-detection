from model_training import train_models
from sklearn.metrics import classification_report, precision_recall_curve, auc

def evaluate_models():
    # Retrieve trained models and test data
    rf_model, xgb_model, X_test, y_test = train_models()
    
    print("\n" + "="*50)
    print("1. RANDOM FOREST (Class Weights) TEST RESULTS")
    print("="*50)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, rf_preds, target_names=['Normal', 'Fraud']))
    precision, recall, _ = precision_recall_curve(y_test, rf_probs)
    print(f">> PR-AUC Score: {auc(recall, precision):.4f}\n")

    print("="*50)
    print("2. XGBOOST (SMOTE) TEST RESULTS")
    print("="*50)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, xgb_preds, target_names=['Normal', 'Fraud']))
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, xgb_probs)
    print(f">> PR-AUC Score: {auc(recall_xgb, precision_xgb):.4f}\n")

if __name__ == "__main__":
    evaluate_models()