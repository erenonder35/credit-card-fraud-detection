from data_preprocessing import load_data, preprocess_data
import os
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def train_models():
    # 1. Load and preprocess data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'creditcard.csv')
    
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\n--- Model Training Initiated ---")

    # APPROACH 1: Random Forest with Class Weights
    # class_weight='balanced': Penalizes misclassification of the minority class (Fraud)
    print("\n1. Training Random Forest (with Class Weights)... This may take 1-2 minutes.")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("Random Forest training completed.")

    # APPROACH 2: XGBoost with SMOTE (Synthetic Data Generation)
    # Applied ONLY to the training data to prevent leakage!
    print("\n2. Applying SMOTE to the training set...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"New training set shape after SMOTE: {X_train_smote.shape}")

    print("Training XGBoost model (with SMOTE data)...")
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    xgb_model.fit(X_train_smote, y_train_smote)
    print("XGBoost training completed.")

    return rf_model, xgb_model, X_test, y_test

if __name__ == "__main__":
    rf_model, xgb_model, X_test, y_test = train_models()