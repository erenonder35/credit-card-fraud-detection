import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def load_data(file_path):
    """Loads the dataset and displays the class distribution."""
    print(f"Loading data from: {file_path}...")
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Splits the data and scales Time/Amount features using RobustScaler."""
    print("Starting data preprocessing...")
    
    # Feature (X) and Target (y) separation
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train-test split
    # stratify=y ensures the 0.17% fraud ratio is maintained in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # RobustScaler: Highly resistant to outliers compared to StandardScaler
    scaler = RobustScaler()
    
    # CRITICAL: Fit ONLY on the training set to prevent data leakage
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    
    print(f"Preprocessing completed. Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'creditcard.csv')
    
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)