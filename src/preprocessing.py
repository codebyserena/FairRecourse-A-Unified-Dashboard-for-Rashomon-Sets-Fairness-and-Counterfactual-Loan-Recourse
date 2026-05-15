import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Feature metadata
SENSITIVE_FEATURES = ['sex', 'age', 'education']
IMMUTABLE_FEATURES = ['sex', 'age', 'education', 'marriage']
MUTABLE_FEATURES = [
    'limit_bal', 'pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
    'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
    'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6'
]
TARGET = 'default'
CATEGORICAL_FEATURES = ['sex', 'education', 'marriage', 'pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']

def get_feature_metadata():
    return {
        'sensitive': SENSITIVE_FEATURES,
        'immutable': IMMUTABLE_FEATURES,
        'mutable': MUTABLE_FEATURES,
        'categorical': CATEGORICAL_FEATURES,
        'target': TARGET
    }

def prepare_data(data_path="data/processed/credit_cleaned.csv", save_scaler=True):
    df = pd.read_csv(data_path)
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    # We will let DiCE and XGBoost handle categorical features directly where possible,
    # but for logistic regression we might need one-hot encoding.
    # For MVP, we will train tree-based models and logreg on un-one-hot-encoded data first to see,
    # or just use standard scaler.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    num_features = [col for col in X.columns if col not in CATEGORICAL_FEATURES]
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
    X_test_scaled[num_features] = scaler.transform(X_test[num_features])
    
    if save_scaler:
        os.makedirs("artifacts/models", exist_ok=True)
        joblib.dump(scaler, "artifacts/models/scaler.pkl")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
