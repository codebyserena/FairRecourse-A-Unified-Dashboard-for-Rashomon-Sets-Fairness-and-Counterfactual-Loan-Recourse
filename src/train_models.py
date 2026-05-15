import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterSampler
from preprocessing import prepare_data

def train_and_evaluate():
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = prepare_data()
    
    # Define model families and their hyperparameter spaces
    model_spaces = {
        'LogisticRegression': {
            'class': LogisticRegression,
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear'],
                'max_iter': [1000]
            },
            'n_iter': 10,
            'use_scaled': True
        },
        'DecisionTree': {
            'class': DecisionTreeClassifier,
            'params': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'n_iter': 10,
            'use_scaled': False
        },
        'RandomForest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'n_iter': 15,
            'use_scaled': False
        },
        'GradientBoosting': {
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5]
            },
            'n_iter': 5,
            'use_scaled': False
        }
    }
    
    os.makedirs('artifacts/models', exist_ok=True)
    results = []
    
    for family, info in model_spaces.items():
        print(f"Training models for {family}...")
        param_list = list(ParameterSampler(info['params'], n_iter=info['n_iter'], random_state=42))
        
        for i, params in enumerate(param_list):
            model_name = f"{family}_{i+1}"
            model = info['class'](**params)
            
            # Use scaled data for LogReg
            X_tr = X_train_scaled if info['use_scaled'] else X_train
            X_te = X_test_scaled if info['use_scaled'] else X_test
            
            model.fit(X_tr, y_train)
            
            y_pred = model.predict(X_te)
            acc = accuracy_score(y_test, y_pred)
            
            model_path = f"artifacts/models/{model_name}.pkl"
            joblib.dump(model, model_path)
            
            results.append({
                'model_name': model_name,
                'family': family,
                'accuracy': acc,
                'params': str(params),
                'model_path': model_path,
                'use_scaled': info['use_scaled']
            })
            print(f"  {model_name}: Acc = {acc:.4f}")
            
    results_df = pd.DataFrame(results)
    results_df.to_csv("artifacts/models_metrics.csv", index=False)
    print("Training complete. Metrics saved to artifacts/models_metrics.csv")

if __name__ == "__main__":
    train_and_evaluate()
