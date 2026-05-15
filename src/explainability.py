import pandas as pd
import joblib
import shap
import json
import os
from preprocessing import prepare_data

def generate_shap_explanations():
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = prepare_data()
    
    rashomon_set = pd.read_csv("artifacts/rashomon_set.csv")
    app_results = pd.read_csv("artifacts/applicant_results.csv")
    
    # Take the best model (usually at index 0)
    best_model_row = rashomon_set.iloc[0]
    model = joblib.load(best_model_row['model_path'])
    
    # We will compute SHAP values for the first disputed applicant
    applicant_idx = app_results.iloc[0]['applicant_index']
    
    X_eval = X_test_scaled if best_model_row['use_scaled'] else X_test
    X_train_eval = X_train_scaled if best_model_row['use_scaled'] else X_train
    
    instance = X_eval.loc[[applicant_idx]]
    
    print(f"Generating SHAP values using {best_model_row['family']}...")
    try:
        if best_model_row['family'] in ['RandomForest', 'DecisionTree', 'GradientBoosting', 'XGBoost']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(instance)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0] # Get values for class 1 (default)
            elif len(shap_values.shape) == 3:
                shap_vals = shap_values[0, :, 1] # shape (1, features, 2)
            else:
                shap_vals = shap_values[0]
        else:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train_eval, 100))
            shap_values = explainer.shap_values(instance)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0] # Get values for class 1 (default)
            elif len(shap_values.shape) == 3:
                shap_vals = shap_values[0, :, 1] # shape (1, features, 2)
            else:
                shap_vals = shap_values[0]
                
        # Flatten to ensure 1D array
        import numpy as np
        shap_vals = np.array(shap_vals).flatten()
        
        # Create a dataframe for feature importance
        feature_names = instance.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_vals,
            'feature_value': instance.iloc[0].values
        })
        
        # Sort by absolute SHAP value
        importance_df['abs_shap'] = importance_df['shap_value'].abs()
        importance_df = importance_df.sort_values(by='abs_shap', ascending=False).drop(columns=['abs_shap'])
        
        importance_df.to_csv("artifacts/shap_values.csv", index=False)
        print("SHAP values saved to artifacts/shap_values.csv")
        
    except Exception as e:
        print(f"Failed to generate SHAP values: {e}")

if __name__ == "__main__":
    generate_shap_explanations()
