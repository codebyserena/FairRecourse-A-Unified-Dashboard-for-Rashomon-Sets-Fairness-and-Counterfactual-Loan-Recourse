import pandas as pd
import numpy as np
import joblib
import os
from preprocessing import prepare_data

def identify_rashomon_set(metrics_path="artifacts/models_metrics.csv", tolerance=0.01):
    df_metrics = pd.read_csv(metrics_path)
    best_acc = df_metrics['accuracy'].max()
    rashomon_threshold = best_acc - tolerance
    
    rashomon_set = df_metrics[df_metrics['accuracy'] >= rashomon_threshold].copy()
    rashomon_set.to_csv("artifacts/rashomon_set.csv", index=False)
    
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Rashomon Set Size: {len(rashomon_set)} / {len(df_metrics)}")
    return rashomon_set

def analyze_disagreement():
    rashomon_set = pd.read_csv("artifacts/rashomon_set.csv")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = prepare_data()
    
    predictions = {}
    for _, row in rashomon_set.iterrows():
        model = joblib.load(row['model_path'])
        X_eval = X_test_scaled if row['use_scaled'] else X_test
        predictions[row['model_name']] = model.predict(X_eval)
        
    preds_df = pd.DataFrame(predictions, index=X_test.index)
    
    # Calculate agreement metrics
    approval_rates = (preds_df == 0).mean(axis=1)
    
    # Calculate instability score (distance from 0.5 approval rate)
    # The closer to 0.5, the more unstable
    instability_score = 1 - abs(approval_rates - 0.5) * 2
    
    # Get the 10 most unstable applicants
    most_unstable_idx = instability_score.sort_values(ascending=False).head(10).index
    
    results = []
    for idx in most_unstable_idx:
        preds = preds_df.loc[idx]
        approvals = (preds == 0).sum()
        rejects = (preds == 1).sum()
        total = len(preds)
        
        results.append({
            'applicant_index': idx,
            'approvals': approvals,
            'rejects': rejects,
            'agreement_pct': max(approvals, rejects) / total,
            'approval_rate': approvals / total,
            'instability_score': instability_score.loc[idx]
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv("artifacts/applicant_results.csv", index=False)
    print("Saved applicant disagreement analysis to artifacts/applicant_results.csv")
    return res_df

if __name__ == "__main__":
    identify_rashomon_set()
    analyze_disagreement()
