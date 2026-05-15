import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import joblib
from preprocessing import prepare_data, get_feature_metadata


def _sensitive_groups(X, sensitive_feature):
    values = X[sensitive_feature]
    if sensitive_feature == 'age':
        return pd.cut(
            values,
            bins=[0, 29, 39, 49, 59, float('inf')],
            labels=['under_30', '30_39', '40_49', '50_59', '60_plus'],
            include_lowest=True,
        )
    return values


def audit_fairness_all_models():
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = prepare_data()
    meta = get_feature_metadata()
    
    rashomon_set = pd.read_csv("artifacts/rashomon_set.csv")
    
    results = []
    
    print("Computing fairness metrics for all models in the Rashomon set...")
    for _, row in rashomon_set.iterrows():
        model = joblib.load(row['model_path'])
        X_eval = X_test_scaled if row['use_scaled'] else X_test
        y_pred = model.predict(X_eval)
        
        for sensitive_feature in meta['sensitive']:
            sensitive_groups = _sensitive_groups(X_test, sensitive_feature)
            dp_diff = demographic_parity_difference(
                y_test,
                y_pred,
                sensitive_features=sensitive_groups,
            )
            eo_diff = equalized_odds_difference(
                y_test,
                y_pred,
                sensitive_features=sensitive_groups,
            )

            results.append({
                'model_name': row['model_name'],
                'family': row['family'],
                'accuracy': row['accuracy'],
                'sensitive_feature': sensitive_feature,
                'demographic_parity_difference': dp_diff,
                'equalized_odds_difference': eo_diff,
                'total_unfairness': dp_diff + eo_diff,
            })
        
    fairness_df = pd.DataFrame(results)
    fairness_df.to_csv("artifacts/rashomon_fairness.csv", index=False)
    
    model_summary = (
        fairness_df
        .groupby(['model_name', 'family', 'accuracy'], as_index=False)['total_unfairness']
        .sum()
    )
    
    most_fair = model_summary.loc[model_summary['total_unfairness'].idxmin()]
    most_unfair = model_summary.loc[model_summary['total_unfairness'].idxmax()]
    
    summary = {
        'most_fair': most_fair.to_dict(),
        'most_unfair': most_unfair.to_dict()
    }
    
    pd.DataFrame([summary['most_fair'], summary['most_unfair']], index=['most_fair', 'most_unfair']).to_csv("artifacts/fairness_summary.csv")
    
    print("Fairness metrics saved to artifacts/rashomon_fairness.csv and artifacts/fairness_summary.csv")

if __name__ == "__main__":
    audit_fairness_all_models()
