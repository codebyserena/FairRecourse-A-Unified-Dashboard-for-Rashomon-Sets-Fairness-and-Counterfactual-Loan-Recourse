import dice_ml
from dice_ml import Data, Model, Dice
import pandas as pd
import joblib
from preprocessing import prepare_data, get_feature_metadata


def _select_query_index(app_results):
    """Prefer a highly unstable applicant whose majority decision is rejection."""
    rejected_or_split = app_results[app_results['approval_rate'] <= 0.5]
    candidates = rejected_or_split if not rejected_or_split.empty else app_results
    return int(candidates.sort_values('instability_score', ascending=False).iloc[0]['applicant_index'])


def _permitted_ranges(X_train, features):
    return {
        feature: [
            float(X_train[feature].min()),
            float(X_train[feature].max()),
        ]
        for feature in features
    }

def generate_multi_model_recourse():
    X_train, X_test, y_train, y_test, _, _ = prepare_data()
    meta = get_feature_metadata()
    
    continuous_features = [col for col in X_train.columns if col not in meta['categorical']]
    
    train_df = X_train.copy()
    train_df[meta['target']] = y_train
    
    d = Data(dataframe=train_df, continuous_features=continuous_features, outcome_name=meta['target'])
    
    rashomon_set = pd.read_csv("artifacts/rashomon_set.csv")
    
    # Select 3 different models from the Rashomon set if possible
    # E.g., one RandomForest, one LogReg, one DecisionTree
    selected_models = []
    for family in ['RandomForest', 'LogisticRegression', 'DecisionTree', 'GradientBoosting']:
        models_in_family = rashomon_set[rashomon_set['family'] == family]
        if not models_in_family.empty:
            selected_models.append(models_in_family.iloc[0])
            if len(selected_models) == 3:
                break
                
    # If we don't have 3 families, just pick top 3 models
    if len(selected_models) < 3:
        selected_models = [row for _, row in rashomon_set.head(3).iterrows()]
        
    app_results = pd.read_csv("artifacts/applicant_results.csv")
    rejected_idx = _select_query_index(app_results)
    
    query_instance = X_test.loc[[rejected_idx]]
    query_instance.to_csv("artifacts/query_instance.csv", index=False)
    
    continuous_mutable = [f for f in meta['mutable'] if f not in meta['categorical']]
    permitted_range = _permitted_ranges(X_train, continuous_mutable)
    
    all_cfs = []
    
    for row in selected_models:
        model = joblib.load(row['model_path'])
        m = Model(model=model, backend="sklearn")
        exp = Dice(d, m, method="random")
        
        try:
            dice_exp = exp.generate_counterfactuals(
                query_instance, 
                total_CFs=1, # Just 1 CF per model for clear comparison
                desired_class=0,
                features_to_vary=continuous_mutable,
                permitted_range=permitted_range
            )
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            cf_dict = cf_df.iloc[0].to_dict()
            cf_dict['model_name'] = row['model_name']
            cf_dict['family'] = row['family']
            all_cfs.append(cf_dict)
            print(f"Generated counterfactual for {row['model_name']}")
        except Exception as e:
            print(f"Failed to generate counterfactuals for {row['model_name']}: {e}")

    if all_cfs:
        pd.DataFrame(all_cfs).to_csv("artifacts/multi_model_cfs.csv", index=False)
        print("Generated multi-model counterfactuals saved to artifacts/multi_model_cfs.csv")

if __name__ == "__main__":
    generate_multi_model_recourse()
