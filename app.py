from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

REQUIRED_ARTIFACTS = {
    'metrics': 'artifacts/models_metrics.csv',
    'rashomon': 'artifacts/rashomon_set.csv',
    'applicants': 'artifacts/applicant_results.csv',
    'query_inst': 'artifacts/query_instance.csv',
}

OPTIONAL_ARTIFACTS = {
    'cfs': 'artifacts/multi_model_cfs.csv',
    'fairness': 'artifacts/rashomon_fairness.csv',
    'fairness_summary': 'artifacts/fairness_summary.csv',
    'shap': 'artifacts/shap_values.csv',
}


def _read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else None


def _artifact_error(message):
    return (
        f"{message} Please run the pipeline with: python3 run_pipeline.py",
        503,
    )


def load_data():
    try:
        missing = [path for path in REQUIRED_ARTIFACTS.values() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"Missing artifacts: {', '.join(missing)}")

        metrics = pd.read_csv(REQUIRED_ARTIFACTS['metrics'])
        rashomon = pd.read_csv(REQUIRED_ARTIFACTS['rashomon'])
        applicants = pd.read_csv(REQUIRED_ARTIFACTS['applicants'])
        query_inst = pd.read_csv(REQUIRED_ARTIFACTS['query_inst'])

        cfs = _read_csv(OPTIONAL_ARTIFACTS['cfs'])
        fairness = _read_csv(OPTIONAL_ARTIFACTS['fairness'])
        fairness_summary = _read_csv(OPTIONAL_ARTIFACTS['fairness_summary'], index_col=0)
        shap_vals = _read_csv(OPTIONAL_ARTIFACTS['shap'])

        if fairness is not None and 'total_unfairness' not in fairness.columns:
            fairness['total_unfairness'] = (
                fairness['demographic_parity_difference']
                + fairness['equalized_odds_difference']
            )
        
        return {
            'metrics': metrics,
            'rashomon': rashomon,
            'applicants': applicants,
            'cfs': cfs,
            'query_inst': query_inst,
            'fairness': fairness,
            'fairness_summary': fairness_summary,
            'shap': shap_vals
        }
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None

@app.route('/')
def overview():
    data = load_data()
    if data is None:
        return _artifact_error("Data is not ready yet.")
        
    best_acc = data['metrics']['accuracy'].max()
    avg_acc = data['metrics']['accuracy'].mean()
    num_models = len(data['metrics'])
    rashomon_size = len(data['rashomon'])
    rashomon_share = rashomon_size / num_models if num_models else 0
    best_model = data['metrics'].sort_values('accuracy', ascending=False).iloc[0].to_dict()
    family_stats = (
        data['metrics']
        .groupby('family', as_index=False)
        .agg(count=('model_name', 'count'), best_accuracy=('accuracy', 'max'), mean_accuracy=('accuracy', 'mean'))
        .sort_values('best_accuracy', ascending=False)
        .to_dict(orient='records')
    )
    
    metrics_data = data['metrics'].to_dict(orient='records')
    
    return render_template('index.html', 
                          best_acc=f"{best_acc:.2%}", 
                          avg_acc=f"{avg_acc:.2%}",
                          num_models=num_models, 
                          rashomon_size=rashomon_size,
                          rashomon_share=f"{rashomon_share:.0%}",
                          best_model=best_model,
                          family_stats=family_stats,
                          metrics_data=metrics_data)

@app.route('/rashomon')
def rashomon_view():
    data = load_data()
    if data is None:
        return _artifact_error("Rashomon artifacts are not ready yet.")

    applicants = data['applicants'].to_dict(orient='records')
    rashomon_data = data['rashomon'].to_dict(orient='records')
    top_applicant = applicants[0] if applicants else {}
    model_count = len(data['rashomon'])
    return render_template(
        'rashomon.html',
        applicants=applicants,
        rashomon_data=rashomon_data,
        top_applicant=top_applicant,
        model_count=model_count,
    )

@app.route('/recourse')
def recourse_view():
    data = load_data()
    if data is None:
        return _artifact_error("Recourse artifacts are not ready yet.")

    cfs = data['cfs'].to_dict(orient='records') if data['cfs'] is not None else []
    query_records = data['query_inst'].to_dict(orient='records')
    query = query_records[0] if query_records else {}
    shap_vals = data['shap'].head(10).to_dict(orient='records') if data['shap'] is not None else []
    
    # We want to group CFs by model_name for easy display
    grouped_cfs = {}
    for cf in cfs:
        model_name = cf.get('model_name', 'Unknown')
        if model_name not in grouped_cfs:
            grouped_cfs[model_name] = []
        grouped_cfs[model_name].append(cf)

    changed_counts = {}
    for model_name, model_cfs in grouped_cfs.items():
        changed_counts[model_name] = sum(
            1
            for key, value in model_cfs[0].items()
            if key not in ['default', 'model_name', 'family'] and query.get(key) != value
        )
        
    return render_template(
        'recourse.html',
        grouped_cfs=grouped_cfs,
        query=query,
        shap=shap_vals,
        changed_counts=changed_counts,
    )

@app.route('/fairness')
def fairness_view():
    data = load_data()
    if data is None:
        return _artifact_error("Fairness artifacts are not ready yet.")

    fairness = data['fairness'].to_dict(orient='records') if data['fairness'] is not None else []
    summary = data['fairness_summary'].to_dict(orient='index') if data['fairness_summary'] is not None else {}
    feature_stats = []
    if data['fairness'] is not None:
        feature_stats = (
            data['fairness']
            .groupby('sensitive_feature', as_index=False)
            .agg(
                min_gap=('total_unfairness', 'min'),
                max_gap=('total_unfairness', 'max'),
                mean_gap=('total_unfairness', 'mean'),
            )
            .sort_values('mean_gap', ascending=False)
            .to_dict(orient='records')
        )
    
    return render_template('fairness.html', fairness=fairness, summary=summary, feature_stats=feature_stats)

@app.route('/summary')
def summary_view():
    return render_template('summary.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
