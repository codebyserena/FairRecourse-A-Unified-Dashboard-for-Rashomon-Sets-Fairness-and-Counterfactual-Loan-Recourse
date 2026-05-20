# Understanding FairRecourse: Code Architecture & Logic

This document explains the underlying codebase of the FairRecourse project, detailing what each Python file does, how the data flows, and the libraries utilized.

## Directory Structure

- `src/`: Contains all the Python scripts that form the core backend pipeline.
- `app.py`: The Flask web server that serves the UI.
- `templates/`: HTML files for the dashboard pages.
- `static/css/`: Styling for the dashboard.
- `artifacts/`: Generated models and CSV data files used by the Flask app.

## The Data Pipeline (`src/`)

The core engine is executed sequentially. Each script reads data, performs computations, and saves the output to the `artifacts/` folder so the next script (and the dashboard) can use it.

### 1. `src/data.py`
**Purpose**: Ingests the raw UCI Default of Credit Card Clients dataset.
- Reads the `.xls` file from the `data/raw/` directory.
- Cleans column names (e.g., lowercasing, renaming `PAY_0` to `pay_1`, and `default payment next month` to `default`).
- Saves the cleaned dataset to `data/processed/credit_cleaned.csv`.

### 2. `src/preprocessing.py`
**Purpose**: Prepares the data for Machine Learning.
- Defines which features are continuous vs. categorical, and which are mutable (can be changed by the applicant) vs. immutable (e.g., sex, age, marriage status).
- Splits the data into Train (80%) and Test (20%) sets.
- Applies standard scaling (`StandardScaler`) to continuous features for models that require it (like Logistic Regression).
- Provides helper functions (`prepare_data`, `get_feature_metadata`) imported by other scripts.

### 3. `src/train_models.py`
**Purpose**: Hyperparameter sweeps to generate the model pool.
- Loops through distinct algorithmic families: `LogisticRegression`, `DecisionTree`, `RandomForest`, and `GradientBoosting`.
- Uses `ParameterSampler` from `scikit-learn` to randomly sample 10-15 hyperparameter configurations per family.
- Trains each configuration on the training set, evaluates accuracy on the test set, and saves the serialized model (`.pkl`) to `artifacts/models/`.
- Saves a master ledger of all trained models to `artifacts/models_metrics.csv`.

### 4. `src/rashomon.py`
**Purpose**: Identifies the Rashomon Set and predictive instability.
- Reads `models_metrics.csv` to find the absolute best accuracy.
- Filters models that are within 1% of this best accuracy to form the **Rashomon Set** (saved to `artifacts/rashomon_set.csv`).
- Evaluates every model in the Rashomon set against the test data to calculate "Instability Scores". Applicants who are approved by half the models and rejected by the other half have an instability score of 1.0 (saved to `artifacts/applicant_results.csv`).

### 5. `src/fairness.py`
**Purpose**: Audits Demographic Parity and Equalized Odds.
- Uses Microsoft's `Fairlearn` library.
- Iterates over every model in the Rashomon set and computes fairness metrics for every sensitive feature declared in `src/preprocessing.py` (`sex`, age bands, and `education`).
- Aggregates those gaps per model to isolate the "Most Fair Near-Optimal Model" and the "Most Unfair but Accurate Model" (saved to `artifacts/rashomon_fairness.csv` and `fairness_summary.csv`).

### 6. `src/recourse.py`
**Purpose**: Generates Counterfactual Explanations.
- Uses `DiCE-ML` (Diverse Counterfactual Explanations).
- Selects the most unstable applicant whose majority model decision is rejection or an exact approve/reject split.
- Generates required feature changes (e.g., "increase pay_amt by X") to flip their decision from Reject to Approve.
- Restricts mutable continuous recourse suggestions to the observed training-data range for each feature.
- **Multi-Model Recourse**: Performs this generation across 3 different models in the Rashomon set to compare how suggested feature changes differ depending on the model chosen (saved to `artifacts/multi_model_cfs.csv`).

### 7. `src/explainability.py`
**Purpose**: Generates Feature Importance.
- Uses `SHAP` (SHapley Additive exPlanations).
- Calculates the exact contribution of each feature to the specific rejection decision made by the absolute best model.
- Determines *why* the applicant was rejected before DiCE determines *how to fix it* (saved to `artifacts/shap_values.csv`).

## The Dashboard (`app.py` & `templates/`)

Once the pipeline finishes, the `artifacts/` folder is populated. 
`app.py` is a simple Flask server that reads these `.csv` files into Pandas DataFrames, passes them to the HTML templates, and renders interactive `Plotly.js` visualizations on the frontend.

The app now checks required artifacts before rendering each analysis route and returns a clear instruction to rerun `python3 run_pipeline.py` if generated files are missing.

## Development Utilities

- `run_pipeline.py`: Executes the full pipeline in the correct order using the active Python interpreter.
- `.gitignore`: Excludes virtual environments, bytecode caches, and common local-only files.
- `tests/`: Lightweight `unittest` checks for route rendering, artifact loading, feature metadata, and recourse helper behavior.
