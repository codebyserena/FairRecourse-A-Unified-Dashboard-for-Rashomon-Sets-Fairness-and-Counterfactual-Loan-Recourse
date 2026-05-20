# FairRecourse

FairRecourse is a credit-risk machine learning dashboard built to compare model performance, fairness, explanations, and counterfactual recourse in one place. It trains multiple classification models, finds near-best models, and shows how their decisions can differ for the same applicant.

**Built by:** Serena Mendanha

## Features

- Trains Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting models.
- Compares model accuracy and identifies near-best models.
- Shows applicants where similar models disagree on approval or rejection.
- Audits fairness across sex, education, and age bands using Fairlearn.
- Explains model decisions with SHAP feature importance.
- Generates counterfactual recourse suggestions with DiCE.
- Presents results in a Flask and Plotly dashboard.

## Setup & Installation

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### 1. Execute the Pipeline
Before running the dashboard, run the data pipeline to clean the data, train models, calculate fairness metrics, generate explanations, and create recourse outputs.

Run this command from the root of the project:
```bash
python3 run_pipeline.py
```
Training the models may take a few minutes depending on your machine.

### 2. Start the Dashboard
Once the pipeline finishes generating the artifacts, you can start the Flask dashboard:

```bash
python3 app.py
```

By default, the app tries port `5001`. If that port is busy, it automatically starts on the next available port and prints the URL in the terminal. To request a different starting port:

```bash
PORT=5050 python3 app.py
```

### 3. View the App
Open your web browser and navigate to:
`http://127.0.0.1:5001`

If the terminal says it started on a different port, use that printed URL instead.

## Project Structure
- `src/`: The Python pipeline scripts.
- `app.py`: The Flask backend.
- `templates/` & `static/`: The frontend UI (HTML/CSS).
- `artifacts/`: Generated models, metrics, SHAP values, and counterfactuals.
- `data/`: The raw and processed UCI Credit dataset.
- `tests/`: Basic tests for the dashboard and pipeline helpers.

## Development Checks

Run the lightweight tests with:

```bash
python3 -m unittest discover -s tests
```

The tests assume the artifact CSVs have already been generated.
