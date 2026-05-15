# FairRecourse

A credit decision research dashboard that demonstrates how equally accurate machine learning models (the Rashomon set) can make conflicting decisions for individual applicants, leading to varying counterfactual recourse and fairness implications.

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
Before running the dashboard, you must run the data pipeline to train the models, find the Rashomon set, generate counterfactuals, compute SHAP values, and audit fairness.

Run this command from the root of the project:
```bash
python3 run_pipeline.py
```
*(Note: Training the models may take 2-5 minutes depending on your machine).*

### 2. Start the Dashboard
Once the pipeline finishes generating the artifacts, you can start the Flask dashboard:

```bash
python3 app.py
```

### 3. View the App
Open your web browser and navigate to:
`http://127.0.0.1:5000`

## Project Structure
- `src/`: The Python pipeline scripts.
- `app.py`: The Flask backend.
- `templates/` & `static/`: The frontend UI (HTML/CSS).
- `artifacts/`: Generated models, metrics, SHAP values, and counterfactuals.
- `data/`: The raw and processed UCI Credit dataset.

## Development Checks

Run the lightweight tests with:

```bash
python3 -m unittest discover -s tests
```

The tests assume the artifact CSVs have already been generated.
