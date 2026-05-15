import pandas as pd
import os

def load_and_clean_data(raw_data_path="data/raw/default of credit card clients.xls", processed_data_path="data/processed/credit_cleaned.csv"):
    """Loads the raw excel file, cleans column names, and saves to CSV."""
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"{raw_data_path} not found.")
        
    print(f"Loading raw data from {raw_data_path}...")
    # The dataset has an extra header row, so we skip the first row (header=1)
    df = pd.read_excel(raw_data_path, header=1)
    
    # Clean column names
    df = df.rename(columns={'default payment next month': 'default', 'PAY_0': 'PAY_1'})
    df.columns = [col.lower() for col in df.columns]
    
    # Save processed data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    print(f"Saved processed data to {processed_data_path}")
    return df

if __name__ == "__main__":
    load_and_clean_data()
