import pandas as pd
import numpy as np
import os

CSV_PATH = r"d:\isl updated\keypoints_data.csv"
CLEANED_CSV_PATH = r"d:\isl updated\keypoints_data.csv" # Overwrite or create new

def clean_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    original_count = len(df)
    
    # 1. Remove rows with any NaN values
    df_clean = df.dropna()
    nan_removed_count = original_count - len(df_clean)
    if nan_removed_count > 0:
        print(f"Removed {nan_removed_count} rows containing NaNs.")
    
    # 2. Remove rows where all feature columns (everything except label) are zero
    # Assuming label is the last column
    features = df_clean.iloc[:, :-1]
    
    # Check if all features are 0
    # We use a small threshold for float comparison just in case, but usually it's exactly 0.0
    zero_mask = (features.abs().sum(axis=1) == 0)
    df_clean = df_clean[~zero_mask]
    
    zero_removed_count = np.sum(zero_mask)
    if zero_removed_count > 0:
        print(f"Removed {zero_removed_count} rows containing only zeros (no detection).")

    final_count = len(df_clean)
    print(f"Original samples: {original_count}")
    print(f"Cleaned samples:  {final_count}")
    print(f"Removed total:    {original_count - final_count}")

    if original_count != final_count:
        print(f"Saving cleaned dataset to {CLEANED_CSV_PATH}...")
        df_clean.to_csv(CLEANED_CSV_PATH, index=False)
        print("Done.")
    else:
        print("No changes needed. Dataset is already clean.")

if __name__ == "__main__":
    clean_dataset()
