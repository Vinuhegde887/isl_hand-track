import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

CSV_PATH = r"d:\isl updated\keypoints_data.csv"
DATA_DIR = r"d:\isl updated\data"

def analyze_dataset():
    # 1. Check raw files
    print("--- Raw Data Stats ---")
    if os.path.exists(DATA_DIR):
        classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        for c in classes:
            files = os.listdir(os.path.join(DATA_DIR, c))
            video_files = [f for f in files if f.endswith(('.mp4', '.avi'))]
            print(f"Class '{c}': {len(video_files)} videos")
    else:
        print("Data directory not found!")

    # 2. Check CSV
    print("\n--- CSV Data Stats ---")
    if not os.path.exists(CSV_PATH):
        print("CSV file not found!")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Class distribution
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    
    # Null checks
    print("\nMissing Values:")
    print(df.isnull().sum().sum())
    
    # Check for all-zero samples (failed detection)
    features = df.iloc[:, :-1].values
    zero_rows = np.all(features == 0, axis=1)
    print(f"\nAll-zero rows (no detection): {np.sum(zero_rows)}")
    
    if np.sum(zero_rows) > 0:
        print("Classes with all-zero rows:")
        print(df.loc[zero_rows, 'label'].value_counts())

    # Check for near-zero detection (mostly padded)
    # Each frame has 126 features. 60 frames -> 7560 columns.
    # Non-zero count per row
    non_zero_counts = np.count_nonzero(features, axis=1)
    avg_non_zero = np.mean(non_zero_counts)
    print(f"\nAvg avg non-zero features per sample: {avg_non_zero:.2f} / {features.shape[1]}")
    
    # Plot length of non-zero content
    plt.figure(figsize=(10, 5))
    plt.hist(non_zero_counts, bins=20)
    plt.title("Distribution of Non-Zero Values per Sample")
    plt.xlabel("Number of Non-Zero Values")
    plt.ylabel("Count")
    plt.savefig("data_quality_hist.png")
    print("\nSaved histogram to data_quality_hist.png")

if __name__ == "__main__":
    analyze_dataset()
