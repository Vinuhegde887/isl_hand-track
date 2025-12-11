import pandas as pd
import os

CSV_PATH = "keypoints_sliding_data.csv"
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    y = df.iloc[:, -1].unique()
    print("Classes in CSV:", sorted(y))
else:
    print("CSV not found")
