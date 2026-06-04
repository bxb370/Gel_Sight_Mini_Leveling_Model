"""This file takes the human rating data and evaluates the MAE of the human baseline model"""

import pandas as pd
import numpy as np

# Read the CSV with its 2-row header (name + trial)
raw = pd.read_csv("analysis/HumanRatings.csv", header=[1, 2])

# Build unique column names by forward-filling the rater name for Trial 2 columns.
rater_names = []
current_rater = ""
for top, _ in raw.columns:
    top_str = str(top)
    if pd.notna(top) and "Unnamed" not in top_str:
        current_rater = top_str
    rater_names.append(current_rater)

raw.columns = [f"{rater}_{trial}" if "Trial" in str(trial) else str(trial)
               for rater, (_, trial) in zip(rater_names, raw.columns)]

# Keep only the 6 trial/rating columns
rating_cols = [
    "Jennifer_Trial 1",
    "Jennifer_Trial 2",
    "Nicole_Trial 1",
    "Nicole_Trial 2",
    "Chase_Trial 1",
    "Chase_Trial 2",
]

# Clean + standardize
df = raw[rating_cols].apply(pd.to_numeric, errors="coerce")
df.columns = [f"rating_{i}" for i in range(1, 7)]

# Add average column
df["rating_avg"] = df.mean(axis=1)

# Optional: remove rows where all 6 ratings are missing
df = df.dropna(subset=[f"rating_{i}" for i in range(1, 7)], how="all")

print(df.head())

#calculate the mean absolute error of the human baseline model

from sklearn.metrics import mean_absolute_error
import numpy as np

rating_cols = ["rating_1", "rating_2", "rating_3",
               "rating_4", "rating_5", "rating_6"]

# Flatten all human ratings into one vector
y_true = df[rating_cols].values.flatten()

# Repeat each row's average 6 times (to match each rating)
y_pred = np.repeat(df["rating_avg"].values, len(rating_cols))

# Compute MAE
human_mae = mean_absolute_error(y_true, y_pred)

print("Human MAE:", human_mae)
