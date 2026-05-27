# function to take the data and build the metadata table.
# Following fields are generated:

# {leveling-1}_P{person}_D{date}_type{STD or DD}_GS{#}_Panel-GS-25-{ID}_state{raw or flat}
# LevelingScore, Person, DateCollected, ImageType (STD or DD), GSCamera, PanelID, State (raw or flat), file path

import os
import pandas as pd
import re

def build_metadata_from_existing(data_types=("raw", "flat"), base_dir="../data", output_file="metadata.csv"):
    """
    Builds metadata.csv from current dataset structure.

    Extracts:
        - LevelingScore (from folder)
        - PanelID (from filename: GS-25-###)
        - State (raw or flat)
        - FilePath

    Other fields are set to None.
    """

    records = []

    for data_type in data_types:  # e.g. "raw", "flat"
        data_dir = os.path.join(base_dir, data_type)

        for label_folder in sorted(os.listdir(data_dir), key=lambda x: int(x)):
            folder_path = os.path.join(data_dir, label_folder)

            if not os.path.isdir(folder_path):
                continue

            level = int(label_folder)

            for filename in os.listdir(folder_path):
                if not filename.lower().endswith(".png"):
                    continue

                filepath = os.path.join(folder_path, filename)

                # ---- Extract PanelID ----
                panel_match = re.search(r"GS-25-(\d+)", filename)
                panel_id = f"GS-25-{panel_match.group(1)}" if panel_match else None

                # ---- Extract State (fallback to folder name if needed) ----
                state_match = re.search(r"(raw|flat)", filename.lower())
                state = state_match.group(1) if state_match else data_type

                # ---- Build record ----
                record = {
                    "LevelingScore": level,
                    "Person": None,
                    "DateCollected": None,
                    "ImageType": None,
                    "GSCamera": None,
                    "PanelID": panel_id,
                    "State": state,
                    "FilePath": filepath
                }

                records.append(record)

    df = pd.DataFrame(records)

    output_path = os.path.join(base_dir, output_file)
    df.to_csv(output_path, index=False)

    print(f"Saved metadata to {output_path}")
    print(f"Total records: {len(df)}")

    return df
