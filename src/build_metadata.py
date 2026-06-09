# function to take the data and build the metadata table.
# Following fields are generated:

# {leveling-1}_P{person}_D{date}_type{STD or DD}_GS{#}_Panel-GS-25-{ID}_state{raw or flat}
# LevelingScore, Person, DateCollected, ImageType (STD or DD), GSCamera, PanelID, State (raw or flat), file path

import os
import pandas as pd
import re

def build_df_from_data(data_types=("raw", "flat"), base_dir="../data"):
    """
    Builds a metadata DataFrame from the data folder.

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

    return df

def build_df_from_new_data(base_dir="../data_new"):
    """
    Builds a metadata DataFrame from the data_new folder.

    Expected filename pattern example:
        {Leveling-1}P{Chase}_D{6.2.2026}_type(STD)_GS{Rob2BCA-WPWU}_State{FLAT}_0degrees_flat (2).png

    Extracts:
        - LevelingScore (from folder name, fallback to filename)
        - Person (from P{...})
        - DateCollected (from D{...})
        - ImageType (from type(...))
        - GSCamera (from GS{...})
        - State (from State{...}, fallback to raw/flat in filename)
        - FilePath

    PanelID is not present in new data and is always set to None.
    """

    records = []

    for label_folder in sorted(os.listdir(base_dir), key=lambda x: int(x)):
        folder_path = os.path.join(base_dir, label_folder)

        if not os.path.isdir(folder_path):
            continue

        level_from_folder = int(label_folder)

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(".png"):
                continue

            filepath = os.path.join(folder_path, filename)

            level_match = re.search(r"Leveling-(\d+)", filename, flags=re.IGNORECASE)
            person_match = re.search(r"P\{([^}]+)\}", filename)
            date_match = re.search(r"_D\{([^}]+)\}", filename)
            image_type_match = re.search(r"_type\(([^)]+)\)", filename, flags=re.IGNORECASE)
            gs_match = re.search(r"_GS\{([^}]+)\}", filename)
            state_match = re.search(r"_State\{([^}]+)\}", filename, flags=re.IGNORECASE)

            level = int(level_match.group(1)) if level_match else level_from_folder
            person = person_match.group(1) if person_match else None
            date_collected = date_match.group(1) if date_match else None
            image_type = image_type_match.group(1) if image_type_match else None
            gs_camera = gs_match.group(1) if gs_match else None

            if state_match:
                state = state_match.group(1).lower()
            else:
                fallback_state_match = re.search(r"(raw|flat)", filename, flags=re.IGNORECASE)
                state = fallback_state_match.group(1).lower() if fallback_state_match else None

            record = {
                "LevelingScore": level,
                "Person": person,
                "DateCollected": date_collected,
                "ImageType": image_type,
                "GSCamera": gs_camera,
                "PanelID": None,
                "State": state,
                "FilePath": filepath,
            }

            records.append(record)

    df = pd.DataFrame(records)

    return df
