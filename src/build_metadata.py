# function to take the data and build the metadata table.
# Following fields are generated:

# {leveling-1}_P{person}_D{date}_type{STD or DD}_GS{#}_Panel-GS-25-{ID}_state{raw or flat}
# LevelingScore, Person, DateCollected, ImageType (STD or DD), GSCamera, PanelID, State (raw or flat), file path

import os
import pandas as pd
import re


METADATA_COLUMNS = [
    "LevelingScore",
    "Person",
    "DateCollected",
    "ImageType",
    "GSCamera",
    "PanelID",
    "State",
    "FilePath",
]

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
        - State (from the filename suffix: `_flat` means flat, otherwise raw)
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

            level = int(level_match.group(1)) if level_match else level_from_folder
            person = person_match.group(1) if person_match else None
            date_collected = date_match.group(1) if date_match else None
            image_type = image_type_match.group(1) if image_type_match else None
            gs_camera = gs_match.group(1) if gs_match else None

            filename_lower = filename.lower()
            if re.search(r"_flat(?:\s*\(\d+\))?\.png$", filename_lower):
                state = "flat"
            elif re.search(r"\.png$", filename_lower):
                state = "raw"
            else:
                state = None

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


def _normalize_metadata_df(df):
    """Return a metadata frame with the expected column order."""

    return df.reindex(columns=METADATA_COLUMNS)


def load_metadata(
    data_base_dir="../data",
    new_data_base_dir="../data_new",
    output_file="metadata.csv",
):
    """
    Load the two metadata sources, verify they share the same schema,
    combine them, and write the result to metadata.csv.
    """

    old_data_df = _normalize_metadata_df(build_df_from_data(base_dir=data_base_dir))
    new_data_df = _normalize_metadata_df(build_df_from_new_data(base_dir=new_data_base_dir))

    if list(old_data_df.columns) != list(new_data_df.columns):
        raise ValueError("Metadata frames are not compatible and cannot be combined.")

    metadata_df = pd.concat([old_data_df, new_data_df], ignore_index=True)

    output_path = os.path.join(data_base_dir, output_file)
    metadata_df.to_csv(output_path, index=False)

    print(f"Saved metadata to {output_path}")
    print(f"Combined records: {len(metadata_df)}")

    return metadata_df


