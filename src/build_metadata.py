"""This file contains functions that handle loading the data from the metadata table"""

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


def _normalize_panel_id(panel_value):
    """Normalize panel IDs by removing only a leading GS- prefix."""

    if panel_value is None:
        return None

    panel_str = str(panel_value).strip()
    if panel_str == "":
        return None

    # Remove leading GS- if present, case-insensitive.
    return re.sub(r"^GS-", "", panel_str, flags=re.IGNORECASE)

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
                panel_id = _normalize_panel_id(panel_match.group(0) if panel_match else None)

                # ---- Extract State (fallback to folder name if needed) ----
                state_match = re.search(r"(raw|flat)", filename.lower())
                state = state_match.group(1) if state_match else data_type

                # ---- Build record ----
                record = {
                    "LevelingScore": level,
                    "Person": None,
                    "DateCollected": None,
                    "ImageType": "DD",
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
            date_collected = _normalize_date_collected(
                date_match.group(1) if date_match else None
            )
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


def build_df_from_new_data_2(base_dir="../data_new_2"):
    """
    Builds a metadata DataFrame from the data_new_2 folder.

    Expected filename pattern example:
        {leveling-1}_P{Nicole}_D{7.9.2026}_type{DD}_GS{2BDR-9F02}_Panel{GS-25-108}_Gloss{Flat}_0degrees_flat.png

    Extracts:
        - LevelingScore (from folder name, fallback to filename)
        - Person (from P{...})
        - DateCollected (from D{...})
        - ImageType (from type{...})
        - GSCamera (from GS{...})
        - PanelID (from Panel{...}, unless it is the placeholder "ID")
        - State (from the filename suffix: `_flat` means flat, otherwise raw)
        - FilePath
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

            level_match = re.search(r"leveling-(\d+)", filename, flags=re.IGNORECASE)
            person_match = re.search(r"_P\{([^}]+)\}", filename)
            date_match = re.search(r"_D\{([^}]+)\}", filename)
            image_type_match = re.search(r"_type\{([^}]+)\}", filename, flags=re.IGNORECASE)
            gs_match = re.search(r"_GS\{([^}]+)\}", filename)
            panel_match = re.search(r"_Panel\{([^}]+)\}", filename)

            level = int(level_match.group(1)) if level_match else level_from_folder
            person = person_match.group(1) if person_match else None
            date_collected = _normalize_date_collected(
                date_match.group(1) if date_match else None
            )
            image_type = image_type_match.group(1) if image_type_match else None

            # Extract GS and Panel values
            gs_value = gs_match.group(1) if gs_match else None
            panel_value = panel_match.group(1) if panel_match else None

            # Determine panel_id and gs_camera
            # GS might contain either a camera code or a panel ID (25-###)
            # Panel might contain a panel ID (GS-25-###) or "ID" (placeholder)
            
            # If GS contains a panel ID pattern, extract it as panel and set camera to None
            if gs_value and "25-" in gs_value:
                panel_id = _normalize_panel_id(gs_value)
                gs_camera = None
            # If Panel is not "ID", use it as the panel and GS as camera
            elif panel_value and panel_value.upper() != "ID":
                panel_id = _normalize_panel_id(panel_value)
                gs_camera = gs_value
            # Otherwise, no valid panel found
            else:
                panel_id = None
                gs_camera = gs_value

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
                "PanelID": panel_id,
                "State": state,
                "FilePath": filepath,
            }

            records.append(record)

    df = pd.DataFrame(records)

    return df


def build_df_from_data_test(base_dir="../data_test"):
    """
    Builds a metadata DataFrame from the data_test folder.

    Expected filename pattern example:
        {leveling-1.8}_P{Brooke}_D{7.21.26}_type{TEST}_GS{2BDR-9F02}_Panel{GS-25-102}_0degrees_flat.png

    Extracts:
        - LevelingScore (from {leveling-x.x})
        - Person (from P{...})
        - DateCollected (from D{...})
        - ImageType (from type{...})
        - GSCamera (from GS{...})
        - PanelID (from Panel{...}, unless it is "ID")
        - State (from filename suffix: `_flat` means flat, otherwise raw)
        - FilePath
    """

    records = []

    if not os.path.isdir(base_dir):
        return pd.DataFrame(columns=METADATA_COLUMNS)

    for filename in os.listdir(base_dir):
        if not filename.lower().endswith(".png"):
            continue

        filepath = os.path.join(base_dir, filename)
        if not os.path.isfile(filepath):
            continue

        level_match = re.search(r"\{leveling-([0-9]+(?:\.[0-9]+)?)\}", filename, flags=re.IGNORECASE)
        person_match = re.search(r"_P\{([^}]+)\}", filename)
        date_match = re.search(r"_D\{([^}]+)\}", filename)
        image_type_match = re.search(r"_type\{([^}]+)\}", filename, flags=re.IGNORECASE)
        gs_match = re.search(r"_GS\{([^}]+)\}", filename)
        panel_match = re.search(r"_Panel\{([^}]+)\}", filename)

        level = float(level_match.group(1)) if level_match else None
        person = person_match.group(1) if person_match else None
        date_collected = _normalize_date_collected(
            date_match.group(1) if date_match else None
        )
        image_type = image_type_match.group(1) if image_type_match else None

        # Use GS as panel when it contains 25-###, otherwise use Panel unless placeholder "ID".
        gs_value = gs_match.group(1) if gs_match else None
        panel_value = panel_match.group(1) if panel_match else None
        if gs_value and "25-" in gs_value:
            panel_id = _normalize_panel_id(gs_value)
            gs_camera = None
        elif panel_value and panel_value.upper() != "ID":
            panel_id = _normalize_panel_id(panel_value)
            gs_camera = gs_value
        else:
            panel_id = None
            gs_camera = gs_value

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
            "PanelID": panel_id,
            "State": state,
            "FilePath": filepath,
        }

        records.append(record)

    df = pd.DataFrame(records)

    return df


def _normalize_metadata_df(df):
    """Return a metadata frame with the expected column order."""

    return df.reindex(columns=METADATA_COLUMNS)

def _normalize_date_collected(date_value, default_year="2026"):
    """Normalize dates to month.day.year when enough pieces are available."""

    if date_value is None:
        return None

    date_parts = [part for part in re.split(r"[^0-9]+", str(date_value)) if part]
    if len(date_parts) == 2:
        date_parts.append(default_year)
    if len(date_parts) == 3:
        return ".".join(date_parts)

    return date_value


def _apply_human_ratings(metadata_df, human_ratings_path):
    """
    Replace LevelingScore with AverageScore from HumanRatings.csv
    for all rows whose PanelID matches the pattern 25-xxx, where xxx == Label.

    Only DD rows are relabelled; STD rows keep their original integer scores.
    """

    if not os.path.exists(human_ratings_path):
        print(f"Warning: HumanRatings.csv not found at {human_ratings_path}, skipping relabelling.")
        return metadata_df

    human_ratings = pd.read_csv(human_ratings_path)

    # Find the AverageScore column (case-insensitive, strip whitespace).
    avg_col = next(
        (c for c in human_ratings.columns if c.strip().lower() == "averagescore"),
        None,
    )

    if avg_col is None:
        print("Warning: AverageScore column not found in HumanRatings.csv, skipping relabelling.")
        return metadata_df

    avg_scores = pd.to_numeric(human_ratings[avg_col], errors="coerce")

    human_ratings = human_ratings.copy()
    human_ratings["_AverageScore"] = avg_scores
    part_to_mode = dict(
        zip(human_ratings["Label"].astype(int), human_ratings["_AverageScore"])
    )

    def extract_part_number(panel_id):
        if pd.isna(panel_id):
            return None
        match = re.search(r"25-(\d+)", str(panel_id))
        return int(match.group(1)) if match else None

    metadata_df = metadata_df.copy()
    metadata_df["LevelingScore"] = metadata_df["LevelingScore"].astype(float)

    dd_mask = metadata_df["ImageType"].str.upper() == "DD"
    part_nums = metadata_df.loc[dd_mask, "PanelID"].apply(extract_part_number)
    new_scores = part_nums.map(part_to_mode)
    matched = new_scores.notna()

    metadata_df.loc[dd_mask & matched, "LevelingScore"] = new_scores[matched]

    updated = int((dd_mask & matched).sum())
    print(f"Applied human ratings to {updated} DD rows.")

    return metadata_df


def load_metadata(
    data_base_dir="../data",
    new_data_base_dir="../data_new",
    new_data_2_base_dir="../data_new_2",
    data_test_base_dir="../data_test",
    output_file="metadata.csv",
    human_ratings_path=None,
):
    """
    Load the metadata sources, verify they share the same schema,
    combine them, apply human ratings relabelling, and write to metadata.csv.

    Parameters:
        human_ratings_path: path to HumanRatings.csv. Defaults to
            <data_base_dir>/../human_ratings/HumanRatings.csv.
    """

    metadata_frames = [
        _normalize_metadata_df(build_df_from_data(base_dir=data_base_dir)),
        _normalize_metadata_df(build_df_from_new_data(base_dir=new_data_base_dir)),
    ]

    if new_data_2_base_dir is not None:
        metadata_frames.append(
            _normalize_metadata_df(build_df_from_new_data_2(base_dir=new_data_2_base_dir))
        )

    if data_test_base_dir is not None:
        metadata_frames.append(
            _normalize_metadata_df(build_df_from_data_test(base_dir=data_test_base_dir))
        )

    expected_columns = list(metadata_frames[0].columns)
    for metadata_frame in metadata_frames[1:]:
        if list(metadata_frame.columns) != expected_columns:
            raise ValueError("Metadata frames are not compatible and cannot be combined.")

    metadata_df = pd.concat(metadata_frames, ignore_index=True)

    # Apply human ratings — default path is ../human_ratings/HumanRatings.csv
    if human_ratings_path is None:
        human_ratings_path = os.path.join(data_base_dir, "..", "human_ratings", "HumanRatings.csv")
    human_ratings_path = os.path.normpath(human_ratings_path)
    metadata_df = _apply_human_ratings(metadata_df, human_ratings_path)

    project_root = os.path.normpath(os.path.join(data_base_dir, ".."))
    output_path = os.path.join(project_root, output_file)
    metadata_df.to_csv(output_path, index=False)

    print(f"Saved metadata to {output_path}")
    print(f"Combined records: {len(metadata_df)}")

    return metadata_df