# Gel Site Mini Leveling CNN Model

## Overview

This project trains a CNN to predict paint leveling score from Gel Site Mini images.

The Surface Vision system gives both raw and flat images. This model is built to use flat images.

Pipeline in plain terms:
- read image folders
- build metadata.csv from file names and paths
- load and preprocess images
- train and evaluate the model
- export an ONNX model for frontend use

## Tech Stack

- Python
- PyTorch and TorchVision
- NumPy and pandas
- OpenCV
- ONNX export with torch.onnx
- Optional ONNX Runtime for inference

## Installation (Windows PowerShell)

1. Clone the repository

```powershell
git clone https://github.com/bxb370/Gel_Site_Mini_Flow_And_Leveling_Model.git
cd Gel_Site_Mini_Flow_And_Leveling_Model
```

2. Confirm Python version

This project is currently run with Python 3.14.5.

```powershell
python --version
```

3. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas pillow opencv-python torch torchvision onnx onnxruntime pytest
```

## Data Needed in Project Root

Default run expects these folders:
- data
- data_new
- data_new_2
- data_test
- human_ratings

If you are using original shared data, copy those folders from:
Sherwin-Williams\Gelsight - General\Gelsight Mini\Leveling Model Python Script Images

## Run the Pipeline

```powershell
python main.py
```

Outputs:
- metadata.csv in project root
- waveletcnn_v2.onnx in models

## Metadata Columns

metadata.csv uses these fields:

LevelingScore, Person, DateCollected, ImageType, GSCamera, PanelID, State, FilePath

Field format notes:
- DateCollected: month.day.year (example: 7.9.2026)
- ImageType: DD or STD
- DD means drawdown
- STD means standard card
- GSCamera: camera code format like Rob2BCA-WPWU
- PanelID: 25-xxx format (example: 25-101)
- State: raw or flat

## Project Structure

- human_ratings: human panel scores used for relabeling DD data
- models: trained checkpoints and ONNX exports
- notebooks: exploration and experiments
- src: data loading, metadata building, training, and evaluation code
- main.py: end-to-end pipeline entry point
- metadata.csv: generated metadata table

## Updating with New Data

To add data, place it in a folder in the project root and extend metadata-building code so rows are appended to metadata.csv using the same columns listed above.

If you want to run only your new dataset, skip the default metadata loading in main.py and build your own metadata.csv with the same schema.

## Owner/Contact

- Author: Brooke Brocker (Data Science Co-op, 2026)
- Mentor: Stacy Conte